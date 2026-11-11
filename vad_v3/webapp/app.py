"""Local HTTP service; inference is owned by the independent queue worker."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException
from vad_v3.webapp import experiments
from vad_v3.webapp.settings import (V3_DIR, ENV_FILE, DEFAULT_MODEL, DEFAULT_MCTS_MODE,
    DEFAULT_MCTS_ACCEPTANCE_POLICY, CATEGORIES,
    load_environment, effective_config, code_hash, file_hash, redact)
from vad_v3.webapp.store import Store, Conflict
from vad_v3.webapp.results import analysis, curves, export_csv, export_html
from vad_v3.webapp.migration import import_legacy
from vad_v3.webapp.diagnostics import check_environment
VIDEO_SUFFIXES={'.mp4','.webm','.mov','.m4v','.avi','.mkv'}


def probe_video(path):
    from evaluation_tools.video_fps import probe_video_metadata
    metadata = probe_video_metadata(path)
    # 上传临时路径不写入视频记录，后续以数据库里的正式文件路径为准。
    metadata.pop('video_path')
    return metadata


def create_app(data_dir=None,max_upload_mb=2048,timeout=3600,migrate=True):
    load_environment(ENV_FILE)
    app=Flask(__name__,static_folder='static',static_url_path='/static')
    app.json.ensure_ascii=False
    app.config.update(MAX_CONTENT_LENGTH=max_upload_mb*1024*1024,VLM_MODEL=os.getenv('DASHSCOPE_MODEL') or DEFAULT_MODEL,
                      TRUSTED_HOSTS=['127.0.0.1','localhost'])
    store=Store(data_dir or V3_DIR/'web_data')
    app.extensions['store']=store
    executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='vad-evaluation')
    app.extensions['executor']=executor
    app.extensions['migration']=import_legacy(store) if migrate else None
    with store.connect() as db:
        db.execute("UPDATE evaluations SET status='failed',error='Web 服务重启，请重新评价' WHERE status IN ('queued','running')")

    @app.before_request
    def origin_check():
        if request.method in {'POST','PUT','PATCH','DELETE'}:
            origin=request.headers.get('Origin')
            if origin and origin!=request.host_url.rstrip('/'):
                return jsonify(error='请从本地系统页面提交请求。'),403

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=f'上传超过 {max_upload_mb} MB 限制' if exc.code==413 else exc.description),exc.code

    @app.errorhandler(FileNotFoundError)
    def missing(exc):
        return jsonify(error=str(exc)),404

    @app.errorhandler(Conflict)
    def conflict(exc):
        return jsonify(error=str(exc)),409

    @app.errorhandler(ValueError)
    def invalid(exc):
        return jsonify(error=redact(str(exc))),400

    @app.get('/')
    def index():
        return app.send_static_file('index.html')

    @app.get('/api/config')
    def config():
        return jsonify(api_key_configured=bool(os.getenv('DASHSCOPE_API_KEY')),
            model=app.config['VLM_MODEL'],max_upload_mb=max_upload_mb,categories=CATEGORIES,
            defaults=dict(device='cuda',mcts_mode=DEFAULT_MCTS_MODE,
                          mcts_acceptance_policy=DEFAULT_MCTS_ACCEPTANCE_POLICY),protocols=experiments.PROTOCOLS)

    @app.get('/api/health')
    def health():
        worker=store.service('worker')
        online=bool(worker and time.time()-worker['updated_at']<5 and worker['payload']['state']!='stopped')
        return jsonify(web='ready',worker=worker,worker_online=online,diagnostics=store.service('diagnostics'))

    @app.post('/api/system/check')
    def environment_check():
        executor.submit(lambda:store.service('diagnostics',check_environment()))
        return jsonify(status='queued'),202

    def page():
        value=int(request.args.get('page',1))
        if not 1<=value<=100000: raise ValueError('页码无效。')
        return value

    def upload_video():
        upload=request.files.get('video')
        if not upload or not upload.filename: raise ValueError('请选择视频文件。')
        filename=upload.filename.replace('\\','/').rsplit('/',1)[-1]
        suffix=Path(filename).suffix.lower()
        if suffix not in VIDEO_SUFFIXES: raise ValueError('支持 MP4、WebM、MOV、M4V、AVI、MKV。')
        vid=uuid.uuid4().hex; relative=f'videos/{vid}/input{suffix}'
        path=store.path(relative); path.parent.mkdir(parents=True,exist_ok=True)
        try:
            upload.save(path)
            metadata=probe_video(path); metadata['size_bytes']=path.stat().st_size
            return store.add_video(vid,filename,relative,file_hash(path),metadata)
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    @app.post('/api/videos')
    def video_upload():
        return jsonify(upload_video()),201

    @app.get('/api/videos')
    def video_list():
        return jsonify(videos=store.videos(request.args.get('q',''),page(),request.args.get('deleted')=='1'))

    @app.get('/api/videos/<vid>/stream')
    def video_stream(vid):
        return send_file(store.path(store.video(vid,deleted=True)['path']),conditional=True)

    @app.delete('/api/videos/<vid>')
    def video_delete(vid):
        store.delete_video(vid)
        return jsonify(deleted=True,id=vid)

    @app.post('/api/videos/<vid>/restore')
    def video_restore(vid):
        store.delete_video(vid,restore=True)
        return jsonify(store.video(vid))

    def public_job(job):
        video=store.video(job['video_id'],deleted=True)
        return dict(job,filename=video['filename'],duration=video['metadata'].get('duration'),
            fps=video['metadata'].get('fps'),device=job['config']['device'],
            mcts_mode=job['config']['mcts_mode'],model=job['config']['model'])

    @app.post('/api/jobs')
    def submit():
        if not os.getenv('DASHSCOPE_API_KEY'):
            return jsonify(error='未配置 DASHSCOPE_API_KEY，请配置 webapp/.env 后重启 Web 和 Worker。'),503
        if request.is_json:
            data=request.get_json()
            if not isinstance(data,dict):raise ValueError('任务请求必须是 JSON 对象。')
            vid=data.get('video_id'); options=data.get('config',{})
        else:
            vid=upload_video()['id']
            options={k:request.form[k] for k in ('device','mcts_mode','context') if k in request.form}
            options['full_classification']=request.form.get('full_classification')=='true'
        cfg=effective_config(options,timeout,app.config['VLM_MODEL'])
        job,created=store.create_job(vid,cfg,code_hash(),request.headers.get('Idempotency-Key'))
        return jsonify(public_job(job)),202 if created else 200

    @app.get('/api/jobs')
    def jobs():
        return jsonify(jobs=[public_job(j) for j in store.jobs(request.args.get('status',''),
            request.args.get('q',''),page(),request.args.get('deleted')=='1',request.args.get('video_id',''))])

    @app.get('/api/jobs/<jid>')
    def job_detail(jid):
        job=public_job(store.job(jid)); job['attempts']=store.attempts(jid)
        if job['result_attempt_id']: job['result']=store.result(jid)
        if job['current_attempt_id']:
            a=store.attempt(job['current_attempt_id']); path=store.path(a['artifact_dir']+'/run.log')
            if path.is_file():
                with path.open('rb') as stream:
                    stream.seek(max(0,path.stat().st_size-16000))
                    job['log']=redact(stream.read().decode('utf-8',errors='replace'))
        return jsonify(job)

    @app.post('/api/jobs/<jid>/cancel')
    def job_cancel(jid):
        store.cancel(jid); return jsonify(public_job(store.job(jid)))

    @app.post('/api/jobs/<jid>/retry')
    def job_retry(jid):
        store.retry(jid,code_hash()); return jsonify(public_job(store.job(jid))),202

    @app.delete('/api/jobs/<jid>')
    def job_delete(jid):
        store.delete_job(jid); return jsonify(id=jid,deleted=True)

    @app.post('/api/jobs/<jid>/restore')
    def job_restore(jid):
        store.delete_job(jid,restore=True); return jsonify(public_job(store.job(jid)))

    @app.get('/api/jobs/<jid>/video')
    def job_video(jid):
        return video_stream(store.job(jid)['video_id'])

    @app.get('/api/jobs/<jid>/result')
    def job_result(jid):
        job=store.job(jid); store.result(jid)
        a=store.attempt(job['result_attempt_id'])
        return send_file(store.path(a['artifact_dir']+'/result.json'),as_attachment=True,download_name=f'{jid}.result.json')

    @app.get('/api/jobs/<jid>/analysis')
    def job_analysis(jid):
        return jsonify(analysis(store,jid))

    @app.get('/api/jobs/<jid>/curves')
    def job_curves(jid):
        return jsonify(curves=curves(store,jid))

    @app.get('/api/jobs/<jid>/events')
    def events(jid):
        job=store.job(jid)
        if not job['current_attempt_id']: return jsonify(events=[])
        a=store.attempt(job['current_attempt_id']); path=store.path(a['artifact_dir']+'/stage_events.jsonl')
        after=int(request.args.get('after',0)); values=[]
        if path.exists():
            for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
                try: row=json.loads(line)
                except ValueError: continue
                if row['sequence']>after: values.append(row)
        return jsonify(events=values[:300] if 'after' in request.args else values[-300:])

    @app.get('/api/jobs/<jid>/evidence/<name>')
    def evidence(jid,name):
        data=analysis(store,jid)
        registered={Path(i['file']).name:i['file'] for r in data['evidence'] for i in r['images']}
        if name not in registered: raise FileNotFoundError('证据图片未登记。')
        a=store.attempt(data['attempt_id'])
        return send_file(store.path(a['artifact_dir']+'/'+registered[name]),conditional=True)

    @app.get('/api/jobs/<jid>/mcts')
    def mcts(jid):
        return jsonify(analysis(store,jid)['mcts'])

    @app.post('/api/jobs/<jid>/reviews')
    def reviews(jid):
        return jsonify(store.add_review(jid,request.get_json())),201

    @app.get('/api/jobs/<jid>/export')
    def export(jid):
        data=analysis(store,jid); name=store.video(store.job(jid)['video_id'],deleted=True)['filename']
        fmt=request.args.get('format','html')
        if fmt=='csv': content=export_csv(store.result(jid),data['reviews']); mime='text/csv'
        elif fmt=='html': content=export_html(data,name); mime='text/html'
        else: raise ValueError('支持 HTML 和 CSV 报告。')
        return send_file(io.BytesIO(content.encode('utf-8')),mimetype=mime,as_attachment=True,
                         download_name=f'{jid}.report.{fmt}')

    @app.get('/api/experiments')
    def experiment_list():
        return jsonify(experiments=experiments.list_experiments(store))

    @app.post('/api/experiments')
    def experiment_import():
        if request.is_json: bundle=request.get_json()
        else:
            upload=request.files.get('bundle')
            if not upload: raise ValueError('请选择实验 JSON。')
            bundle=json.load(upload.stream)
        return jsonify(experiments.create_experiment(store,bundle)),201

    @app.post('/api/experiments/import-history')
    def import_history():
        return jsonify(experiments.create_experiment(store,experiments.historical_bundle())),201

    @app.post('/api/experiments/from-jobs')
    def experiment_from_jobs():
        return jsonify(experiments.from_jobs(store,request.get_json())),201

    @app.post('/api/experiments/<eid>/evaluate')
    def experiment_evaluate(eid):
        data=request.get_json(silent=True) or {}
        if not isinstance(data,dict): raise ValueError('评价请求必须是 JSON 对象。')
        rid=experiments.create_evaluation(store,eid,data.get('protocol','vadclip_native_fps'))
        executor.submit(experiments.evaluate,store,rid)
        return jsonify(id=rid,status='queued'),202

    @app.get('/api/evaluations/<rid>')
    def evaluation(rid):
        with store.connect() as db:
            record=store.unpack(db.execute('SELECT * FROM evaluations WHERE id=?',(rid,)).fetchone())
        if record['status']=='completed':
            record['metrics']=json.loads(store.path(record['metrics_path']).read_text(encoding='utf-8'))
        return jsonify(record)

    @app.get('/api/evaluations/<rid>/download')
    def evaluation_download(rid):
        with store.connect() as db:
            record=store.unpack(db.execute('SELECT * FROM evaluations WHERE id=?',(rid,)).fetchone())
        if record['status']!='completed': raise Conflict('评价尚未完成。')
        path=store.path(record['metrics_path'])
        if request.args.get('format')=='csv':
            content=experiments.metrics_csv(json.loads(path.read_text(encoding='utf-8')))
            return send_file(io.BytesIO(content.encode('utf-8')),mimetype='text/csv',
                as_attachment=True,download_name=f'{rid}-metrics.csv')
        if request.args.get('audit')=='1': path=path.with_name('matching_audit.json')
        return send_file(path,as_attachment=True,download_name=f'{rid}-{path.name}')
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='V3 视频异常分析系统 Web 服务')
    parser.add_argument('--port',type=int,default=7860)
    parser.add_argument('--data-dir',type=Path,default=V3_DIR/'web_data')
    parser.add_argument('--max-upload-mb',type=int,default=2048)
    parser.add_argument('--timeout',type=int,default=3600)
    args=parser.parse_args()
    from waitress import serve
    from vad_v3.webapp.worker import InstanceLock
    with InstanceLock(args.data_dir/'.web.lock'):
        application=create_app(args.data_dir,args.max_upload_mb,args.timeout)
        print(f'V3 视频异常分析系统：http://127.0.0.1:{args.port}',flush=True)
        try:
            serve(application,host='127.0.0.1',port=args.port,threads=6,
                  max_request_body_size=args.max_upload_mb*1024*1024)
        finally:
            application.extensions['executor'].shutdown(wait=True)
