"""Local-only upload, background detection and seekable video serving.

Run with the same Python environment as v3: python vad_v3/webapp/app.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

from flask import Flask, jsonify, request, send_file
from dotenv import load_dotenv
from werkzeug.exceptions import HTTPException

V3_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = Path(__file__).resolve().with_name('.env')
# qwen-vl-max was retired on 2026-10-10. Keep the historical experiment
# CLI defaults intact; the web application explicitly selects its model.
DEFAULT_MODEL = 'qwen3-vl-plus'
VIDEO_SUFFIXES = {'.mp4', '.webm', '.mov', '.m4v', '.avi', '.mkv'}
ACTIVE_STATES = {'uploading', 'running'}


def probe_video(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        readable, _ = capture.read()
        duration = frames / fps if fps > 0 else 0
        if not readable or not math.isfinite(duration) or duration <= 0:
            raise ValueError('无法读取视频，请上传有效的视频文件。')
        return {'duration': duration, 'fps': fps}
    finally:
        capture.release()


def build_command(job, directory):
    """Invoke the actual CLI; all unspecified model settings stay at v3 defaults."""
    command = [sys.executable, '-u', str(V3_DIR / 'video_anomaly_detection' / 'run.py'),
               '--video_path', str(directory / job['video_file']),
               '--output_file', str(directory / 'result.json'),
               '--device', job['device'], '--mcts_mode', job['mcts_mode'],
               '--model', job.get('model', DEFAULT_MODEL),
               '--context', job['context']]
    if job['full_classification']:
        command.append('--no_classify_skip_class_agnostic')
    return command


class JobManager:
    def __init__(self, root, timeout):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.lock = threading.RLock()
        self.active_id = None
        self.process = None
        self.closed = False
        # Incomplete jobs from a previous server process cannot be resumed safely.
        for path in self.root.glob('*/job.json'):
            try:
                job = json.loads(path.read_text(encoding='utf-8'))
                if job['status'] in ACTIVE_STATES:
                    job.update(status='failed', error='上次服务已中断，请重新提交检测。')
                    self.save(job)
            except (OSError, ValueError, KeyError):
                continue

    def directory(self, job_id):
        if not re.fullmatch(r'[0-9a-f]{32}', job_id):
            raise FileNotFoundError('任务不存在。')
        directory = self.root / job_id
        if directory.resolve() != directory:
            raise FileNotFoundError('任务目录无效。')
        return directory

    def trash_directory(self, job_id):
        self.directory(job_id)  # Validate the ID and the active destination.
        directory = self.root / '.trash' / job_id
        if directory.resolve() != directory:
            raise FileNotFoundError('任务回收目录无效。')
        return directory

    def delete(self, job_id):
        """Move one finished job into a recoverable, local trash directory."""
        with self.lock:
            job = self.read(job_id)
            if job_id == self.active_id or job['status'] in ACTIVE_STATES:
                raise ValueError('正在运行的任务不能删除，请等待检测完成。')
            source = self.directory(job_id)
            destination = self.trash_directory(job_id)
            if destination.exists():
                raise ValueError('回收目录已存在同名任务，无法覆盖。')
            destination.parent.mkdir(exist_ok=True)
            # Both resolved absolute paths were checked to stay within root.
            source.rename(destination)

    def restore(self, job_id):
        with self.lock:
            source = self.trash_directory(job_id)
            destination = self.directory(job_id)
            if not (source / 'job.json').is_file():
                raise FileNotFoundError('回收目录中没有这个任务。')
            if destination.exists():
                raise ValueError('任务已存在，无法重复恢复。')
            source.rename(destination)
            return self.read(job_id)

    def save(self, job):
        with self.lock:
            directory = self.directory(job['id'])
            directory.mkdir(exist_ok=True)
            temporary = directory / 'job.tmp'
            temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(directory / 'job.json')

    def read(self, job_id):
        with self.lock:
            return json.loads((self.directory(job_id) / 'job.json').read_text(encoding='utf-8'))

    def list_jobs(self):
        with self.lock:
            jobs = []
            for path in self.root.glob('*/job.json'):
                try:
                    jobs.append(self.read(path.parent.name))
                except (OSError, ValueError):
                    continue
            return sorted(jobs, key=lambda job: job['created_at'], reverse=True)[:30]

    def start(self, job):
        threading.Thread(target=self._run, args=(job,), daemon=True).start()

    def _run(self, job):
        directory = self.directory(job['id'])
        try:
            job.update(status='running', started_at=time.time())
            self.save(job)
            environment = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
            with (directory / 'run.log').open('w', encoding='utf-8') as log:
                with self.lock:
                    if self.closed:
                        raise RuntimeError('服务已停止。')
                    self.process = subprocess.Popen(
                        build_command(job, directory), cwd=str(V3_DIR.parent), env=environment,
                        stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    process = self.process
                try:
                    code = process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError(f'检测超过 {self.timeout} 秒，已终止。可通过 --timeout 调整。')
            if code:
                raise RuntimeError(f'v3 检测失败（退出码 {code}），请查看运行日志。')
            result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            if not isinstance(result.get('anomalies'), list) or 'duration' not in result:
                raise RuntimeError('v3 未生成有效的检测结果。')
            job.update(status='completed', finished_at=time.time(),
                       anomaly_count=len(result['anomalies']),
                       processing_time=result.get('processing_time'))
        except Exception as exc:
            job.update(status='failed', finished_at=time.time(), error=str(exc))
        finally:
            try:
                self.save(job)
            finally:
                with self.lock:
                    self.active_id = None
                    self.process = None

    def close(self):
        with self.lock:
            self.closed = True
            process = self.process
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()


def create_app(data_dir=None, max_upload_mb=2048, timeout=3600):
    # Resolve relative to this file, not the launch directory. Windows editors
    # may write a UTF-8 BOM; existing process environment settings take priority.
    load_dotenv(ENV_FILE, override=False, encoding='utf-8-sig')
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    app.config['MAX_CONTENT_LENGTH'] = max_upload_mb * 1024 * 1024
    app.config['VLM_MODEL'] = (os.environ.get('DASHSCOPE_MODEL') or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    app.json.ensure_ascii = False
    manager = JobManager(data_dir or V3_DIR / 'web_data', timeout)
    app.extensions['jobs'] = manager

    @app.before_request
    def check_origin():
        if request.method in {'POST', 'DELETE'}:
            origin = request.headers.get('Origin')
            if origin and origin != request.host_url.rstrip('/'):
                return jsonify(error='请从本地界面提交请求。'), 403

    @app.errorhandler(HTTPException)
    def http_error(exc):
        message = ('文件过大，上传上限为 %s MB。' % max_upload_mb
                   if exc.code == 413 else exc.description)
        return jsonify(error=message), exc.code

    @app.get('/')
    def index():
        return app.send_static_file('index.html')

    @app.get('/api/config')
    def config():
        return jsonify(api_key_configured=bool(os.environ.get('DASHSCOPE_API_KEY')),
                       model=app.config['VLM_MODEL'],
                       max_upload_mb=max_upload_mb)

    @app.get('/api/jobs')
    def jobs():
        return jsonify(jobs=manager.list_jobs())

    @app.post('/api/jobs')
    def submit():
        if not os.environ.get('DASHSCOPE_API_KEY'):
            return jsonify(error='服务端未配置 DASHSCOPE_API_KEY。请设置环境变量或 webapp/.env 后重启服务。'), 503
        job_id = uuid.uuid4().hex
        with manager.lock:
            if manager.active_id or manager.closed:
                return jsonify(error='已有检测任务正在运行，请等待完成后再提交。'), 409
            manager.active_id = job_id
        launched = False
        job = None
        try:
            upload = request.files.get('video')
            if upload is None or not upload.filename:
                return jsonify(error='请选择一个视频文件。'), 400
            filename = upload.filename.replace('\\', '/').rsplit('/', 1)[-1]
            suffix = Path(filename).suffix.lower()
            if suffix not in VIDEO_SUFFIXES:
                return jsonify(error='支持 MP4、WebM、MOV、M4V、AVI、MKV 视频。'), 400
            device = request.form.get('device', 'cuda')
            mode = request.form.get('mcts_mode', 'off')
            context = request.form.get('context', 'surveillance camera').strip()
            if device not in {'cpu', 'cuda', 'mps'} or mode not in {'off', 'shadow', 'refine'}:
                return jsonify(error='无效的检测配置。'), 400
            if len(context) > 1000:
                return jsonify(error='场景描述不能超过 1000 字符。'), 400
            job = dict(id=job_id, filename=filename, video_file='input' + suffix,
                       status='uploading', created_at=time.time(), device=device,
                       mcts_mode=mode, context=context,
                       model=app.config['VLM_MODEL'],
                       full_classification=request.form.get('full_classification') == 'true')
            manager.save(job)
            path = manager.directory(job_id) / job['video_file']
            upload.save(path)
            metadata = probe_video(path)
            job.update(metadata)
            job['status'] = 'running'
            manager.save(job)
            manager.start(job)
            launched = True
            return jsonify(job), 202
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        finally:
            if not launched:
                try:
                    if job is not None:
                        job.update(status='failed', error='上传或视频校验失败，请重新提交。')
                        manager.save(job)
                        (manager.directory(job_id) / job['video_file']).unlink(missing_ok=True)
                finally:
                    with manager.lock:
                        manager.active_id = None

    def existing(job_id):
        try:
            return manager.read(job_id)
        except (OSError, ValueError):
            from flask import abort
            abort(404, description='任务不存在。')

    @app.delete('/api/jobs/<job_id>')
    def delete_job(job_id):
        try:
            manager.delete(job_id)
        except FileNotFoundError:
            return jsonify(error='任务不存在。'), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 409
        except OSError:
            return jsonify(error='任务文件正在被占用或无法移动，请暂停其他窗口中的视频后重试。'), 409
        return jsonify(id=job_id, deleted=True)

    @app.post('/api/jobs/<job_id>/restore')
    def restore_job(job_id):
        try:
            job = manager.restore(job_id)
        except FileNotFoundError:
            return jsonify(error='回收目录中没有这个任务。'), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 409
        except OSError:
            return jsonify(error='无法恢复任务，请检查文件占用和目录权限。'), 409
        return jsonify(job)

    @app.get('/api/jobs/<job_id>')
    def job_status(job_id):
        # Keep the metadata and result reads atomic with respect to deletion.
        with manager.lock:
            job = existing(job_id)
            directory = manager.directory(job_id)
            if job['status'] == 'completed':
                job['result'] = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            log = directory / 'run.log'
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 16000))
                    tail = stream.read().decode('utf-8', errors='replace')
                key = os.environ.get('DASHSCOPE_API_KEY')
                job['log'] = tail.replace(key, '[REDACTED]') if key else tail
        return jsonify(job)

    @app.get('/api/jobs/<job_id>/video')
    def video(job_id):
        with manager.lock:
            job = existing(job_id)
            path = manager.directory(job_id) / job['video_file']
            if not path.is_file():
                return jsonify(error='视频不存在。'), 404
            # Werkzeug handles HTTP Range (206), essential for browser seeking.
            return send_file(path, conditional=True)

    @app.get('/api/jobs/<job_id>/result')
    def result_download(job_id):
        with manager.lock:
            job = existing(job_id)
            if job['status'] != 'completed':
                return jsonify(error='检测结果尚未生成。'), 409
            return send_file(manager.directory(job_id) / 'result.json', as_attachment=True,
                             download_name=Path(job['filename']).stem + '.result.json')

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V3 本地视频异常检测界面')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--data-dir', type=Path, default=V3_DIR / 'web_data')
    parser.add_argument('--max-upload-mb', type=int, default=2048)
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    if args.max_upload_mb <= 0 or args.timeout <= 0:
        parser.error('上传上限和超时必须为正数。')
    application = create_app(args.data_dir, args.max_upload_mb, args.timeout)
    from waitress import serve
    print(f'V3 视频异常检测：http://127.0.0.1:{args.port}', flush=True)
    try:
        serve(application, host='127.0.0.1', port=args.port, threads=6,
              max_request_body_size=args.max_upload_mb * 1024 * 1024)
    finally:
        application.extensions['jobs'].close()
