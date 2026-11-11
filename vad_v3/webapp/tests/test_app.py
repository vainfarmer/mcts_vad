"""Queue, API, immutable review, migration and evaluation acceptance tests."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import pytest
from vad_v3.webapp import app as web
from vad_v3.webapp.settings import atomic_json, code_hash, effective_config, file_hash
from vad_v3.webapp.store import Store, Conflict
from vad_v3.webapp.worker import Worker, InstanceLock
from vad_v3.webapp.results import validate_result
from vad_v3.webapp import experiments

RESULT=dict(duration=10.,is_normal=False,anomalies=[
    dict(t_start=2.,t_end=4.,confidence=.8,anomaly_type='Fighting',
         description='Original description',original_descriptions=[
             dict(t_start=0.,t_end=6.,description='Original description')])],
    timestamps=[0.,2.,4.,6.,8.],frame_scores=[.1,.9,.8,.1,.1])


@pytest.fixture(scope='module')
def source_videos(tmp_path_factory):
    """原生 FPS 评估使用真实的小视频；API/Worker 测试仍不调用 VLM。"""
    import cv2
    import numpy as np
    root=tmp_path_factory.mktemp('source-videos')
    for name,fps in (('a',30),('b',25)):
        writer=cv2.VideoWriter(str(root/(name+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),fps,(32,24))
        assert writer.isOpened()
        try:
            for _ in range(fps*10):writer.write(np.zeros((24,32,3),dtype=np.uint8))
        finally:writer.release()
    return root


@pytest.fixture
def app(tmp_path,monkeypatch,source_videos):
    monkeypatch.setattr(web,'ENV_FILE',tmp_path/'missing.env')
    monkeypatch.setenv('DASHSCOPE_API_KEY','test-only-key')
    monkeypatch.setattr(web,'probe_video',lambda p:dict(duration=10.,fps=30.,frame_count=300))
    monkeypatch.setattr(experiments,'probe_video_metadata',lambda p:dict(duration=10.,fps=30.,frame_count=300))
    monkeypatch.setattr(experiments,'DATASET_ROOT',source_videos)
    application=web.create_app(tmp_path/'data',max_upload_mb=1,timeout=5,migrate=False)
    application.config['TESTING']=True
    yield application
    application.extensions['executor'].shutdown()


def video(client):
    response=client.post('/api/videos',data={'video':(io.BytesIO(b'video-bytes-1234'),'test.mp4')})
    assert response.status_code==201,response.get_json()
    return response.get_json()


def submit(client,vid=None,key=None,config=None):
    vid=vid or video(client)['id']
    return client.post('/api/jobs',json=dict(video_id=vid,config=config or {}),
                       headers={'Idempotency-Key':key} if key else {})


def complete(store,job_id):
    claimed=store.claim('unit-test')
    assert claimed[0]['id']==job_id
    attempt=claimed[1]
    atomic_json(store.path(attempt['artifact_dir']+'/result.json'),RESULT)
    store.finish(attempt['id'],attempt['token'],'completed')
    return attempt


def test_upload_queue_idempotency_and_range(app):
    c=app.test_client();v=video(c)
    first=submit(c,v['id'],'same-key-123')
    second=submit(c,v['id'],'same-key-123')
    assert first.status_code==202 and second.status_code==200
    assert first.json['id']==second.json['id']
    assert first.json['device']=='cuda'
    assert first.json['mcts_mode']=='refine'
    assert c.get('/api/config').json['defaults']['mcts_mode']=='refine'
    assert c.get('/api/config').json['defaults']['mcts_acceptance_policy']=='force_best'
    saved=app.extensions['store'].job(first.json['id'])
    assert saved['config']['detector_config']['mcts']['acceptance_policy']=='force_best'
    assert first.json['status']=='queued'
    assert submit(c,v['id'],'same-key-123',{'device':'cpu'}).status_code==409
    explicit_off=submit(c,v['id'],'new-key-123',{'mcts_mode':'off'})
    assert explicit_off.status_code==202 and explicit_off.json['mcts_mode']=='off'
    explicit_strict=submit(c,v['id'],config={'mcts':{'acceptance_policy':'strict'}})
    assert app.extensions['store'].job(explicit_strict.json['id'])['config']['detector_config']['mcts']['acceptance_policy']=='strict'
    response=c.get('/api/videos/'+v['id']+'/stream',headers={'Range':'bytes=2-5'})
    assert response.status_code==206 and response.data==b'deo-'
    response.close()


def test_origin_path_and_configuration_validation(app):
    c=app.test_client()
    assert c.post('/api/videos',headers={'Origin':'https://example.invalid'}).status_code==403
    assert c.get('/api/config',headers={'Host':'attacker.invalid'}).status_code==400
    assert c.get('/api/videos/not-found/stream').status_code==404
    v=video(c)
    assert submit(c,v['id'],config={'device':'not-a-device'}).status_code==400
    assert submit(c,v['id'],config={'shell':'anything'}).status_code==400
    assert c.post('/api/videos',data={'video':(io.BytesIO(b'x'),'bad.exe')}).status_code==400
    with pytest.raises(ValueError):
        app.extensions['store'].path('../outside')


def test_cancel_queued_retry_and_soft_delete(app):
    c=app.test_client();j=submit(c).json;store=app.extensions['store']
    assert c.delete('/api/jobs/'+j['id']).status_code==409
    assert c.post('/api/jobs/'+j['id']+'/cancel').json['status']=='cancelled'
    assert c.post('/api/jobs/'+j['id']+'/retry').status_code==202
    complete(store,j['id'])
    assert c.delete('/api/jobs/'+j['id']).status_code==200
    assert c.get('/api/jobs/'+j['id']).status_code==404
    assert len(c.get('/api/jobs?deleted=1').json['jobs'])==1
    assert c.post('/api/jobs/'+j['id']+'/restore').status_code==200
    assert store.result(j['id'])==RESULT
    assert store.path(store.video(j['video_id'])['path']).exists()


def test_review_versions_manual_identity_and_original_unchanged(app):
    c=app.test_client();j=submit(c).json;store=app.extensions['store'];complete(store,j['id'])
    before=store.result(j['id'])
    payload=dict(subject_id='det-0',expected_revision=0,action='corrected',
        after=dict(t_start=1.,t_end=5.,confidence=.8,anomaly_type='Assault'),reason='review')
    assert c.post('/api/jobs/'+j['id']+'/reviews',json=payload).status_code==201
    assert c.post('/api/jobs/'+j['id']+'/reviews',json=payload).status_code==409
    manual=c.post('/api/jobs/'+j['id']+'/reviews',json=dict(action='added',after=payload['after'])).json
    payload.update(subject_id=manual['subject_id'],expected_revision=1)
    assert c.post('/api/jobs/'+j['id']+'/reviews',json=payload).status_code==201
    assert store.result(j['id'])==before
    assert len(c.get('/api/jobs/'+j['id']+'/analysis').json['reviews'])==3
    for fmt in ('csv','html'):
        assert c.get('/api/jobs/'+j['id']+'/export?format='+fmt).status_code==200


def test_export_escapes_untrusted_description(app):
    c=app.test_client();j=submit(c).json;store=app.extensions['store'];a=complete(store,j['id'])
    result=json.loads(json.dumps(RESULT));result['anomalies'][0]['description']='<script>alert(1)</script>'
    atomic_json(store.path(a['artifact_dir']+'/result.json'),result)
    body=c.get('/api/jobs/'+j['id']+'/export').get_data(as_text=True)
    assert '<script>alert(1)</script>' not in body
    assert '&lt;script&gt;' in body


def fake_command(job,attempt,store):
    path=store.path(attempt['artifact_dir']+'/result.json')
    return [sys.executable,'-c','import pathlib,sys;pathlib.Path(sys.argv[1]).write_text(sys.argv[2])',
            str(path),json.dumps(RESULT)]


def test_worker_runs_multiple_jobs_serially_and_persists_attempts(app,tmp_path):
    c=app.test_client();one=submit(c).json;two=submit(c).json;store=app.extensions['store']
    worker=Worker(store,fake_command,tmp_path/'lock')
    worker.run(once=True)
    assert store.job(one['id'])['status']==store.job(two['id'])['status']=='completed'
    assert len(store.attempts(one['id']))==1
    assert store.attempts(one['id'])[0]['finished_at']<=store.attempts(two['id'])[0]['started_at']
    assert store.result(one['id'])==RESULT


def test_worker_failure_is_not_normal_and_retry_retains_attempt(app,tmp_path):
    c=app.test_client();j=submit(c).json;store=app.extensions['store']
    worker=Worker(store,lambda *unused:[sys.executable,'-c','raise RuntimeError("failed")'],tmp_path/'lock')
    worker.run(once=True)
    assert store.job(j['id'])['status']=='failed'
    assert c.get('/api/jobs/'+j['id']+'/result').status_code==409
    store.retry(j['id'],code_hash())
    Worker(store,fake_command,tmp_path/'lock').run(once=True)
    assert [a['state'] for a in store.attempts(j['id'])]==['failed','completed']


def test_cancel_running_stops_subprocess_and_does_not_commit(app,tmp_path):
    c=app.test_client();j=submit(c).json;store=app.extensions['store']
    worker=Worker(store,lambda *unused:[sys.executable,'-c','import time;time.sleep(30)'],tmp_path/'lock')
    thread=threading.Thread(target=worker.run,kwargs={'once':True});thread.start()
    deadline=time.time()+8
    while time.time()<deadline:
        attempts=store.attempts(j['id'])
        if attempts and attempts[0]['pid']:break
        time.sleep(.05)
    store.cancel(j['id']);thread.join(12)
    assert not thread.is_alive()
    assert store.job(j['id'])['status']=='cancelled'
    assert store.job(j['id'])['result_attempt_id'] is None


def test_timeout_fails_with_retained_log(app,tmp_path):
    c=app.test_client();j=submit(c,config={'timeout':1}).json;store=app.extensions['store']
    Worker(store,lambda *unused:[sys.executable,'-c','import time;time.sleep(20)'],tmp_path/'lock').run(once=True)
    assert store.job(j['id'])['status']=='failed'
    assert '超时' in store.job(j['id'])['error']


def test_single_instance_and_stale_result_token(app,tmp_path):
    with InstanceLock(tmp_path/'lock'):
        with pytest.raises(RuntimeError):
            with InstanceLock(tmp_path/'lock'):pass
    c=app.test_client();j=submit(c).json;store=app.extensions['store'];a=complete(store,j['id'])
    with pytest.raises(Conflict):
        store.finish(a['id'],a['token'],'completed')


def test_restart_marks_active_attempt_interrupted(app,tmp_path):
    c=app.test_client();j=submit(c).json;store=app.extensions['store']
    store.claim('dead-worker')
    Worker(store,fake_command,tmp_path/'lock').run(once=True)
    assert store.job(j['id'])['status']=='interrupted'
    assert store.claim('new-worker') is None


def test_worker_does_not_inherit_config_only_from_web(tmp_path,monkeypatch):
    from vad_v3.webapp.settings import load_environment
    path=tmp_path/'.env';path.write_text('DASHSCOPE_API_KEY=fixture-only\n',encoding='utf-8-sig')
    monkeypatch.delenv('DASHSCOPE_API_KEY',raising=False);load_environment(path)
    assert os.environ['DASHSCOPE_API_KEY']=='fixture-only'
    monkeypatch.setenv('DASHSCOPE_API_KEY','environment-priority');load_environment(path)
    assert os.environ['DASHSCOPE_API_KEY']=='environment-priority'


@pytest.mark.parametrize('mutate',[
    lambda r:r['anomalies'][0].update(t_end=11),
    lambda r:r.update(timestamps=[0,0,2,3,4]),
    lambda r:r.update(frame_scores=[float('nan')]*5),
    lambda r:r.update(frame_scores=[.1]),
    lambda r:r.update(is_normal=True),
])
def test_invalid_outputs_are_rejected(mutate):
    result=json.loads(json.dumps(RESULT));mutate(result)
    with pytest.raises(ValueError):validate_result(result)


def bundle():
    return dict(name='fixture',ground_truth={
        'a':dict(category='Fighting',fps=30.,frame_count=300,segments=[[60,120]]),
        'b':dict(category='Normal',fps=25.,frame_count=250,segments=[])},
        groups={'on':{'a':RESULT,'b':dict(duration=10.,is_normal=True,anomalies=[],
            timestamps=[0.,5.,9.],frame_scores=[.01,.01,.01])}})


def test_evaluation_uses_classes_coverage_and_immutable_runs(app):
    store=app.extensions['store'];b=bundle();exp=experiments.create_experiment(store,b)
    one=experiments.create_evaluation(store,exp['id']);experiments.evaluate(store,one)
    two=experiments.create_evaluation(store,exp['id'],'report_historical_v1');experiments.evaluate(store,two)
    c=app.test_client();row=c.get('/api/evaluations/'+one).json
    assert row['status']=='completed',row
    metrics=row['metrics']['groups']['on']
    assert metrics['abnormal']['per_class']['Fighting']['AP@0.5']==1
    assert metrics['classification']['accuracy']==1
    assert metrics['classification']['normal_false_positive_rate']==0
    assert len(experiments.list_experiments(store)[0]['evaluations'])==2
    b['groups']['on'].pop('b')
    with pytest.raises(ValueError,match='覆盖'):
        experiments.create_experiment(store,b)


def test_evaluation_missing_curves_reports_unavailable_not_zero(app):
    store=app.extensions['store'];b=bundle();b['groups']['on']['b'].pop('frame_scores')
    b['groups']['on']['b'].pop('timestamps')
    eid=experiments.create_experiment(store,b)['id'];rid=experiments.create_evaluation(store,eid)
    experiments.evaluate(store,rid)
    frame=app.test_client().get('/api/evaluations/'+rid).json['metrics']['groups']['on']['frame']
    assert frame['auc'] is None and frame['missing_videos']==['b']


def test_native_evaluation_reprobes_video_instead_of_using_bundle_fps(app):
    store=app.extensions['store'];b=bundle()
    # 模拟过期清单：记录的 FPS/帧数均错，但时长仍为 10 秒。
    b['ground_truth']['a'].update(fps=10.,frame_count=100)
    eid=experiments.create_experiment(store,b)['id'];rid=experiments.create_evaluation(store,eid)
    experiments.evaluate(store,rid)
    row=app.test_client().get('/api/evaluations/'+rid).json
    assert row['status']=='completed',row
    assert row['metrics']['source_video_metadata']['a']['fps']==30.
    assert row['metrics']['groups']['on']['abnormal']['per_class']['Fighting']['AP@0.5']==1.
    original=json.loads(store.path(experiments.experiment(store,eid)['manifest']['input_path']).read_text(encoding='utf-8'))
    assert original['ground_truth']['a']['fps']==10.


def test_native_evaluation_requires_readable_source_videos(app,tmp_path,monkeypatch):
    monkeypatch.setattr(experiments,'DATASET_ROOT',tmp_path/'missing-source')
    store=app.extensions['store'];eid=experiments.create_experiment(store,bundle())['id']
    rid=experiments.create_evaluation(store,eid);experiments.evaluate(store,rid)
    row=app.test_client().get('/api/evaluations/'+rid).json
    assert row['status']=='failed' and row['error']


@pytest.mark.parametrize('legacy_mode',[None,'off','shadow','refine'])
def test_legacy_import_preserves_original_directories(tmp_path,legacy_mode):
    from vad_v3.webapp.migration import import_legacy
    store=Store(tmp_path/'data');jid='a'*32;old=store.root/jid;old.mkdir()
    (old/'input.mp4').write_bytes(b'old-video')
    legacy=dict(id=jid,filename='old.mp4',video_file='input.mp4',
        status='completed',duration=10.,fps=30.,created_at=time.time(),device='cuda',
        context='camera',full_classification=False)
    if legacy_mode is not None: legacy['mcts_mode']=legacy_mode
    atomic_json(old/'job.json',legacy)
    atomic_json(old/'result.json',RESULT)
    result=import_legacy(store)
    assert result['imported']==1 and not result['errors']
    with store.connect() as db:
        config=json.loads(db.execute('SELECT config_json FROM jobs WHERE id=?',(jid,)).fetchone()[0])
    assert config['mcts_mode']==(legacy_mode or 'off')
    assert config['detector_config']['mcts']['mode']==(legacy_mode or 'off')
    assert config['detector_config']['mcts']['acceptance_policy']=='strict'
    assert store.result(jid)==RESULT and (old/'input.mp4').exists()
    assert import_legacy(store)['imported']==0


def test_experiment_from_completed_tasks_keeps_input_identity(app):
    c=app.test_client();v=video(c);store=app.extensions['store']
    one=submit(c,v['id'],config={'mcts_mode':'off'}).json;complete(store,one['id'])
    two=submit(c,v['id'],config={'mcts_mode':'shadow'}).json;complete(store,two['id'])
    payload=dict(name='two configs',groups={'off':[one['id']],'shadow':[two['id']]},
        ground_truth={'test':dict(category='Fighting',segments=[[60,120]])})
    response=c.post('/api/experiments/from-jobs',json=payload)
    assert response.status_code==201,response.json
    assert response.json['manifest']['video_count']==1
    source=response.json['manifest']['provenance']['jobs']
    assert source['off']['test']['video_sha256']==source['shadow']['test']['video_sha256']
    assert source['off']['test']['config_hash']!=source['shadow']['test']['config_hash']
    payload['ground_truth']['extra']=payload['ground_truth']['test']
    assert c.post('/api/experiments/from-jobs',json=payload).status_code==400


def test_backup_restore_keeps_reviews_and_rejects_live_writers(app,tmp_path):
    from vad_v3.webapp.service import backup,restore
    c=app.test_client();store=app.extensions['store'];j=submit(c).json;complete(store,j['id'])
    c.post('/api/jobs/'+j['id']+'/reviews',json=dict(subject_id='det-0',action='confirmed',expected_revision=0))
    destination=tmp_path/'backup.zip'
    with InstanceLock(store.root/'.web.lock'):
        with pytest.raises(RuntimeError):backup(store,destination)
    assert not destination.exists()
    backup(store,destination)
    restored_path=tmp_path/'restored';restore(destination,restored_path)
    restored=Store(restored_path)
    assert restored.result(j['id'])==RESULT
    assert restored.reviews(j['id'])==store.reviews(j['id'])
    assert file_hash(restored.path(restored.video(j['video_id'])['path']))==store.video(j['video_id'])['sha256']
    with pytest.raises(ValueError):restore(destination,restored_path)


def test_global_worker_recovers_previous_data_directory(app,tmp_path):
    store=app.extensions['store'];c=app.test_client();job=submit(c).json
    store.claim('crashed')
    lock=tmp_path/'global.lock'
    atomic_json(lock.with_suffix('.owner.json'),dict(data_dir=str(store.root)))
    other=Store(tmp_path/'another-data')
    Worker(other,fake_command,lock).run(once=True)
    assert store.job(job['id'])['status']=='interrupted'


def test_invalid_json_shapes_are_client_errors(app):
    c=app.test_client();vid=video(c)['id']
    assert c.post('/api/jobs',json=[]).status_code==400
    assert c.post('/api/jobs',json=dict(video_id=vid,config=[])).status_code==400
    assert c.post('/api/experiments',json=dict(ground_truth={'a':[]},groups={'a':{}})).status_code==400
    assert c.post('/api/experiments/from-jobs',json=[]).status_code==400


def test_historical_frame_precision_and_length_match_original_evaluator(app):
    from vad_v3.video_anomaly_detection.evaluate import compute_frame_auc
    store=app.extensions['store'];b=bundle()
    b['ground_truth']['b'].update(fps=30.,frame_count=300)
    # A decimal duration whose product can be just below the true integer count.
    b['ground_truth']['a']['frame_count']=302
    b['groups']['on']['a']=json.loads(json.dumps(RESULT))
    b['groups']['on']['a']['duration']=302/30
    expected=compute_frame_auc(b['ground_truth'],b['groups']['on'],fps=30)
    eid=experiments.create_experiment(store,b)['id']
    rid=experiments.create_evaluation(store,eid,'report_historical_v1');experiments.evaluate(store,rid)
    c=app.test_client();actual=c.get('/api/evaluations/'+rid).json['metrics']['groups']['on']['frame']
    assert actual['auc']==expected['auc']
    assert actual['aupr']==expected['aupr']
    downloaded=c.get('/api/evaluations/'+rid+'/download?format=csv')
    assert downloaded.status_code==200 and b'frame,auc' in downloaded.data
