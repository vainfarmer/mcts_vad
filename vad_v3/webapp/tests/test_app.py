"""Exercise HTTP uploads, subprocess completion/failure and byte-range playback."""
import io
import json
from pathlib import Path
import sys
import time

import pytest

from vad_v3.webapp import app as web


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'ENV_FILE', tmp_path / 'missing.env')
    monkeypatch.delenv('DASHSCOPE_MODEL', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-key-not-a-real-credential')
    application = web.create_app(tmp_path / 'jobs', max_upload_mb=1, timeout=5)
    application.config['TESTING'] = True
    yield application
    application.extensions['jobs'].close()


def test_dotenv_uses_explicit_path_with_bom_and_inherits_into_detector(tmp_path, monkeypatch):
    env_file = tmp_path / 'config' / '.env'
    env_file.parent.mkdir()
    env_file.write_text('DASHSCOPE_API_KEY="test-only-from-dotenv"\n', encoding='utf-8-sig')
    monkeypatch.setattr(web, 'ENV_FILE', env_file)
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    monkeypatch.chdir(tmp_path)
    application = web.create_app(tmp_path / 'jobs')
    try:
        assert application.test_client().get('/api/config').get_json()['api_key_configured'] is True
        # Exercise the real subprocess environment, without printing or saving a key.
        script = ('import json, os, sys; from pathlib import Path; '
                  'assert os.environ.get("DASHSCOPE_API_KEY") == "test-only-from-dotenv"; '
                  'Path(sys.argv[1]).write_text(json.dumps({"duration": 10, "anomalies": []}))')
        monkeypatch.setattr(web, 'probe_video', lambda path: {'duration': 10., 'fps': 30.})
        monkeypatch.setattr(web, 'build_command', lambda job, directory:
                            [sys.executable, '-c', script, str(directory / 'result.json')])
        client = application.test_client()
        response = upload(client)
        assert response.status_code == 202
        result = wait_done(client, response.get_json()['id'])
        assert result['status'] == 'completed'
        assert 'test-only-from-dotenv' not in json.dumps(result)
    finally:
        application.extensions['jobs'].close()


def test_process_environment_overrides_dotenv(tmp_path, monkeypatch):
    import os

    env_file = tmp_path / '.env'
    env_file.write_text('DASHSCOPE_API_KEY=test-only-from-file\n', encoding='utf-8')
    monkeypatch.setattr(web, 'ENV_FILE', env_file)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-only-from-process')
    application = web.create_app(tmp_path / 'jobs')
    try:
        assert os.environ['DASHSCOPE_API_KEY'] == 'test-only-from-process'
    finally:
        application.extensions['jobs'].close()


def upload(client, name='测试 clip.mp4', data=b'0123456789', **options):
    return client.post('/api/jobs', data={'video': (io.BytesIO(data), name), **options})


def wait_done(client, job_id):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        job = client.get('/api/jobs/' + job_id).get_json()
        if job['status'] in {'completed', 'failed'}:
            return job
        time.sleep(.02)
    pytest.fail('Background task did not finish')


def test_upload_real_subprocess_result_download_seek_and_reload(app, monkeypatch, tmp_path):
    # Stub ONLY model computation; keep the HTTP/storage/process lifecycle real.
    expected = {'duration': 10., 'is_normal': False, 'processing_time': .1,
                'anomalies': [{'t_start': 2., 't_end': 4., 'confidence': .9,
                               'anomaly_type': 'Arson', 'description': 'Original text.'}]}
    script = tmp_path / 'detector_stub.py'
    script.write_text('import json,sys\nfrom pathlib import Path\n'
                      'Path(sys.argv[1]).write_text(json.dumps(' + repr(expected) + '))\n', encoding='utf-8')
    monkeypatch.setattr(web, 'probe_video', lambda path: {'duration': 10., 'fps': 30.})
    monkeypatch.setattr(web, 'build_command', lambda job, directory: [sys.executable, str(script), str(directory / 'result.json')])
    client = app.test_client()
    response = upload(client)
    assert response.status_code == 202
    job_id = response.get_json()['id']
    job = wait_done(client, job_id)
    assert job['status'] == 'completed'
    assert job['filename'] == '测试 clip.mp4'
    assert job['model'] == 'qwen3-vl-plus'
    assert job['device'] == 'cuda'
    assert job['result'] == expected
    partial = client.get(f'/api/jobs/{job_id}/video', headers={'Range': 'bytes=3-6'})
    assert partial.status_code == 206
    assert partial.data == b'3456'
    assert partial.headers['Content-Range'] == 'bytes 3-6/10'
    partial.close()
    download = client.get(f'/api/jobs/{job_id}/result')
    assert json.loads(download.data) == expected
    assert 'attachment' in download.headers['Content-Disposition']
    download.close()
    restarted = web.create_app(app.extensions['jobs'].root)
    assert restarted.test_client().get(f'/api/jobs/{job_id}').get_json()['result'] == expected
    restarted.extensions['jobs'].close()


@pytest.mark.parametrize('failure', ['exit', 'timeout'])
def test_failed_or_timed_out_detection_is_reported_and_unlocks(app, monkeypatch, failure):
    monkeypatch.setattr(web, 'probe_video', lambda path: {'duration': 10., 'fps': 30.})
    script = 'import sys; print("model failed"); sys.exit(2)' if failure == 'exit' else 'import time; time.sleep(10)'
    monkeypatch.setattr(web, 'build_command', lambda *args: [sys.executable, '-c', script])
    if failure == 'timeout':
        app.extensions['jobs'].timeout = .1
    client = app.test_client()
    job_id = upload(client).get_json()['id']
    job = wait_done(client, job_id)
    assert job['status'] == 'failed'
    assert '退出码 2' in job['error'] if failure == 'exit' else '超过' in job['error']
    assert app.extensions['jobs'].active_id is None
    assert client.get(f'/api/jobs/{job_id}/result').status_code == 409


def test_missing_key_invalid_input_oversize_and_cross_origin(app, monkeypatch):
    client = app.test_client()
    monkeypatch.delenv('DASHSCOPE_API_KEY')
    assert client.get('/api/config').get_json()['api_key_configured'] is False
    assert upload(client).status_code == 503
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-only')
    assert upload(client, name='bad.exe').status_code == 400
    assert upload(client, device='cuda; evil').status_code == 400
    assert upload(client, data=b'x' * (1024 * 1024 + 1)).status_code == 413
    assert client.post('/api/jobs', headers={'Origin': 'https://unrelated.example'}).status_code == 403
    assert app.extensions['jobs'].active_id is None


def test_corrupt_video_is_rejected_without_starting_detector(app):
    client = app.test_client()
    response = upload(client, data=b'not a video')
    assert response.status_code == 400
    assert '无法读取视频' in response.get_json()['error']
    assert app.extensions['jobs'].active_id is None
    assert not list(app.extensions['jobs'].root.glob('*/input.mp4'))


def test_valid_video_is_decoded_and_metadata_is_recorded(app, monkeypatch, tmp_path):
    import cv2
    import numpy as np

    video = tmp_path / 'valid.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 32))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(app.extensions['jobs'], 'start', lambda job: None)
    response = upload(app.test_client(), name='valid.avi', data=video.read_bytes())
    assert response.status_code == 202
    assert response.get_json()['duration'] == pytest.approx(1.)
    assert response.get_json()['fps'] == pytest.approx(10.)


def test_busy_job_and_untrusted_filename(app, monkeypatch, tmp_path):
    monkeypatch.setattr(web, 'probe_video', lambda path: {'duration': 10., 'fps': 30.})
    monkeypatch.setattr(app.extensions['jobs'], 'start', lambda job: None)
    client = app.test_client()
    job = upload(client, name='../../outside.mp4').get_json()
    assert job['filename'] == 'outside.mp4'
    assert job['video_file'] == 'input.mp4'
    assert upload(client).status_code == 409
    assert not (tmp_path / 'outside.mp4').exists()
    assert client.get('/api/jobs/invalid-id/video').status_code == 404
    restarted = web.create_app(app.extensions['jobs'].root)
    assert restarted.test_client().get('/api/jobs/' + job['id']).get_json()['status'] == 'failed'
    restarted.extensions['jobs'].close()


def test_command_calls_v3_cli_without_shell_or_key_argument(tmp_path):
    job = dict(video_file='input.mp4', device='cuda', mcts_mode='refine', context='shop camera', full_classification=True)
    command = web.build_command(job, tmp_path)
    assert command[:3] == [sys.executable, '-u', str(web.V3_DIR / 'video_anomaly_detection/run.py')]
    assert command[command.index('--video_path') + 1] == str(tmp_path / 'input.mp4')
    assert '--no_classify_skip_class_agnostic' in command
    assert '--api_key' not in command
    assert command[command.index('--model') + 1] == 'qwen3-vl-plus'


def test_dotenv_model_reaches_job_and_real_cli_command(tmp_path, monkeypatch):
    env_file = tmp_path / '.env'
    env_file.write_text('DASHSCOPE_MODEL=qwen3-vl-flash\n', encoding='utf-8')
    monkeypatch.setattr(web, 'ENV_FILE', env_file)
    monkeypatch.delenv('DASHSCOPE_MODEL', raising=False)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-only')
    monkeypatch.setattr(web, 'probe_video', lambda path: {'duration': 10., 'fps': 30.})
    application = web.create_app(tmp_path / 'jobs')
    monkeypatch.setattr(application.extensions['jobs'], 'start', lambda job: None)
    try:
        client = application.test_client()
        assert client.get('/api/config').get_json()['model'] == 'qwen3-vl-flash'
        response = upload(client)
        assert response.status_code == 202
        job = response.get_json()
        assert job['model'] == 'qwen3-vl-flash'
        command = web.build_command(job, tmp_path)
        assert command[command.index('--model') + 1] == 'qwen3-vl-flash'
        assert application.extensions['jobs'].read(job['id'])['model'] == 'qwen3-vl-flash'
    finally:
        application.extensions['jobs'].close()


@pytest.mark.parametrize('device', [None, 'cpu'])
def test_device_default_and_explicit_override_reach_cli(app, monkeypatch, tmp_path, device):
    monkeypatch.setattr(web, 'probe_video', lambda path: {'duration': 10., 'fps': 30.})
    monkeypatch.setattr(app.extensions['jobs'], 'start', lambda job: None)
    response = upload(app.test_client(), **({'device': device} if device else {}))
    assert response.status_code == 202
    job = response.get_json()
    assert job['device'] == (device or 'cuda')
    command = web.build_command(job, tmp_path)
    assert command[command.index('--device') + 1] == (device or 'cuda')


def seed_job(app, job_id='b' * 32, status='completed'):
    manager = app.extensions['jobs']
    job = dict(id=job_id, filename='saved.mp4', video_file='input.mp4',
               status=status, created_at=time.time(), device='cuda', duration=10.)
    manager.save(job)
    directory = manager.directory(job_id)
    (directory / 'input.mp4').write_bytes(b'0123456789')
    (directory / 'run.log').write_text('saved log', encoding='utf-8')
    (directory / 'result.json').write_text(json.dumps({'duration': 10., 'anomalies': []}), encoding='utf-8')
    return job_id, directory


@pytest.mark.parametrize('status', ['completed', 'failed'])
def test_delete_and_restore_job_with_all_files(app, status):
    job_id, directory = seed_job(app, status=status)
    original = {file.name: file.read_bytes() for file in directory.iterdir()}
    client = app.test_client()
    assert client.delete('/api/jobs/' + job_id).status_code == 200
    assert client.get('/api/jobs').get_json()['jobs'] == []
    assert not directory.exists()
    for suffix in ['', '/video', '/result']:
        assert client.get('/api/jobs/' + job_id + suffix).status_code == 404
    archived = app.extensions['jobs'].root / '.trash' / job_id
    assert {file.name: file.read_bytes() for file in archived.iterdir()} == original
    assert client.delete('/api/jobs/' + job_id).status_code == 404
    assert client.post('/api/jobs/' + job_id + '/restore').status_code == 200
    assert len(client.get('/api/jobs').get_json()['jobs']) == 1
    assert {file.name: file.read_bytes() for file in directory.iterdir()} == original
    response = client.get('/api/jobs/' + job_id + '/video', headers={'Range': 'bytes=3-6'})
    assert response.status_code == 206 and response.data == b'3456'
    response.close()
    assert client.post('/api/jobs/' + job_id + '/restore').status_code == 404


@pytest.mark.parametrize('status', ['uploading', 'running', 'completed'])
def test_running_job_cannot_be_deleted_even_during_final_save(app, status):
    job_id, directory = seed_job(app, status=status)
    if status == 'completed':
        app.extensions['jobs'].active_id = job_id
    response = app.test_client().delete('/api/jobs/' + job_id)
    assert response.status_code == 409
    assert (directory / 'input.mp4').read_bytes() == b'0123456789'


def test_delete_is_isolated_from_other_jobs_and_checks_origin(app):
    removed, directory = seed_job(app)
    other, other_directory = seed_job(app, job_id='c' * 32, status='running')
    app.extensions['jobs'].active_id = other
    client = app.test_client()
    assert client.delete('/api/jobs/' + removed, headers={'Origin': 'https://unrelated.example'}).status_code == 403
    assert directory.exists()
    assert client.delete('/api/jobs/' + removed).status_code == 200
    assert client.post('/api/jobs/' + removed + '/restore', headers={'Origin': 'https://unrelated.example'}).status_code == 403
    assert [job['id'] for job in client.get('/api/jobs').get_json()['jobs']] == [other]
    assert (other_directory / 'input.mp4').read_bytes() == b'0123456789'
    assert app.extensions['jobs'].active_id == other
    assert client.delete('/api/jobs/invalid-id').status_code == 404
    assert client.post('/api/jobs/invalid-id/restore').status_code == 404


def test_restore_does_not_overwrite_new_data(app):
    job_id, directory = seed_job(app)
    client = app.test_client()
    assert client.delete('/api/jobs/' + job_id).status_code == 200
    directory.mkdir()
    (directory / 'important.txt').write_text('keep')
    assert client.post('/api/jobs/' + job_id + '/restore').status_code == 409
    assert (directory / 'important.txt').read_text() == 'keep'
    assert (app.extensions['jobs'].root / '.trash' / job_id / 'input.mp4').is_file()


def test_delete_rejects_directory_outside_data_root(app, tmp_path, monkeypatch):
    job_id, directory = seed_job(app)
    outside = tmp_path / 'unrelated'
    outside.mkdir()
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        # Simulate a symlink/junction without requiring Windows symlink privileges.
        if path == directory:
            return outside
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'resolve', resolve)
    assert app.test_client().delete('/api/jobs/' + job_id).status_code == 404
    assert (directory / 'input.mp4').read_bytes() == b'0123456789'
    assert outside.is_dir()


def test_file_in_use_does_not_remove_task_or_other_files(app, monkeypatch):
    job_id, directory = seed_job(app)
    original_rename = Path.rename

    def rename(path, destination):
        if path == directory:
            raise PermissionError('File in use')
        return original_rename(path, destination)

    monkeypatch.setattr(Path, 'rename', rename)
    client = app.test_client()
    assert client.delete('/api/jobs/' + job_id).status_code == 409
    assert len(client.get('/api/jobs').get_json()['jobs']) == 1
    assert (directory / 'input.mp4').read_bytes() == b'0123456789'
    assert client.get('/api/jobs/' + job_id).status_code == 200
