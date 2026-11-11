"""Single-GPU persistent queue worker; executable with python -m."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import psutil
from .settings import ROOT, V3_DIR, atomic_json, code_hash, load_environment, redact, runtime_identity
from .store import Store
from .results import validate_result


class InstanceLock:
    def __init__(self,path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream = self.path.open('a+b')
        self.stream.seek(0)
        if self.path.stat().st_size==0:
            self.stream.write(b'0'); self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError('已有 Worker 正在占用本项目的推理执行锁。')
        return self

    def __exit__(self,*args):
        if self.stream and not self.stream.closed:
            self.stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_UNLCK,1)
            self.stream.close()


def stop_tree(pid, created=None):
    try:
        parent = psutil.Process(pid)
        if created is not None and abs(parent.create_time()-created)>.1:
            raise RuntimeError('进程身份不匹配，拒绝终止复用的 PID。')
        members = parent.children(recursive=True)+[parent]
        for process in reversed(members):
            try: process.terminate()
            except psutil.NoSuchProcess: pass
        _,alive = psutil.wait_procs(members,timeout=5)
        for process in alive:
            try: process.kill()
            except psutil.NoSuchProcess: pass
        _,alive = psutil.wait_procs(alive,timeout=5)
        if alive:
            raise RuntimeError('旧推理进程仍在运行，执行锁不能释放。')
    except psutil.NoSuchProcess:
        pass


def build_command(job, attempt, store):
    directory = store.path(attempt['artifact_dir'])
    video = store.video(job['video_id'],deleted=True)
    cfg = job['config']
    client_args=[]
    for key,value in cfg.get('client',{}).items():
        client_args.extend(['--'+key,str(value)])
    return [sys.executable,'-u',str(V3_DIR/'video_anomaly_detection/run.py'),
            '--video_path',str(store.path(video['path'])),
            '--output_file',str(directory/'result.json'),
            '--model',cfg['model'],'--context',cfg['context'],
            '--config_json',str(directory/'detector_config.json'),
            '--research_dir',str(directory),'--strict_discovery']+client_args


class Worker:
    def __init__(self,store,command_builder=build_command,lock_path=None):
        self.store = store
        self.command_builder = command_builder
        self.id = uuid.uuid4().hex
        self.stopping = False
        self.process = None
        self.lock_path = lock_path or V3_DIR/'web_data/.worker.lock'

    def recover(self):
        with self.store.connect() as db:
            active = [self.store.unpack(r) for r in db.execute(
                "SELECT * FROM attempts WHERE state IN ('running','cancelling')")]
        for a in active:
            directory = str(self.store.path(a['artifact_dir']))
            # Also close the narrow spawn-before-PID-commit crash window.
            for process in psutil.process_iter(['pid','cmdline','create_time']):
                args = process.info.get('cmdline') or []
                if '--research_dir' in args and directory in args:
                    stop_tree(process.pid,process.info['create_time'])
            if a['pid']:
                try:
                    p = psutil.Process(a['pid'])
                    if a['process_created'] and abs(p.create_time()-a['process_created'])<.1:
                        stop_tree(p.pid,a['process_created'])
                except psutil.NoSuchProcess:
                    pass
            self.store.finish(a['id'],a['token'],'interrupted','上次执行中断；旧进程已确认停止，可重试。')

    def execute(self,job,attempt):
        directory = self.store.path(attempt['artifact_dir'])
        directory.mkdir(parents=True,exist_ok=True)
        cfg = job['config']
        atomic_json(directory/'detector_config.json',cfg['detector_config'])
        video = self.store.video(job['video_id'],deleted=True)
        atomic_json(directory/'manifest.json',dict(schema_version=1,job_id=job['id'],
            attempt_id=attempt['id'],input_sha256=video['sha256'],code_hash=job['code_hash'],
            config=cfg,created_at=time.time()))
        state,error,code = 'failed',None,None
        started=time.monotonic()
        peak=0
        try:
            if job['code_hash'] not in ('legacy',code_hash()) and self.command_builder is build_command:
                raise RuntimeError('排队后算法代码已改变，请创建新任务。')
            if cfg.get('environment') and cfg['environment']!=runtime_identity() and self.command_builder is build_command:
                raise RuntimeError('排队后模型缓存或运行依赖版本已改变，请创建新任务。')
            with (directory/'run.log').open('w',encoding='utf-8') as log:
                self.process = subprocess.Popen(self.command_builder(job,attempt,self.store),
                    cwd=ROOT,env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'),
                    stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                proc=psutil.Process(self.process.pid)
                created=proc.create_time()
                self.store.heartbeat(attempt['id'],proc.pid,created)
                while self.process.poll() is None:
                    self.store.service('worker',dict(id=self.id,pid=os.getpid(),state='running',job_id=job['id']))
                    a=self.store.attempt(attempt['id'])
                    control=self.store.service('control')
                    if control and control['payload'].get('stop'):
                        self.stopping=True
                    try:
                        peak=max(peak,sum(p.memory_info().rss for p in [proc]+proc.children(recursive=True)))
                    except (psutil.NoSuchProcess,psutil.AccessDenied):
                        pass
                    self.store.heartbeat(attempt['id'],metrics=dict(peak_ram_bytes=peak,
                        elapsed_seconds=time.monotonic()-started,peak_gpu_bytes=None))
                    if self.stopping or a['cancel_requested']:
                        stop_tree(proc.pid,created)
                        state='interrupted' if self.stopping else 'cancelled'
                        break
                    if time.monotonic()-started>cfg['timeout']:
                        stop_tree(proc.pid,created)
                        raise TimeoutError('检测超过任务超时限制。')
                    time.sleep(.3)
                code=self.process.wait()
            if state not in ('cancelled','interrupted'):
                if code!=0:
                    tail=(directory/'run.log').read_text(encoding='utf-8',errors='replace')[-2500:]
                    raise RuntimeError(f'检测进程退出码 {code}。\n'+redact(tail))
                result=json.loads((directory/'result.json').read_text(encoding='utf-8'))
                validate_result(result,video['metadata']['duration'])
                atomic_json(directory/'result.normalized.json',dict(schema_version=1,
                    attempt_id=attempt['id'],detections=result['anomalies']))
                state='completed'
        except Exception as exc:
            error=redact(str(exc))
            if self.process and self.process.poll() is None:
                stop_tree(self.process.pid)
        finally:
            if self.process is not None and self.process.poll() is None:
                # Keep the attempt active. Recovery must confirm process death before claiming work.
                raise RuntimeError('推理进程尚未退出，暂停调度，保留活动执行供恢复检查。')
            self.process=None
            self.store.heartbeat(attempt['id'],metrics=dict(peak_ram_bytes=peak,
                elapsed_seconds=time.monotonic()-started,peak_gpu_bytes=None))
            self.store.finish(attempt['id'],attempt['token'],state,error,code)

    def run(self,once=False):
        with InstanceLock(self.lock_path):
            owner_path=Path(self.lock_path).with_suffix('.owner.json')
            if owner_path.exists():
                previous=Path(json.loads(owner_path.read_text(encoding='utf-8'))['data_dir'])
                if previous.resolve()!=self.store.root and (previous/'system.sqlite3').exists():
                    # A worker for another data directory may have died with a live GPU child.
                    # Recover it under the same global lock before switching queues.
                    with InstanceLock(previous/'.data-worker.lock'):
                        Worker(Store(previous),lock_path=self.lock_path).recover()
            with InstanceLock(self.store.root/'.data-worker.lock'):
                atomic_json(owner_path,dict(data_dir=str(self.store.root),worker_id=self.id))
                self.recover()
                while not self.stopping:
                    control=self.store.service('control')
                    if control and control['payload'].get('stop'):
                        self.stopping=True
                        break
                    self.store.service('worker',dict(id=self.id,pid=os.getpid(),state='idle'))
                    claimed=self.store.claim(self.id)
                    if claimed:
                        self.execute(*claimed)
                    elif once:
                        break
                    else:
                        time.sleep(.5)
                self.store.service('worker',dict(id=self.id,pid=os.getpid(),state='stopped'))


def main():
    parser=argparse.ArgumentParser(description='V3 持久化检测队列')
    parser.add_argument('--data-dir',type=Path,default=V3_DIR/'web_data')
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    load_environment()
    worker=Worker(Store(args.data_dir))
    def shutdown(*unused):
        worker.stopping=True
    signal.signal(signal.SIGTERM,shutdown)
    signal.signal(signal.SIGINT,shutdown)
    worker.run(args.once)


if __name__=='__main__':
    main()
