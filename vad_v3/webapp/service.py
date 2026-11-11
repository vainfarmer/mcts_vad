"""Process lifecycle and consistent backup for the single-workstation service."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
import psutil
from .settings import ROOT,V3_DIR,load_environment,atomic_json
from .store import Store
from .worker import stop_tree,InstanceLock
from .diagnostics import check_environment


def alive(record):
    if not record:return False
    try:
        p=psutil.Process(record['pid'])
        return abs(p.create_time()-record['created'])<.1 and any(
            x.startswith('vad_v3.webapp.') for x in p.cmdline())
    except (psutil.NoSuchProcess,psutil.AccessDenied):
        return False


def read_runtime(store):
    path=store.root/'runtime.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def start(store,port):
    runtime=read_runtime(store)
    if any(alive(runtime.get(k)) for k in ('web','worker')):
        raise RuntimeError('服务进程已存在，请先 status 检查；需要重启时先 stop。')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:
            raise RuntimeError(f'端口 {port} 已被占用。')
    store.service('control',dict(stop=False))
    runtime=dict(port=port,started_at=time.time())
    flags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
    try:
        for name,module in (('worker','vad_v3.webapp.worker'),('web','vad_v3.webapp.app')):
            args=[sys.executable,'-u','-m',module,'--data-dir',str(store.root)]
            if name=='web':args+=['--port',str(port)]
            with (store.root/(name+'.service.log')).open('a',encoding='utf-8') as log:
                p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'),
                    creationflags=flags,start_new_session=os.name!='nt')
            runtime[name]=dict(pid=p.pid,created=psutil.Process(p.pid).create_time())
            atomic_json(store.root/'runtime.json',runtime)
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            if not all(alive(runtime.get(k)) for k in ('web','worker')):
                raise RuntimeError('启动失败，请查看 web.service.log / worker.service.log。')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health',timeout=1) as response:
                    health=json.load(response)
                if health['worker_online']:
                    print(f'服务已启动：http://127.0.0.1:{port}')
                    return
            except (OSError,ValueError):
                pass
            time.sleep(.25)
        raise RuntimeError('Web / Worker 健康检查超时。')
    except BaseException:
        for name in ('worker','web'):
            if alive(runtime.get(name)):
                stop_tree(runtime[name]['pid'],runtime[name]['created'])
        raise


def stop(store):
    runtime=read_runtime(store)
    store.service('control',dict(stop=True))
    deadline=time.monotonic()+12
    while alive(runtime.get('worker')) and time.monotonic()<deadline:
        time.sleep(.25)
    for name in ('worker','web'):
        if alive(runtime.get(name)):
            stop_tree(runtime[name]['pid'],runtime[name]['created'])
    print('服务已停止。未完成执行保留为中断记录。')


def backup(store,destination):
    runtime=read_runtime(store)
    if any(alive(runtime.get(k)) for k in ('web','worker')):
        raise RuntimeError('请先停止服务，再执行数据库与文件一致性备份。')
    destination=Path(destination).resolve()
    if destination.exists():raise ValueError('备份目标已存在，拒绝覆盖。')
    if destination.is_relative_to(store.root):raise ValueError('备份文件必须位于数据目录之外。')
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Locks also protect against processes launched without this controller.
    with InstanceLock(store.root/'.web.lock'),InstanceLock(store.root/'.data-worker.lock'):
        with store.connect() as db:
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for path in store.root.rglob('*'):
                if path.is_file() and not path.name.endswith(('.lock','-wal','-shm')) and path.name not in ('runtime.json','.worker.owner.json'):
                    archive.write(path,path.relative_to(store.root))
    print(f'备份已保存：{destination}')


def restore(source,destination):
    destination=Path(destination).resolve()
    if destination.exists():
        raise ValueError('恢复目标必须是尚不存在的新目录，避免覆盖现有数据。')
    from .settings import safe_path
    with zipfile.ZipFile(source) as archive:
        if 'system.sqlite3' not in archive.namelist():
            raise ValueError('备份中缺少数据库。')
        # Validate every path before creating any file.
        for entry in archive.infolist():
            safe_path(destination,entry.filename)
        destination.mkdir(parents=True)
        for entry in archive.infolist():
            path=safe_path(destination,entry.filename)
            if entry.is_dir():
                path.mkdir(parents=True,exist_ok=True)
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(entry) as src,path.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src,dst)
    restored=Store(destination)
    restored.service('control',dict(stop=False))
    restored.service('worker',dict(state='stopped'))
    print(f'恢复完成：{destination}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['start','stop','status','check','backup','restore'])
    parser.add_argument('--data-dir',type=Path,default=V3_DIR/'web_data')
    parser.add_argument('--port',type=int,default=7860)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();load_environment()
    if args.command=='restore':
        if not args.output:parser.error('restore 需要 --output 指向备份 ZIP，--data-dir 指向新目录')
        restore(args.output,args.data_dir)
        return
    store=Store(args.data_dir)
    if args.command=='start':start(store,args.port)
    elif args.command=='stop':stop(store)
    elif args.command=='status':
        runtime=read_runtime(store)
        print(json.dumps(dict(web=alive(runtime.get('web')),worker=alive(runtime.get('worker')),
            url=f"http://127.0.0.1:{runtime.get('port',args.port)}"),ensure_ascii=False))
    elif args.command=='check':
        result=check_environment();store.service('diagnostics',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.command=='backup':
        if not args.output:parser.error('backup 需要 --output')
        backup(store,args.output)


if __name__=='__main__':
    main()
