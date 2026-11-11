"""Non-destructive import of the original directory-based web tasks."""
import json
from pathlib import Path
import shutil
import time
import uuid
from .settings import atomic_json, dumps, digest, effective_config, file_hash


def import_legacy(store):
    imported,errors=0,[]
    paths=list(store.root.glob('*/job.json'))+list(store.root.glob('.trash/*/job.json'))
    for path in paths:
        try:
            data=json.loads(path.read_text(encoding='utf-8'))
            jid=data['id']
            with store.connect() as db:
                if db.execute('SELECT 1 FROM jobs WHERE id=?',(jid,)).fetchone():
                    continue
            video_path=path.parent/data['video_file']
            if not video_path.is_file():
                errors.append(dict(id=jid,error='原视频缺失，保留旧目录供人工恢复'))
                continue
            resolved=video_path.resolve()
            if not resolved.is_relative_to(store.root):
                raise ValueError('旧视频路径不在数据目录内')
            vid,aid=uuid.uuid4().hex,uuid.uuid4().hex
            # Missing modes in legacy tasks meant off, irrespective of new-task defaults.
            options={k:data[k] for k in
                ('device','mcts_mode','context','full_classification','model') if k in data}
            options.setdefault('mcts_mode','off')
            # 旧目录任务的接收规则是 strict，不能套用新任务的 force_best 默认值。
            options['mcts'] = {'acceptance_policy': 'strict'}
            config=effective_config(options)
            created=data.get('created_at',time.time())
            deleted=created if path.parent.parent.name=='.trash' else None
            state=data['status']
            if state in ('uploading','running'): state='interrupted'
            has_result=state=='completed' and (path.parent/'result.json').exists()
            if state=='completed' and not has_result: state='failed'
            relative=f'jobs/{jid}/attempts/{aid}'
            dest=store.path(relative);dest.mkdir(parents=True,exist_ok=True)
            for filename in ('result.json','run.log'):
                source=path.parent/filename
                if source.is_file(): shutil.copy2(source,dest/filename)
            with store.connect(True) as db:
                db.execute('INSERT INTO videos VALUES(?,?,?,?,?,?,?)',(vid,file_hash(resolved),
                    data['filename'],str(resolved.relative_to(store.root)),
                    dumps(dict(duration=data.get('duration',0),fps=data.get('fps',0))),created,deleted))
                db.execute("""INSERT INTO jobs(id,video_id,status,config_json,config_hash,code_hash,
                    created_at,updated_at,current_attempt_id,result_attempt_id,deleted_at,error,request_hash)
                    VALUES(?,?,?,?,?,'legacy',?,?,?,?,?,?,?)""",
                    (jid,vid,state,dumps(config),digest(config),created,created,aid,
                     aid if has_result else None,deleted,data.get('error'),digest(dict(legacy=jid))))
                db.execute("""INSERT INTO attempts(id,job_id,number,state,token,started_at,
                    finished_at,artifact_dir,error) VALUES(?,?,1,?,?,?,?,?,?)""",
                    (aid,jid,state,uuid.uuid4().hex,data.get('started_at'),data.get('finished_at'),
                     relative,data.get('error')))
            imported+=1
        except Exception as exc:
            errors.append(dict(path=path.name,error=str(exc)))
    return dict(imported=imported,errors=errors)
