"""SQLite persistence. Each operation owns a short transaction/connection."""
from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
import uuid
from .settings import ACTIVE, atomic_json, digest, dumps, safe_path


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / 'system.sqlite3'
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY);
            INSERT OR IGNORE INTO schema_version VALUES(1);
            CREATE TABLE IF NOT EXISTS videos(
                id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, filename TEXT NOT NULL,
                path TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at REAL NOT NULL,
                deleted_at REAL);
            CREATE INDEX IF NOT EXISTS video_hash ON videos(sha256);
            CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, video_id TEXT NOT NULL REFERENCES videos(id),
                status TEXT NOT NULL, config_json TEXT NOT NULL, config_hash TEXT NOT NULL,
                code_hash TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                current_attempt_id TEXT, result_attempt_id TEXT, deleted_at REAL, error TEXT,
                idempotency_key TEXT UNIQUE, request_hash TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS queue ON jobs(status,created_at);
            CREATE TABLE IF NOT EXISTS attempts(
                id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), number INTEGER NOT NULL,
                state TEXT NOT NULL, token TEXT NOT NULL UNIQUE, worker_id TEXT,
                heartbeat_at REAL, started_at REAL, finished_at REAL, pid INTEGER,
                process_created REAL, exit_code INTEGER, error TEXT, artifact_dir TEXT NOT NULL,
                cancel_requested INTEGER NOT NULL DEFAULT 0, metrics_json TEXT,
                UNIQUE(job_id,number));
            CREATE TABLE IF NOT EXISTS reviews(
                id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                subject_id TEXT NOT NULL, revision INTEGER NOT NULL, action TEXT NOT NULL,
                before_json TEXT, after_json TEXT, reason TEXT NOT NULL,
                reviewer TEXT NOT NULL, created_at REAL NOT NULL,
                UNIQUE(job_id,subject_id,revision));
            CREATE TABLE IF NOT EXISTS experiments(
                id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL,
                manifest_json TEXT NOT NULL, status TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evaluations(
                id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
                protocol_json TEXT NOT NULL, input_hash TEXT NOT NULL,
                created_at REAL NOT NULL, status TEXT NOT NULL, metrics_path TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS service_state(
                name TEXT PRIMARY KEY, payload_json TEXT NOT NULL, updated_at REAL NOT NULL);
            """)

    @contextmanager
    def connect(self, immediate=False):
        db = sqlite3.connect(self.database, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA busy_timeout=15000')
        try:
            if immediate:
                db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def unpack(row):
        if row is None:
            raise FileNotFoundError('记录不存在。')
        result = dict(row)
        for key in list(result):
            if key.endswith('_json'):
                value = result.pop(key)
                result[key[:-5]] = json.loads(value) if value is not None else None
        return result

    def path(self, relative):
        return safe_path(self.root, relative)

    def video(self, video_id, deleted=False):
        with self.connect() as db:
            return self.unpack(db.execute('SELECT * FROM videos WHERE id=?' +
                ('' if deleted else ' AND deleted_at IS NULL'), (video_id,)).fetchone())

    def videos(self, search='', page=1, deleted=False):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM videos WHERE deleted_at IS ' +
                ('NOT NULL' if deleted else 'NULL') +
                ' AND filename LIKE ? ORDER BY created_at DESC LIMIT 30 OFFSET ?',
                ('%' + search + '%', (page-1)*30)).fetchall()
            return [self.unpack(r) for r in rows]

    def add_video(self, video_id, filename, relative, sha, metadata):
        with self.connect() as db:
            db.execute('INSERT INTO videos VALUES(?,?,?,?,?,?,NULL)',
                       (video_id, sha, filename, relative, dumps(metadata), time.time()))
        return self.video(video_id)

    def create_job(self, video_id, config, source_hash, key=None):
        if key is not None and (not isinstance(key, str) or not 8 <= len(key) <= 200):
            raise ValueError('幂等键长度应为 8—200。')
        request_hash = digest(dict(video_id=video_id, config=config))
        now = time.time()
        with self.connect(True) as db:
            if key:
                existing = db.execute('SELECT * FROM jobs WHERE idempotency_key=?', (key,)).fetchone()
                if existing:
                    if existing['request_hash'] != request_hash:
                        raise Conflict('该幂等键已用于不同的请求。')
                    return self.unpack(existing), False
            if not db.execute('SELECT id FROM videos WHERE id=? AND deleted_at IS NULL',
                              (video_id,)).fetchone():
                raise FileNotFoundError('视频不存在。')
            job_id = uuid.uuid4().hex
            db.execute("""INSERT INTO jobs(id,video_id,status,config_json,config_hash,code_hash,
                created_at,updated_at,idempotency_key,request_hash) VALUES(?,?,'queued',?,?,?,?,?,?,?)""",
                (job_id,video_id,dumps(config),digest(config),source_hash,now,now,key,request_hash))
        return self.job(job_id), True

    def job(self, job_id, deleted=False):
        with self.connect() as db:
            return self.unpack(db.execute('SELECT * FROM jobs WHERE id=?' +
                ('' if deleted else ' AND deleted_at IS NULL'), (job_id,)).fetchone())

    def jobs(self, status='', search='', page=1, deleted=False, video_id=''):
        with self.connect() as db:
            rows = db.execute("""SELECT j.*,v.filename FROM jobs j JOIN videos v ON v.id=j.video_id
                WHERE j.deleted_at IS """ + ('NOT NULL' if deleted else 'NULL') + """
                AND (?='' OR j.status=?) AND v.filename LIKE ?
                AND (?='' OR j.video_id=?) ORDER BY j.created_at DESC LIMIT 30 OFFSET ?""",
                (status,status,'%'+search+'%',video_id,video_id,(page-1)*30)).fetchall()
            return [self.unpack(r) for r in rows]

    def attempts(self, job_id):
        with self.connect() as db:
            return [self.unpack(r) for r in db.execute(
                'SELECT * FROM attempts WHERE job_id=? ORDER BY number', (job_id,))]

    def attempt(self, attempt_id):
        with self.connect() as db:
            return self.unpack(db.execute('SELECT * FROM attempts WHERE id=?', (attempt_id,)).fetchone())

    def claim(self, worker_id):
        now = time.time()
        with self.connect(True) as db:
            # Defence in depth in addition to the OS-held worker lifetime lock.
            if db.execute("SELECT 1 FROM attempts WHERE state IN ('running','cancelling')").fetchone():
                return None
            row = db.execute("""SELECT * FROM jobs WHERE status='queued' AND deleted_at IS NULL
                                ORDER BY created_at,id LIMIT 1""").fetchone()
            if not row:
                return None
            aid, token = uuid.uuid4().hex, uuid.uuid4().hex
            number = db.execute('SELECT COALESCE(MAX(number),0)+1 FROM attempts WHERE job_id=?',
                                (row['id'],)).fetchone()[0]
            relative = f"jobs/{row['id']}/attempts/{aid}"
            db.execute("""INSERT INTO attempts(id,job_id,number,state,token,worker_id,heartbeat_at,
                started_at,artifact_dir) VALUES(?,?,?,'running',?,?,?,?,?)""",
                (aid,row['id'],number,token,worker_id,now,now,relative))
            db.execute("UPDATE jobs SET status='running',current_attempt_id=?,updated_at=?,error=NULL WHERE id=?",
                       (aid,now,row['id']))
        return self.job(row['id']), self.attempt(aid)

    def heartbeat(self, aid, pid=None, process_created=None, metrics=None):
        with self.connect() as db:
            db.execute("""UPDATE attempts SET heartbeat_at=?,pid=COALESCE(?,pid),
                process_created=COALESCE(?,process_created),metrics_json=COALESCE(?,metrics_json)
                WHERE id=? AND state IN ('running','cancelling')""",
                (time.time(),pid,process_created,dumps(metrics) if metrics is not None else None,aid))

    def finish(self, aid, token, state, error=None, exit_code=None):
        if state not in ('completed','failed','cancelled','interrupted'):
            raise ValueError('无效终止状态。')
        now = time.time()
        with self.connect(True) as db:
            a = db.execute('SELECT * FROM attempts WHERE id=? AND token=?', (aid,token)).fetchone()
            if not a or a['state'] not in ('running','cancelling'):
                raise Conflict('执行令牌已失效。')
            if a['cancel_requested'] and state == 'completed':
                state = 'cancelled'
            db.execute('UPDATE attempts SET state=?,error=?,exit_code=?,finished_at=? WHERE id=?',
                       (state,error,exit_code,now,aid))
            db.execute("""UPDATE jobs SET status=?,error=?,updated_at=?,
                result_attempt_id=CASE WHEN ?='completed' THEN ? ELSE result_attempt_id END
                WHERE id=? AND current_attempt_id=?""",
                (state,error,now,state,aid,a['job_id'],aid))

    def cancel(self, job_id):
        with self.connect(True) as db:
            j = self.unpack(db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())
            if j['status'] == 'queued':
                db.execute("UPDATE jobs SET status='cancelled',updated_at=? WHERE id=?", (time.time(),job_id))
            elif j['status'] in ('running','cancelling'):
                db.execute("UPDATE jobs SET status='cancelling',updated_at=? WHERE id=?", (time.time(),job_id))
                db.execute("UPDATE attempts SET state='cancelling',cancel_requested=1 WHERE id=?",
                           (j['current_attempt_id'],))
            else:
                raise Conflict('只有排队或运行任务可以取消。')

    def retry(self, job_id, source_hash):
        with self.connect(True) as db:
            j = self.unpack(db.execute('SELECT * FROM jobs WHERE id=? AND deleted_at IS NULL',
                                      (job_id,)).fetchone())
            if j['status'] not in ('failed','interrupted','cancelled'):
                raise Conflict('该状态不能重试。')
            if j['code_hash'] != source_hash and j['code_hash'] != 'legacy':
                raise Conflict('算法代码已变化，请用原视频创建新任务以记录新版本。')
            db.execute("UPDATE jobs SET status='queued',error=NULL,updated_at=? WHERE id=?", (time.time(),job_id))

    def delete_job(self, job_id, restore=False):
        with self.connect(True) as db:
            j = self.unpack(db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())
            if j['status'] in ACTIVE:
                raise Conflict('请先取消任务并等待执行停止。')
            db.execute('UPDATE jobs SET deleted_at=?,updated_at=? WHERE id=?',
                       (None if restore else time.time(),time.time(),job_id))

    def delete_video(self, video_id, restore=False):
        self.video(video_id, deleted=True)
        with self.connect(True) as db:
            if not restore and db.execute("""SELECT 1 FROM jobs WHERE video_id=?
                AND status IN ('queued','running','cancelling')""", (video_id,)).fetchone():
                raise Conflict('视频仍有活动任务。')
            db.execute('UPDATE videos SET deleted_at=? WHERE id=?',
                       (None if restore else time.time(),video_id))

    def result(self, job_id):
        j = self.job(job_id)
        if not j['result_attempt_id']:
            raise Conflict('尚无已提交结果。')
        a = self.attempt(j['result_attempt_id'])
        return json.loads(self.path(a['artifact_dir'] + '/result.json').read_text(encoding='utf-8'))

    def reviews(self, job_id):
        self.job(job_id)
        with self.connect() as db:
            return [self.unpack(r) for r in db.execute(
                'SELECT * FROM reviews WHERE job_id=? ORDER BY created_at,id', (job_id,))]

    def add_review(self, job_id, data):
        if not isinstance(data,dict):
            raise ValueError('复核内容必须是对象。')
        result = self.result(job_id)
        subject = data.get('subject_id') or 'manual-' + uuid.uuid4().hex
        revision = data.get('expected_revision', 0)
        action = data.get('action')
        if action not in ('confirmed','corrected','rejected','added'):
            raise ValueError('无效复核操作。')
        if not isinstance(subject,str) or len(subject)>100 or type(revision) is not int:
            raise ValueError('无效复核对象或版本。')
        reason = str(data.get('reason',''))[:2000]
        reviewer = str(data.get('reviewer','local'))[:100]
        after = data.get('after')
        from .results import validate_detection
        if action in ('added','corrected'):
            validate_detection(after, result['duration'])
        with self.connect(True) as db:
            previous = db.execute("""SELECT * FROM reviews WHERE job_id=? AND subject_id=?
                ORDER BY revision DESC LIMIT 1""",(job_id,subject)).fetchone()
            current = previous['revision'] if previous else 0
            if current != revision:
                raise Conflict('复核版本已更新，请刷新后重试。')
            before = json.loads(previous['after_json']) if previous and previous['after_json'] else None
            if not previous and not subject.startswith('manual-'):
                try:
                    index = int(subject.removeprefix('det-'))
                    if subject != f'det-{index}' or index < 0:
                        raise ValueError()
                    before = result['anomalies'][index]
                except (ValueError,IndexError):
                    raise ValueError('检测对象不存在。')
            if not previous and subject.startswith('manual-') and action != 'added':
                raise ValueError('新人工区间必须使用 added 操作。')
            if action == 'confirmed':
                after = before
            elif action == 'rejected':
                after = None
            rid = uuid.uuid4().hex
            db.execute('INSERT INTO reviews VALUES(?,?,?,?,?,?,?,?,?,?)',
                (rid,job_id,subject,current+1,action,dumps(before),dumps(after),reason,reviewer,time.time()))
        return dict(id=rid,subject_id=subject,revision=current+1)

    def service(self, name, payload=None):
        with self.connect() as db:
            if payload is not None:
                db.execute('INSERT OR REPLACE INTO service_state VALUES(?,?,?)',
                           (name,dumps(payload),time.time()))
            row = db.execute('SELECT * FROM service_state WHERE name=?',(name,)).fetchone()
            return self.unpack(row) if row else None
