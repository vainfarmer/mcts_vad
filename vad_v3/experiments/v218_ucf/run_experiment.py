"""v218_ucf: frozen visual Yes/No scoring on all 290 videos, two video workers."""
import argparse
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from scoring import PROMPT, extract_score, windows

HERE=Path(__file__).resolve().parent
PILOT=HERE.parent/'auc_yesno_20261120'
ROOT=HERE.parents[2]
VIDEO_CONCURRENCY=2
API_CONCURRENCY=4


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare():
    parent=read(PILOT/'manifest.json')
    rows=read(PILOT/'selected_290.json')
    pilot=read(PILOT/'selected_65.json')
    assert len(rows)==290 and len({r['video_id'] for r in rows})==290
    assert sum(r['category']=='Normal' for r in rows)==150
    assert parent['prompt']==PROMPT
    assert sha(HERE/'scoring.py')==parent['source_sha256']['scoring.py']
    assert sha(PILOT/'run_experiment.py')==parent['source_sha256']['run_experiment.py']
    imported={}
    for r in pilot:
        p=PILOT/'results'/(r['video_id']+'.json');value=read(p)
        assert value['fingerprint']==parent['fingerprint']
        imported[r['video_id']]=dict(path=str(p),sha256=sha(p),fingerprint=parent['fingerprint'])
    payload=dict(version='v218_ucf',settings=parent['settings'],prompt=PROMPT,
                 runtime=dict(video_concurrency=VIDEO_CONCURRENCY,api_concurrency=API_CONCURRENCY,
                              api_retry_attempts=3,video_retry_rounds=1),
                 primary_arm='fusion_50',authorization='User explicitly requested full290, overriding prior pilot >80 gate.',
                 source_results=parent['source_results'],inputs=parent['inputs'],
                 full_ids=parent['full_ids'],pilot_ids=parent['pilot_ids'],
                 reused_pilot_results=imported,parent_manifest_sha256=sha(PILOT/'manifest.json'),
                 source_sha256={p.name:sha(p) for p in [Path(__file__),HERE/'scoring.py',HERE/'evaluate_scores.py']},
                 dependencies={k:__import__('importlib.metadata',fromlist=['version']).version(k) for k in ['openai','numpy','scikit-learn']}|{'opencv':cv2.__version__})
    payload['fingerprint']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    if (HERE/'manifest.json').exists():assert read(HERE/'manifest.json')==payload,'v218_ucf manifest changed'
    else:write(HERE/'manifest.json',payload)
    write(HERE/'selected_290.json',rows)
    write(HERE/'selected_65.json',pilot)
    write(HERE/'protocol.json',dict(
        version='v218_ucf',scope='All 290: 140 abnormal and 150 Normal. User-authorized full test regardless of pilot threshold.',
        reuse='65 exact pilot score outputs reused with hashes and original fingerprints; fresh visual inference on remaining225 (80 abnormal,145 Normal). All videos use the identical prompt/model/sampling/scoring settings.',
        concurrency='Two independent video workers; a shared four-request API semaphore, bounded pending work within each video. Each worker owns its own video decoder and request/result paths.',
        primary='fusion_50, chosen before full testing from the frozen pilot. Also report baseline, standalone yesno_probability, and hard Yes/No diagnostic. No selecting fusion weights using full-test labels.',
        inference='Category-blind images only: 10-second visual context, 5-second target step, up to10 frames. Same-position logP(Yes)/logP(No) normalization. No filenames, GT labels, VideoMAE labels, or historical captions in prompts.',
        evaluation='Pooled original-frame ROC-AUC with all Normal videos; Ano-AUC excludes pure Normal videos. Same native FPS/frame counts and inclusive annotation convention. Also report pilot65 and remaining225 separately.',
        baseline='Frozen historical MCTS+BUE/CBT frame curves. This is a new frame-scoring experiment, not a rerun or improvement claim for mAP or MCTS boundaries.',
        limitations='Previously used test videos, pilot is part of full290, cloud model alias is not a pinned backend. Historical baseline and fusion retain prior VideoMAE split-provenance concerns.'))
    for name,origin in imported.items():
        target=HERE/'results'/(name+'.json')
        value=read(Path(origin['path']));value['origin']=dict(kind='reused_pilot',**origin)
        value['fingerprint']=payload['fingerprint'];value['score_type']='yesno_probability'
        if target.exists():assert read(target)==value,'Imported score changed'
        else:write(target,value)
    if not (HERE/'source_snapshot.zip').exists():
        with zipfile.ZipFile(HERE/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in HERE.glob('*.py'):z.write(p,p.name)
    return rows,payload


def sample_content(cap,row,win,settings):
    positions=np.linspace(win['context_start'],win['context_end'],settings['frames_per_clip'])
    indices=np.unique(np.clip(np.rint(positions*row['fps']).astype(int),0,row['frame_count']-1))
    content=[];digest=hashlib.sha256()
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=cap.read()
        if not ok:raise RuntimeError('Source frame could not be read')
        ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,settings['jpeg_quality']])
        if not ok:raise RuntimeError('JPEG encoding failed')
        data=encoded.tobytes();digest.update(data)
        content.append({'type':'text','text':f'Frame time {index/row["fps"]:.3f} seconds.'})
        content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(data).decode()}})
    prompt=PROMPT+f'\nTarget interval: {win["start"]:.3f} to {win["end"]:.3f} seconds.'
    content.append({'type':'text','text':prompt});digest.update(prompt.encode())
    return content,digest.hexdigest(),indices.tolist()


async def infer(rows,manifest):
    load_dotenv(ROOT/'vad_v3/webapp/.env',override=False,encoding='utf-8-sig')
    if not os.getenv('DASHSCOPE_API_KEY'):raise RuntimeError('Configured API key missing')
    settings=manifest['settings'];started=time.time()
    api_sem=asyncio.Semaphore(API_CONCURRENCY)
    completed=set();pending_rows=[];active={};failures={}
    for r in rows:
        target=HERE/'results'/(r['video_id']+'.json')
        if target.exists():
            assert read(target)['fingerprint']==manifest['fingerprint']
            completed.add(r['video_id'])
        else:pending_rows.append(r)
    counts=dict(new_calls=0,cache_hits=0,failed_attempts=0,prompt_tokens=0,completion_tokens=0,windows_completed=0,
                active_api_requests=0,max_observed_api_requests=0,max_observed_video_workers=0)
    run_id=str(time.time_ns())

    def progress(state='running'):
        value=dict(version='v218_ucf',run_id=run_id,status=state,completed_videos=len(completed),total_videos=290,
                   reused_pilot_videos=65,remaining_videos=290-len(completed),active_videos=active,
                   failed_videos=failures,runtime=manifest['runtime'],counts=counts,
                   elapsed_seconds=round(time.time()-started,2))
        write(HERE/'progress.json',value)
        write(HERE/'runtime'/f'{run_id}.json',value)

    async with AsyncOpenAI(api_key=os.environ['DASHSCOPE_API_KEY'],base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',max_retries=0,timeout=90) as client:
        async def call_window(row,win,content,request_hash,frames):
            destination=HERE/'requests'/row['video_id']/f'{win["index"]:05d}.json'
            last_error=None
            for attempt in range(1,4):
                raw=None
                try:
                    async with api_sem:
                        counts['new_calls']+=1;counts['active_api_requests']+=1
                        counts['max_observed_api_requests']=max(counts['max_observed_api_requests'],counts['active_api_requests'])
                        try:
                            response=await client.chat.completions.create(model=settings['model'],
                                messages=[{'role':'user','content':content}],temperature=settings['temperature'],
                                max_tokens=settings['max_tokens'],logprobs=True,top_logprobs=5,
                                extra_body={'enable_thinking':False})
                        finally:counts['active_api_requests']-=1
                    raw=response.model_dump();usage=raw.get('usage') or {}
                    counts['prompt_tokens']+=usage.get('prompt_tokens',0)
                    counts['completion_tokens']+=usage.get('completion_tokens',0)
                    score=extract_score(raw,settings['max_censor_width'])
                    item=dict(fingerprint=manifest['fingerprint'],video_id=row['video_id'],window=win,
                              request_sha256=request_hash,frame_indices=frames,score=score,attempt=attempt,response=raw)
                    write(destination,item);counts['windows_completed']+=1
                    active[row['video_id']]['completed_windows']+=1
                    if counts['windows_completed']%20==0:progress()
                    return item
                except Exception as exc:
                    counts['failed_attempts']+=1;last_error=type(exc).__name__
                    write(HERE/'attempt_errors'/row['video_id']/f'{win["index"]:05d}_{time.time_ns()}.json',
                          dict(error_type=last_error,status_code=getattr(exc,'status_code',None),window=win,raw_response=raw))
                    if getattr(exc,'status_code',None) in (400,401,403,404):break
                    if attempt<3:await asyncio.sleep(2*attempt)
            raise RuntimeError(f'Window failed: {row["video_id"]}/{win["index"]} ({last_error})')

        async def video(original,worker):
            row={k:original[k] for k in ['video_id','path','fps','duration','frame_count','sha256']}
            n=row['video_id'];t0=time.time()
            assert await asyncio.to_thread(sha,Path(row['path']))==row['sha256'],'Source video changed'
            clips=windows(row['duration'],settings['stride_seconds'],settings['context_seconds'])
            active[n]=dict(worker=worker,windows=len(clips),completed_windows=0)
            counts['max_observed_video_workers']=max(counts['max_observed_video_workers'],len(active))
            assert len(active)<=VIDEO_CONCURRENCY
            progress();cap=cv2.VideoCapture(row['path']);assert cap.isOpened()
            pending=set();items=[]
            try:
                for win in clips:
                    cache=HERE/'requests'/n/f'{win["index"]:05d}.json'
                    if cache.exists():
                        item=read(cache);assert item['fingerprint']==manifest['fingerprint'] and item['window']==win
                        items.append(item);counts['cache_hits']+=1;active[n]['completed_windows']+=1
                        continue
                    content,request_hash,frames=await asyncio.to_thread(sample_content,cap,row,win,settings)
                    pending.add(asyncio.create_task(call_window(row,win,content,request_hash,frames)))
                    if len(pending)>=4:
                        done,pending=await asyncio.wait(pending,return_when=asyncio.FIRST_COMPLETED)
                        items.extend(task.result() for task in done)
                if pending:items.extend(await asyncio.gather(*pending))
                items.sort(key=lambda x:x['window']['index']);assert len(items)==len(clips)
                result=dict(video_id=n,fingerprint=manifest['fingerprint'],origin=dict(kind='new_v218',worker=worker),
                    score_type='yesno_probability',duration=row['duration'],timestamps=[x['window']['center'] for x in items],
                    frame_scores=[x['score']['score'] for x in items],score_lower=[x['score']['lower'] for x in items],
                    score_upper=[x['score']['upper'] for x in items],hard_scores=[x['score']['hard_score'] for x in items],
                    exact_pairs=sum(x['score']['kind']=='exact_pair' for x in items),bounded_pairs=sum(x['score']['kind']!='exact_pair' for x in items),
                    prompt_tokens=sum(x['response']['usage']['prompt_tokens'] for x in items),
                    completion_tokens=sum(x['response']['usage']['completion_tokens'] for x in items),processing_seconds=round(time.time()-t0,3))
                write(HERE/'results'/(n+'.json'),result);completed.add(n);failures.pop(n,None)
                print(json.dumps(dict(version='v218_ucf',worker=worker,video=n,completed=len(completed),total=290,
                    windows=len(clips),exact=result['exact_pairs'],bounded=result['bounded_pairs'],elapsed=round(time.time()-started,1))),flush=True)
            finally:
                for task in pending:
                    if not task.done():task.cancel()
                if pending:await asyncio.gather(*pending,return_exceptions=True)
                cap.release();active.pop(n,None);progress()

        async def round_run(todo):
            queue=asyncio.Queue()
            for row in todo:queue.put_nowait(row)
            async def worker(i):
                while not queue.empty():
                    row=queue.get_nowait()
                    try:await video(row,i)
                    except Exception as exc:
                        failures[row['video_id']]=dict(error_type=type(exc).__name__)
                        active.pop(row['video_id'],None);progress()
                        print(json.dumps(dict(video=row['video_id'],status='failed',error_type=type(exc).__name__)),flush=True)
                    finally:queue.task_done()
            await asyncio.gather(*(worker(i) for i in range(VIDEO_CONCURRENCY)))
        progress()
        await round_run(pending_rows)
        if failures:
            await round_run([r for r in rows if r['video_id'] in failures])
        if failures:
            progress('failed');raise RuntimeError('Incomplete v218_ucf videos; resumable caches retained')
        assert len(completed)==290
        assert counts['max_observed_video_workers']<=2 and counts['max_observed_api_requests']<=4
        progress('completed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    rows,manifest=prepare()
    if args.prepare:return
    asyncio.run(infer(rows,manifest))
    from evaluate_scores import evaluate
    evaluate(rows,manifest,'full290',selected_arm=manifest['primary_arm'])
    pilot_ids=set(manifest['pilot_ids'])
    evaluate([r for r in rows if r['video_id'] in pilot_ids],manifest,'pilot65',selected_arm=manifest['primary_arm'])
    evaluate([r for r in rows if r['video_id'] not in pilot_ids],manifest,'remaining225',selected_arm=manifest['primary_arm'])
    write(HERE/'exit_status.json',dict(status='completed',version='v218_ucf',videos=290))


if __name__=='__main__':main()
