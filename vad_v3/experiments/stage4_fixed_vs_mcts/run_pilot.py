"""Stage 4 -> fixed BUE/CBT versus Stage 4 -> original MCTS -> fixed BUE/CBT.

Run from the repository root with the project's common Python environment.
No dataset labels are passed into the detector. --prepare only freezes inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CODE=ROOT/'vad_v3/video_anomaly_detection'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(CODE))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-dir',default='F:/dataset/ucf_crime')
    parser.add_argument('--output-dir',type=Path,default=HERE/'results')
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'vad_v3/webapp/.env',override=False,encoding='utf-8-sig')
    from batch_evaluate import select_videos
    import cv2
    selected=select_videos(args.dataset_dir,1)
    rows=[]
    for _,row in selected.iterrows():
        path=Path(row.video_path)
        cap=cv2.VideoCapture(str(path))
        fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        if fps<=0 or frames<=0: raise ValueError('Invalid video metadata: '+str(path))
        with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
        rows.append(dict(video_id=path.stem,path=str(path),category=row.category,
            fps=fps,frame_count=frames,duration=frames/fps,sha256=digest))
    selection=HERE/'selected_videos.json'
    if selection.exists() and json.loads(selection.read_text(encoding='utf-8'))!=rows:
        raise ValueError('Selected inputs changed; use a new experiment directory')
    write(selection,rows)
    command=[sys.executable,'-u',str(CODE/'batch_evaluate.py'),
        '--dataset_dir',args.dataset_dir,'--output_dir',str(args.output_dir.resolve()),
        '--max_per_category','1','--concurrency','1','--device','cuda',
        '--detect_model','qwen3-vl-plus','--classify_model','qwen3-vl-plus',
        '--mcts_mode','refine','--mcts_action_set','shrink','--mcts_paired_eval','--mcts_stage4_direct_eval',
        '--mcts_trace','--mcts_simulations','8','--mcts_max_depth','4',
        '--mcts_trigger_policy','all_valid','--mcts_acceptance_policy','strict',
        '--mcts_max_candidates_per_video','2','--mcts_max_vlm_evaluations_per_candidate','10',
        '--mcts_max_vlm_evaluations_per_video','20','--mcts_max_images_per_evaluation','32',
        '--mcts_max_boundary_gap_seconds','1.5','--mcts_max_evidence_gap_seconds','3.0',
        '--api_concurrency','2','--video_retry_rounds','1']
    write(HERE/'COMMAND.json',command)
    previous=json.loads((HERE.parent/'bue_cbt_pilot/selected_videos.json').read_text(encoding='utf-8'))
    if rows!=previous:raise ValueError('Inputs differ from the previous one-video-per-class pilot')
    write(HERE/'protocol.json',dict(selection='Same 14 source videos as the previous pilot; file hashes verified; test split sample(n=1, random_state=42) per class.',
        comparison='Stage 4 raw candidates -> fixed BUE -> CBT versus the same candidates -> original shrink MCTS -> fixed BUE -> CBT.',
        skipped_stages='Stage 5 CLIP boundaries, Stage 5b, intermediate tighten/merge/NMS, candidate top-k, classification VLM, confidence gating and QSCORE are bypassed in BOTH arms.',
        action_geometry='Original shrink actions only; seed-relative +/-25% endpoint bounds; no BUE/CBT search actions.',
        interpretation='Both arms use all Stage 4 candidates, identical fixed postprocessing, predicted upstream category and frozen discovery confidence. GT is used only after inference.',
        priors='Existing frozen category caps; no tuning on selected labels. Provenance of older GT-derived caps is not independently established.',
        evidence='Existing v217 coverage settings (32 images, boundary gap 1.5s, evidence gap 3s); strict acceptance; all_valid trigger.',
        cost='Result token_usage and processing_time cover the entire paired run, not an individual arm.',
        evaluation='VadCLIP rank AP, class-aware 13 abnormal classes, IoU 0.1:0.1:0.5, source FPS, integer half-open intervals. Normal false positives reported separately.',
        reference='https://raw.githubusercontent.com/nwpu-zxr/VadCLIP/main/src/utils/ucf_detectionMAP.py',
        limitations='One test video per class is a smoke/pilot comparison, not a full benchmark or significance test.'))
    print(json.dumps([dict(category=r['category'],video=r['video_id'],duration=round(r['duration'],1)) for r in rows],ensure_ascii=False),flush=True)
    if args.prepare:return
    if not os.getenv('DASHSCOPE_API_KEY'): raise RuntimeError('Missing API key')
    from vad_v3.webapp.store import Store
    from vad_v3.webapp.worker import InstanceLock
    from vad_v3.webapp.service import read_runtime,alive
    from vad_v3.webapp.settings import atomic_json
    import psutil
    store=Store(ROOT/'vad_v3/web_data');runtime=read_runtime(store)
    resume_worker=alive(runtime.get('worker'))
    with store.connect() as db:
        if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running','cancelling')").fetchone()[0]:
            raise RuntimeError('Web jobs are active; wait before running the GPU pilot')
    try:
        if resume_worker:
            store.service('control',dict(stop=True))
            deadline=time.monotonic()+15
            while alive(runtime.get('worker')) and time.monotonic()<deadline:time.sleep(.2)
            if alive(runtime.get('worker')):raise RuntimeError('Worker did not release GPU lock')
        with InstanceLock(store.root/'.worker.lock'):
            with zipfile.ZipFile(HERE/'executed_source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
                for source in sorted(CODE.glob('*.py')):
                    archive.write(source,arcname='video_anomaly_detection/'+source.name)
                for source in sorted((ROOT/'evaluation_tools').glob('*.py')):
                    archive.write(source,arcname='evaluation_tools/'+source.name)
            env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1')
            with (HERE/'console.log').open('a',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(HERE/'exit_status.json',dict(returncode=result.returncode,finished_at=time.time()))
            if result.returncode:raise RuntimeError('Pilot failed; inspect console.log')
    finally:
        if resume_worker:store.service('control',dict(stop=False))
        if resume_worker and not alive(runtime.get('worker')):
            flags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
            with (store.root/'worker.service.log').open('a',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,'-u','-m','vad_v3.webapp.worker','--data-dir',str(store.root)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'),creationflags=flags)
            runtime['worker']=dict(pid=process.pid,created=psutil.Process(process.pid).create_time())
            atomic_json(store.root/'runtime.json',runtime)


if __name__=='__main__':main()
