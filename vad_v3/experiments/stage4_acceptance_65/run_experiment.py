"""65-video live comparison: Stage 4 -> off/strict/relaxed_node/force_best -> BUE/CBT."""
import argparse
from collections import Counter
import hashlib
import importlib.util
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
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--dataset-dir',default='F:/dataset/ucf_crime')
    args=parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'vad_v3/webapp/.env',override=False,encoding='utf-8-sig')
    from batch_evaluate import select_videos
    import cv2
    selected=select_videos(args.dataset_dir,5)
    assert len(selected)==65,'Dataset selection changed'
    rows=[]
    for _,row in selected.iterrows():
        path=Path(row.video_path)
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS)
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        if fps<=0 or frames<=0:raise ValueError('Invalid source video: '+str(path))
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        rows.append(dict(video_id=path.stem,path=str(path),category=row.category,
            fps=fps,frame_count=frames,duration=frames/fps,sha256=digest))
    prior=set()
    for path in (HERE.parent/'v215/results').glob('*.json'):
        if 'paired_results' in json.loads(path.read_text(encoding='utf-8')):prior.add(path.stem)
    assert prior=={r['video_id'] for r in rows},'Selection differs from historical 65-video set'
    selection=HERE/'selected_videos.json'
    if selection.exists() and json.loads(selection.read_text(encoding='utf-8'))!=rows:
        raise ValueError('Input files changed; use a fresh experiment directory')
    write(selection,rows)
    command=[sys.executable,'-u',str(CODE/'batch_evaluate.py'),
        '--dataset_dir',args.dataset_dir,'--output_dir',str(HERE/'results'),
        '--max_per_category','5','--concurrency','1','--device','cuda',
        '--detect_model','qwen3-vl-plus','--classify_model','qwen3-vl-plus',
        '--mcts_mode','refine','--mcts_action_set','shrink','--mcts_paired_eval',
        '--mcts_stage4_direct_eval','--mcts_stage4_acceptance_eval',
        '--mcts_trace','--mcts_simulations','8','--mcts_max_depth','4',
        '--mcts_trigger_policy','all_valid','--mcts_acceptance_policy','strict',
        '--mcts_max_candidates_per_video','2','--mcts_max_vlm_evaluations_per_candidate','10',
        '--mcts_max_vlm_evaluations_per_video','20','--mcts_max_images_per_evaluation','32',
        '--mcts_max_boundary_gap_seconds','1.5','--mcts_max_evidence_gap_seconds','3.0',
        '--api_concurrency','2','--video_retry_rounds','1']
    write(HERE/'COMMAND.json',command)
    write(HERE/'protocol.json',dict(
        scope='Stage 4 raw candidates -> optional MCTS -> identical fixed BUE -> CBT',
        arms=['off','strict','relaxed_node','force_best'],
        selection='Historical 65-video selection reproduced by max_per_category=5, random_state=42; 60 abnormal and 5 normal; Abuse=2 and Assault=3 due to test-set availability.',
        category_counts=dict(Counter(r['category'] for r in rows)),
        pairing='One shared upstream discovery; off completes before any search. Each active policy independently executes the current MCTS refiner with its own budget/tree and video state.',
        api_replay='Per-video exact-request cache: only identical model settings, prompts and image bytes share successful replies. This is a live paired run, not frozen-tree replay.',
        frozen='All Stage 4 candidates retained, discovery confidences and predicted categories fixed. Stage 5 and intermediate refinements/top-k/NMS/classification/gating/QSCORE bypassed.',
        search='shrink, all_valid, 8 simulations, depth 4, 2 candidates/video, 10 logical requests/candidate and 20/video, 32 evidence images, boundary gap 1.5s/evidence gap 3s.',
        evaluation='Current VadCLIP noninterpolated class-aware AP; 13 abnormal classes, native FPS, integer half-open frames, IoU 0.1:0.1:0.5. Normal false positives reported separately.',
        costs='Whole-run tokens/latency are actual shared four-arm usage. Later arms benefit from cache; branch time/tokens are not standalone deployment comparisons.',
        limitations='65-video subset, not full test set. Includes earlier pilot inputs; not an independent holdout. Existing CBT priors unchanged; older provenance unverified. GT only used for evaluation.'))
    print(json.dumps(dict(videos=len(rows),categories=Counter(r['category'] for r in rows),
        total_source_hours=sum(r['duration'] for r in rows)/3600),ensure_ascii=False),flush=True)
    if args.prepare:return
    if not os.getenv('DASHSCOPE_API_KEY'):raise RuntimeError('Missing configured API key')
    helper=HERE.parent/'stage4_acceptance_ablation/run_replay.py'
    spec=importlib.util.spec_from_file_location('acceptance_gpu_slot',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with module.gpu_slot():
        archive_path=HERE/'executed_source.zip'
        if not archive_path.exists():
            with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
                for source in sorted(CODE.glob('*.py')):archive.write(source,'video_anomaly_detection/'+source.name)
                for source in sorted((ROOT/'evaluation_tools').glob('*.py')):archive.write(source,'evaluation_tools/'+source.name)
                archive.write(Path(__file__),'experiment/run_experiment.py')
                archive.write(helper,'experiment/gpu_slot_source.py')
        else:
            with zipfile.ZipFile(archive_path) as archive:
                for source in CODE.glob('*.py'):
                    assert archive.read('video_anomaly_detection/'+source.name)==source.read_bytes(),source.name
        env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1')
        with (HERE/'console.log').open('a',encoding='utf-8') as log:
            process=subprocess.run(command,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(HERE/'exit_status.json',dict(returncode=process.returncode,finished_at=time.time()))
        if process.returncode:raise RuntimeError('Run incomplete; inspect console.log before resuming')


if __name__=='__main__':main()
