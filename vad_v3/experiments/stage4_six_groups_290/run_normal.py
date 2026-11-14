"""补测 150 个正常视频：四个搜索策略及两个无后处理分支，视频并发为 2。

沿用现有 batch_evaluate 的重试、请求缓存、独立视频上下文和断点恢复。
实验包装只额外记录同一视频状态上的两个帧评分，不改变原四组推理。
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CODE=ROOT/'vad_v3/video_anomaly_detection'
PREVIOUS=HERE.parent/'stage4_acceptance_140'
ABNORMAL=HERE.parent/'stage4_six_groups_140'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(CODE))
os.environ['HF_HUB_OFFLINE']='1'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def prepare():
    from batch_evaluate import select_videos
    from evaluation_tools.video_fps import probe_video_metadata
    # 当前推理实现与已有 140 个视频保持相同；只允许评估/批处理入口变化。
    with zipfile.ZipFile(PREVIOUS/'executed_source.zip') as archive:
        for source in sorted(CODE.glob('*.py')):
            name='video_anomaly_detection/'+source.name
            if source.name not in ('evaluate.py','batch_evaluate.py'):
                assert name in archive.namelist(),name
                assert ast.dump(ast.parse(archive.read(name)))==ast.dump(ast.parse(source.read_bytes())),name
    rows=[]
    for _,row in select_videos('F:/dataset/ucf_crime',100000,categories=['Normal']).iterrows():
        meta=probe_video_metadata(row.video_path)
        rows.append(dict(video_id=Path(row.video_path).stem,path=row.video_path,category='Normal',
                         sha256=sha(row.video_path),**{k:meta[k] for k in ('fps','frame_count','duration')}))
    assert len(rows)==150 and len({r['video_id'] for r in rows})==150
    old=read(PREVIOUS/'results/metadata/manifest.json')
    payload=dict(normal_videos=rows,config=old['config'],abnormal_source=str(ABNORMAL),
        abnormal_results_sha256={p.stem:sha(p) for p in sorted((ABNORMAL/'results').glob('*.json'))},
        source_sha256={p.name:sha(p) for p in sorted(CODE.glob('*.py'))},
        runner_sha256=sha(__file__),video_concurrency=2,api_concurrency=2,
        temporal_ap_sha256=sha(ROOT/'evaluation_tools/temporal_ap.py'))
    assert len(payload['abnormal_results_sha256'])==140
    payload['fingerprint']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    if (HERE/'manifest.json').exists():assert read(HERE/'manifest.json')==payload,'Experiment changed'
    else:write(HERE/'manifest.json',payload)
    write(HERE/'selected_normal_videos.json',rows)
    write(HERE/'config.json',old['config'])
    command=read(PREVIOUS/'COMMAND.json')[2:]
    for flag,value in (('--output_dir',str(HERE/'normal_results')),('--categories','Normal'),('--concurrency','2')):
        command[command.index(flag)+1]=value
    write(HERE/'COMMAND.json',[sys.executable,str(Path(__file__)),dict(batch_argv=command)])
    write(HERE/'protocol.json',dict(
        scope='140 frozen abnormal videos plus 150 freshly tested normal videos; video concurrency 2, API concurrency 2.',
        primary_map='All 290 videos; predictions on normal videos are false positives for their predicted anomaly class. Macro average over 13 anomaly classes, not a synthetic Normal detection class.',
        additional_map='Same 140 abnormal videos under the new sorting rule, for a controlled scope comparison.',
        sorting='Per-class input ordered by video ID, retaining detection order within a video; np.argsort(-confidence) with default kind, no secondary tie keys.',
        coordinates='Source-video FPS/frame count; integer half-open frames; GT unchanged.',
        six_groups='Stage 4 -> off/strict/relaxed_node/force_best -> BUE -> CBT; Stage 4 -> force_best stop; Stage 4 stop.',
        inference='Same Stage 1-4, candidate identities/confidences/categories and search budgets as prior 140. Normal ground-truth labels only select inputs; detector receives only video_path.',
        extras='Two no-tail frame scores are computed from the same live video features; no additional VLM search.',
        provenance='Abnormal outputs are frozen historical runs, normals are new requests; not a single simultaneous fresh benchmark. Existing test-set tuning/CBT-prior limitations remain.'))
    archive_path=HERE/'executed_source.zip'
    if not archive_path.exists():
        with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(CODE.glob('*.py')):archive.write(p,'video_anomaly_detection/'+p.name)
            for p in sorted((ROOT/'evaluation_tools').glob('*.py')):archive.write(p,'evaluation_tools/'+p.name)
            archive.write(__file__,'experiment/run_normal.py')
    return command,old


def install_recording_wrapper(expected):
    import factory
    from pipeline import MultiAnomalyVAD
    from video_state import fork_video_state
    original_build=factory.build_vad
    def checked_build(config,detect_client,classify_client):
        current=read(HERE/'normal_results/metadata/manifest.json')
        for key in ('config','models','api_settings','local_model_sha256','packages','cached_model_revisions'):
            assert current[key]==expected[key],f'Inference inputs changed: {key}'
        return original_build(config,detect_client,classify_client)
    factory.build_vad=checked_build
    original=MultiAnomalyVAD._compare_stage4_boundaries
    async def six_groups(self,candidates,video_data,video_path,category):
        result=await original(self,candidates,video_data,video_path,category)
        stages=result.paired_results['force_best']['mcts_trace']['stages']
        for group,stage in (('force_best_no_tail','post_mcts'),('stage4_only','post_stage4')):
            rows=copy.deepcopy(stages[stage])
            scores=self.frame_scorer.compute_frame_scores(fork_video_state(video_data),rows,category)
            result.paired_results[group]=dict(video_path=video_path,duration=video_data['duration'],
                is_normal=not rows,anomalies=rows,frame_scores=scores,timestamps=video_data['timestamps'].tolist(),
                frame_scores_stage=stage)
        return result
    MultiAnomalyVAD._compare_stage4_boundaries=six_groups


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'vad_v3/webapp/.env',override=False,encoding='utf-8-sig')
    command,expected=prepare()
    if args.prepare:return
    if not os.getenv('DASHSCOPE_API_KEY'):raise RuntimeError('Configured API key missing')
    helper=HERE.parent/'stage4_acceptance_ablation/run_replay.py'
    spec=importlib.util.spec_from_file_location('normal_gpu_slot',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        with module.gpu_slot():
            install_recording_wrapper(expected)
            sys.argv=command
            runpy.run_path(str(CODE/'batch_evaluate.py'),run_name='__main__')
        write(HERE/'exit_status.json',dict(status='completed',finished_at=time.time()))
    except BaseException as exc:
        write(HERE/'exit_status.json',dict(status='failed',error_type=type(exc).__name__,finished_at=time.time()))
        raise


if __name__=='__main__':main()
