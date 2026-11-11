"""All 140 abnormal test videos; verified reuse of 60 + live four-arm inference of 80."""
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
from types import SimpleNamespace
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CODE=ROOT/'vad_v3/video_anomaly_detection'
PREVIOUS=HERE.parent/'stage4_acceptance_65'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(CODE))


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--dataset-dir',default='F:/dataset/ucf_crime');args=parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'vad_v3/webapp/.env',override=False,encoding='utf-8-sig')
    from batch_evaluate import select_videos
    from run import UCF_CRIME_CATEGORIES
    from config import VADConfig
    from experiment import prepare_manifest
    import cv2
    selected=select_videos(args.dataset_dir,100000,categories=UCF_CRIME_CATEGORIES)
    assert len(selected)==140 and set(selected['category'])==set(UCF_CRIME_CATEGORIES)
    assert 'Normal' not in set(selected['category'])
    rows=[]
    for _,row in selected.iterrows():
        path=Path(row.video_path)
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS)
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        if fps<=0 or frames<=0:raise ValueError('Invalid source video: '+str(path))
        rows.append(dict(video_id=path.stem,path=str(path),category=row.category,
            fps=fps,frame_count=frames,duration=frames/fps,sha256=sha(path)))
    selection=HERE/'selected_videos.json'
    if selection.exists() and read(selection)!=rows:
        raise ValueError('Source videos changed; use a fresh experiment directory')
    write(selection,rows)
    old=read(PREVIOUS/'results/metadata/manifest.json')
    write(HERE/'config.json',old['config']);cfg=VADConfig.load(HERE/'config.json')
    clients=[SimpleNamespace(**{k:v for k,v in model.items() if k!='base_url'},
        client=SimpleNamespace(base_url=model['base_url']),api_settings=api)
        for model,api in zip(old['models'],old['api_settings'])]
    manifest=prepare_manifest(HERE/'results/metadata/manifest.json',cfg,clients,
        [r['path'] for r in rows],runtime_settings=old['runtime_settings'])
    # Videos/fingerprint are intentionally different; every inference-relevant
    # source, config, model hash/revision, package and runtime setting must match.
    assert {k:v for k,v in old.items() if k not in ('videos','fingerprint')}=={
        k:v for k,v in manifest.items() if k not in ('videos','fingerprint')},'Inference environment changed; cannot reuse'
    with zipfile.ZipFile(PREVIOUS/'executed_source.zip') as archive:
        for name in archive.namelist():
            if name.startswith('evaluation_tools/'):
                assert archive.read(name)==(ROOT/name).read_bytes(),name
    old_selection={r['video_id']:r for r in read(PREVIOUS/'selected_videos.json') if r['category']!='Normal'}
    assert len(old_selection)==60 and set(old_selection)<={r['video_id'] for r in rows}
    origins={}
    for row in rows:
        name=row['video_id']
        if name not in old_selection:continue
        assert row==old_selection[name],name
        source=PREVIOUS/'results'/f'{name}.json';result=read(source)
        assert result['experiment_fingerprint']==old['fingerprint'],name
        assert all(p in result['paired_results'] for p in ('original_off','relaxed_node','force_best'))
        origin=dict(kind='verified_import',source_result=str(source),source_sha256=sha(source),
            source_experiment_fingerprint=old['fingerprint'],
            verification='Identical inference manifest except video set, identical source-video metadata/hash and evaluation source.')
        origins[name]=origin
        target=HERE/'results'/f'{name}.json'
        # Preserve original provenance while associating the verified portable
        # result with the expanded manifest. Never modify the source experiment.
        result['result_origin']=origin;result['experiment_fingerprint']=manifest['fingerprint']
        if target.exists():assert read(target)==result,name
        else:write(target,result)
    write(HERE/'reuse_provenance.json',dict(imported=60,new_required=80,
        original_manifest_sha256=sha(PREVIOUS/'results/metadata/manifest.json'),origins=origins))
    command=read(PREVIOUS/'COMMAND.json')
    command[0]=sys.executable
    command[command.index('--dataset_dir')+1]=args.dataset_dir
    command[command.index('--output_dir')+1]=str(HERE/'results')
    command[command.index('--max_per_category')+1]='100000'
    command+=['--categories',','.join(UCF_CRIME_CATEGORIES)]
    write(HERE/'COMMAND.json',command)
    protocol=read(PREVIOUS/'protocol.json')
    protocol.update(selection='All 140 abnormal test videos across 13 categories. Normal videos excluded before inference.',
        category_counts=dict(Counter(r['category'] for r in rows)),
        reuse='60 verified prior four-arm abnormal-video results + 80 new live four-arm videos; source files and inference manifests matched. Full mAP reranked/recomputed jointly on all 140.',
        evaluation='VadCLIP noninterpolated class-aware AP; 13 abnormal classes, native FPS, integer half-open frames, IoU 0.1:0.1:0.5. No normal-video evaluation.',
        costs='Result-level aggregate includes historical imported usage; current execution API usage is separately saved in results/metadata/token_usage*.json. Cache-affected branch times are not standalone comparisons.',
        limitations='All abnormal test videos but no normal videos, so no normal-video FPR estimate. Prior test-set use means this is not an independent holdout. CBT priors unchanged; historical provenance unverified. GT only used for evaluation.')
    write(HERE/'protocol.json',protocol)
    print(json.dumps(dict(total=140,reused=60,new=80,normal=0,categories=protocol['category_counts'],
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
        if process.returncode:raise RuntimeError('Incomplete run; inspect console.log before resuming')


if __name__=='__main__':main()
