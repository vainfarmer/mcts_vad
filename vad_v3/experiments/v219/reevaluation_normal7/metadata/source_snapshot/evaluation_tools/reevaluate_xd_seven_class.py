"""Offline seven-class reevaluation of frozen v219 paired detections.

Normal is derived ONLY from the existing predicted is_normal decision. This is
an explicit whole-video Normal detector, not VadCLIP's thresholded Normal-curve
proposal method. No label, filename suffix or GT is used to create a prediction.
The original six-class detections and all inference artifacts stay read-only.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT/'vad_v3/video_anomaly_detection'
if str(SOURCE) not in sys.path:
    sys.path.insert(0,str(SOURCE))
from xd_config import XD_CLASSES
from xd_dataset import load_xd_index, annotation_fingerprint
from xd_evaluate import temporal_metrics

ARM_NAMES=('baseline_off','mcts','mcts_qscore','mcts_qscore_cbt',
           'mcts_qscore_cbt_bue','mcts_cbt_bue')
POLICY={
    'name':'existing_video_normal_decision_v1',
    'normal_prediction_gate':'saved is_normal == true and no anomalous detections',
    'normal_prediction_interval':'[0, source_frame_count / native_fps)',
    'normal_prediction_confidence':'1 - mean(interpolated anomaly score over ALL source frames)',
    'normal_ground_truth':'[0, source_frame_count) on GT-Normal videos only',
    'abnormal_video_background_is_normal_gt':False,
    'score_calibrated':False,
    'gt_used_for_prediction':False,
    'vadclip_prediction_decoder_reproduced':False,
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def add_normal_prediction(arm,timestamps,fps,frame_count):
    """Pure prediction transform: deliberately has no GT/video-name argument."""
    if type(arm.get('is_normal')) is not bool:
        raise ValueError('Missing/invalid saved Normal decision')
    if arm['is_normal'] != (not arm['anomalies']):
        raise ValueError('Normal decision conflicts with saved anomalous detections')
    if any(e['anomaly_type'] not in XD_CLASSES for e in arm['anomalies']):
        raise ValueError('Input must contain only the six original anomaly classes')
    if not np.isfinite(fps) or fps<=0 or type(frame_count) is not int or frame_count<=0:
        raise ValueError('Invalid source FPS/frame count')
    times=np.asarray(timestamps,dtype=float)
    scores=np.asarray(arm['frame_scores'],dtype=float)
    if (times.ndim!=1 or len(times)==0 or times.shape!=scores.shape or
            not np.isfinite(times).all() or not np.isfinite(scores).all() or
            np.any(np.diff(times)<=0) or np.any(times<0) or np.any(times>=frame_count/fps) or
            np.any(scores<0) or np.any(scores>1)):
        raise ValueError('Invalid saved prediction time series')
    rows=copy.deepcopy(arm['anomalies'])
    audit=dict(predicted_is_normal=arm['is_normal'],normal_prediction_count=0)
    if arm['is_normal']:
        mean_anomaly=float(np.mean(np.interp(np.arange(frame_count)/fps,times,scores)))
        row=dict(anomaly_type='Normal',t_start=0.,t_end=frame_count/fps,
                 confidence=float(np.clip(1.-mean_anomaly,0.,1.)),
                 description='Whole-video Normal prediction from the saved pipeline decision',
                 prediction_source=POLICY['name'])
        rows.append(row)
        audit.update(normal_prediction_count=1,mean_anomaly_score=mean_anomaly,
                     confidence=row['confidence'],interval=[row['t_start'],row['t_end']])
    return rows,audit


def reevaluate(experiment,output_dir,dataset_root):
    experiment=Path(experiment).resolve();output_dir=Path(output_dir).resolve()
    protected=[experiment,experiment/'result',experiment/'metadata']
    if any(output_dir==p or output_dir.is_relative_to(experiment/'result') or
           output_dir.is_relative_to(experiment/'metadata') for p in protected):
        raise ValueError('Choose a separate reevaluation output directory')
    selection_path=experiment/'metadata/selection.json'
    selection=read(selection_path)
    names=sorted({n for rows in selection['slots'].values() for n in rows})
    inputs={n:experiment/'result'/(n+'.json') for n in names}
    hashes={n:digest(p) for n,p in inputs.items()}
    original_report_path=experiment/'evaluation_report.json'
    original_report=read(original_report_path)
    metadata_path=experiment/'metadata/video_fps.json'
    metadata=read(metadata_path)['videos']
    original_manifest=read(experiment/'metadata/manifest.json')
    manifest=dict(schema='xd-normal7-reevaluation-v1',normal_policy=POLICY,
        input_experiment=str(experiment),input_results_sha256=hashes,
        original_report_sha256=digest(original_report_path),fps_sha256=digest(metadata_path),
        selection_sha256=digest(selection_path),
        source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in
            [Path(__file__),SOURCE/'xd_evaluate.py',SOURCE/'xd_dataset.py',PROJECT/'evaluation_tools/temporal_ap.py']})
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    manifest['fingerprint']=fingerprint
    manifest_path=output_dir/'metadata/manifest.json'
    if manifest_path.exists() and read(manifest_path)['fingerprint']!=fingerprint:
        raise ValueError('Reevaluation manifest mismatch; use a new output directory')

    # Construct every prediction BEFORE loading labels/GT. The chosen policy is
    # frozen and applied to ALL videos, including videos that could be mistaken
    # for Normal. It cannot benefit from ground-truth category membership.
    predictions={arm:{} for arm in ARM_NAMES}
    video_outputs={}
    for name in names:
        src=read(inputs[name]);meta=metadata[name]
        if src.get('dataset')!='xd_violence' or Path(src['video_path']).stem!=name:
            raise ValueError('Input video identity mismatch')
        if src['experiment_fingerprint']!=original_manifest['fingerprint']:
            raise ValueError('Input experiment fingerprint mismatch')
        video=dict(schema='xd-normal7-predictions-v1',dataset='xd_violence',video_path=src['video_path'],
                   duration=meta['frame_count']/meta['fps'],fps=meta['fps'],frame_count=meta['frame_count'],
                   source_result_sha256=hashes[name],reevaluation_fingerprint=fingerprint,arms={})
        for arm in ARM_NAMES:
            source=src['baseline_off'] if arm=='baseline_off' else src['arms'][arm]
            rows,audit=add_normal_prediction(source,src['timestamps'],meta['fps'],meta['frame_count'])
            predictions[arm][name]={'anomalies':rows}
            video['arms'][arm]=dict(detections=rows,normal_prediction_audit=audit)
        video_outputs[name]=video

    all_records=load_xd_index(dataset_root)
    records={n:all_records[n] for n in names}
    actual_fingerprints=annotation_fingerprint(dataset_root)
    if actual_fingerprints!=original_report['protocol']['annotation_sha256']:
        raise ValueError('Annotations differ from the source experiment')
    arms={};audits={}
    for arm in ARM_NAMES:
        metric,audit=temporal_metrics(records,predictions[arm],metadata,True,normal_as_class=True)
        old=original_report['arms'][arm]
        for category in XD_CLASSES:
            if metric['per_class'][category]!=old['temporal_mAP_full']['per_class'][category]:
                raise ValueError('Original six-class metrics changed: '+arm+'/'+category)
        normal_ap_avg=float(np.mean([metric['per_class']['Normal'][f'AP@{t:.1f}'] for t in (.1,.2,.3,.4,.5)]))
        expected=(6*old['temporal_mAP_full']['values']['AVG']+normal_ap_avg)/7
        if abs(metric['values']['AVG']-expected)>1e-12:
            raise AssertionError('Invalid macro average')
        arms[arm]=dict(temporal_mAP_7class=metric,temporal_mAP_6class_reference=old['temporal_mAP_full'],
                      frame_AP_unchanged=old['frame_AP'],frame_AUC_unchanged=old['frame_AUC'],
                      normal_AP_AVG=normal_ap_avg,anomaly_metrics_unchanged=True,
                      total_predictions=sum(len(p['anomalies']) for p in predictions[arm].values()))
        audits[arm]=audit
    report=dict(schema='xd-normal7-reevaluation-v1',dataset='xd_violence',video_count=len(names),
                normal_video_count=sum(r['category']=='Normal' for r in records.values()),
                classes=['Normal',*XD_CLASSES],normal_policy=POLICY,
                protocol=dict(iou_thresholds=[.1,.2,.3,.4,.5],
                    ap='sum(precision_at_rank * TP) / N_GT',
                    interval='native source frames [start,end)',
                    macro_average='all seven classes, including Normal',
                    zero_prediction_class_ap=0.,
                    normal_gt_differs_from_vadclip='Uses full source-frame length, not CLIP feature length * 16',
                    scope='same 35-video v219 subset; not the full XD benchmark',
                    note='Whole-video Normal predictions reuse saved model decisions; not VadCLIP Normal-curve proposal reproduction',
                    annotation_sha256=actual_fingerprints),
                arms=arms,inference_calls=0,new_api_tokens=0,
                reevaluation_fingerprint=fingerprint,created_at=datetime.now(timezone.utc).isoformat())
    # Recheck all immutable inputs after computation and before writing outputs.
    if hashes!={n:digest(p) for n,p in inputs.items()} or digest(original_report_path)!=manifest['original_report_sha256']:
        raise ValueError('Source results changed during reevaluation')
    save(manifest_path,manifest)
    for name,value in video_outputs.items():
        save(output_dir/'result'/(name+'.json'),value)
    for arm,value in audits.items():
        save(output_dir/'metadata'/('matching_'+arm+'.json'),value)
    save(output_dir/'metadata/video_fps.json',dict(schema='video-fps-v1',videos={n:metadata[n] for n in names}))
    save(output_dir/'evaluation_report.json',report)
    # Keep the exact scoring implementation alongside hashes for reproducibility.
    for relative in manifest['source_sha256']:
        dest=output_dir/'metadata/source_snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((PROJECT/relative).read_bytes())
    lines=['# v219：Normal 作为第七个检测类别的离线重评','',
        f"覆盖原批次 {len(names)} 个视频，含 {report['normal_video_count']} 个真实 Normal 视频；未新增视频，未调用模型或 API。",'',
        'Normal 预测来自原结果的 `is_normal=True`，并要求无异常检测，输出整段 `[0, 视频时长)`；置信度为 `1 - 源帧时间轴上的平均异常分数`。所有视频统一应用同一规则，构造预测的函数不接收视频名称、真实类别或 GT。',
        'Normal GT 仅在真实 Normal 视频上标为完整视频；异常视频中的正常背景不额外标为 Normal GT。异常视频若被模型误判成 Normal，其预测会计为 Normal 类 FP。','',
        '| 组别 | 原六类 AVG % | Normal AP AVG % | 七类 AVG % | 七类 mAP@0.5 % |',
        '|---|---:|---:|---:|---:|']
    for arm,row in arms.items():
        lines.append(f"| {arm} | {row['temporal_mAP_6class_reference']['percent']['AVG']:.3f} | {100*row['normal_AP_AVG']:.3f} | {row['temporal_mAP_7class']['percent']['AVG']:.3f} | {row['temporal_mAP_7class']['percent']['mAP@0.5']:.3f} |")
    lines+=['','## 指标解释','',
        '`七类 mAP = (六类 mAP × 6 + Normal AP) / 7`，每个 IoU 阈值分别计算，再对五个阈值取平均。七类指标包含了 Normal 的贡献；数值增加不意味着原六类异常的定位变好。',
        '所有原六类的 AP、TP、FP 和预测数量已逐类校验完全不变。二分类帧级 AP/AUC 也不变。',
        '本重评使用现有模型的视频级 Normal 判定，不是根据真实 Normal 视频名单补预测，也没有新增 Normal 视觉识别模型。','',
        '与 VadCLIP 默认七类均值相近的部分是类别集合、AP 公式和 IoU 阈值；本项目仍采用整视频 Normal 预测和源视频完整帧长，未复现其 Normal 类曲线阈值提案或特征长度×16的时间范围。35 视频子集的七类结果不能直接当作完整 XD 基准成绩。','',
        '## 文件','',
        '- `evaluation_report.json`：七类 mAP、Normal AP、原六类对照与逐类 TP/FP。',
        '- `result/<视频名>.json`：五组及 off 对照的七类检测，字段 `arms.<组名>.detections`。',
        '- `metadata/matching_<组名>.json`：包括 Normal 在内的逐预测 GT 匹配。',
        '- `metadata/manifest.json`：原文件哈希、评分规则及源码哈希；原 v219 结果未覆盖。','',
        '复现命令：','',
        '```powershell',
        "conda run -n common python -m evaluation_tools.reevaluate_xd_seven_class --experiment vad_v3/experiments/v219 --dataset_root F:/dataset/xd-violence",'```']
    (output_dir/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=PROJECT/'vad_v3/experiments/v219')
    parser.add_argument('--dataset_root',type=Path,default=Path('F:/dataset/xd-violence'))
    parser.add_argument('--output_dir',type=Path)
    args=parser.parse_args()
    output=args.output_dir or args.experiment/'reevaluation_normal7'
    report=reevaluate(args.experiment,output,args.dataset_root)
    print(json.dumps({a:dict(normal_AP=100*r['normal_AP_AVG'],
          seven_class_percent=r['temporal_mAP_7class']['percent']) for a,r in report['arms'].items()},indent=2))


if __name__=='__main__':
    main()
