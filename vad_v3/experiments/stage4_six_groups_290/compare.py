"""六组联合评估：140 异常历史结果 + 150 正常新结果，统一 NumPy 排序。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ABNORMAL=HERE.parent/'stage4_six_groups_140'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vad_v3/video_anomaly_detection'))
from run_normal import read,write,sha
from dataset_index import parquet_directory
from evaluate import (load_ground_truth,compute_frame_auc,compute_vadclip_temporal_map,
                      compute_video_level,compute_category_accuracy)
from evaluation_tools.video_fps import source_video_metadata

GROUPS=('off_fixed','strict_fixed','relaxed_node_fixed','force_best_fixed',
        'force_best_no_tail','stage4_only')
LABELS=dict(off_fixed='G1 · off → BUE→CBT',strict_fixed='G2 · strict → BUE→CBT',
    relaxed_node_fixed='G3 · relaxed_node → BUE→CBT',force_best_fixed='G4 · force_best → BUE→CBT',
    force_best_no_tail='G5 · force_best，无后处理',stage4_only='G6 · Stage 4 直接输出')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    manifest=read(HERE/'manifest.json');selected_normal=read(HERE/'selected_normal_videos.json')
    normal_dir=HERE/('normal_resume_results' if (HERE/'resume_provenance.json').exists() else 'normal_results')
    missing=[x['video_id'] for x in selected_normal if not (normal_dir/(x['video_id']+'.json')).exists()]
    if missing:
        if not args.partial:raise RuntimeError(f'Incomplete normals: {150-len(missing)}/150')
        print(json.dumps(dict(completed_normal=150-len(missing),expected_normal=150,next=missing[:3])));return
    run_status=read(normal_dir/'metadata/run_status.json')
    assert run_status['failed']==0 and run_status['total']==150
    selected_abnormal=read(ABNORMAL/'selected_videos.json')
    selected=selected_abnormal+selected_normal
    assert len(selected)==290 and len({x['video_id'] for x in selected})==290
    normal_ids={x['video_id'] for x in selected_normal};abnormal_ids={x['video_id'] for x in selected_abnormal}
    label_dir=parquet_directory('F:/dataset/ucf_crime')
    gt=load_ground_truth(str(label_dir))
    assert set(gt)==normal_ids|abnormal_ids
    assert all(gt[n]['category']=='Normal' and not gt[n]['segments'] for n in normal_ids)
    assert all(gt[n]['category']!='Normal' and gt[n]['segments'] for n in abnormal_ids)
    # 每次评估都读取源视频，不从旧 FPS map 或结果缓存读取时间基准。
    metadata=source_video_metadata({x['video_id']:x['path'] for x in selected},video_root='F:/dataset/ucf_crime')
    fps={n:x['fps'] for n,x in metadata.items()}
    predictions={g:{} for g in GROUPS};normal_source_hashes={};early=0
    inference_manifest=read(normal_dir/'metadata/manifest.json')
    for item in selected_abnormal:
        n=item['video_id'];path=ABNORMAL/'results'/(n+'.json')
        assert sha(path)==manifest['abnormal_results_sha256'][n],n
        result=read(path);assert result['baseline_verified']
        for g in GROUPS:predictions[g][n]=result['arms'][g]
    for item in selected_normal:
        n=item['video_id'];path=normal_dir/(n+'.json');result=read(path)
        assert result['experiment_fingerprint']==inference_manifest['fingerprint'],n
        normal_source_hashes[n]=sha(path)
        paired=result['paired_results']
        arms=dict(off_fixed=paired['original_off'],strict_fixed=result,
                  relaxed_node_fixed=paired['relaxed_node'],force_best_fixed=paired['force_best'])
        if (result.get('mcts_trace') or {}).get('route')=='stage4_direct':
            arms.update({g:paired[g] for g in GROUPS[4:]})
            stages=paired['force_best']['mcts_trace']['stages']
            assert arms['stage4_only']['anomalies']==stages['post_stage4'],n
            assert arms['force_best_no_tail']['anomalies']==stages['post_mcts'],n
        else:
            early+=1
            assert all(not row['anomalies'] and row['frame_scores']==result['frame_scores'] for row in arms.values()),n
            arms.update({g:result for g in GROUPS[4:]})
        identities=[(a['anomaly_type'],a['confidence']) for a in arms['off_fixed']['anomalies']]
        assert all([(a['anomaly_type'],a['confidence']) for a in row['anomalies']]==identities
                   for row in arms.values()),n
        for g,row in arms.items():
            predictions[g][n]={k:row[k] for k in ('video_path','duration','is_normal','anomalies','frame_scores','timestamps')}
    for g,pred in predictions.items():
        assert set(pred)==set(gt)
        for n,row in pred.items():
            row['source_frame_count']=metadata[n]['frame_count']
            assert row['frame_scores'] and len(row['frame_scores'])==len(row['timestamps'])
            assert np.isfinite(row['frame_scores']).all()
    metrics={};summary={};old=read(ABNORMAL/'six_group_comparison.json')
    for g,pred in predictions.items():
        frame=compute_frame_auc(gt,pred,fps)
        temporal=compute_vadclip_temporal_map(gt,pred,fps,include_normal_videos=True)
        agt={n:gt[n] for n in sorted(abnormal_ids)};apred={n:pred[n] for n in sorted(abnormal_ids)}
        abnormal_temporal=compute_vadclip_temporal_map(agt,apred,fps)
        video=compute_video_level(gt,pred);category=compute_category_accuracy(gt,pred)
        for row in category['per_category'].values():row['accuracy']=row['correct']/row['total']
        tp=sum(bool(pred[n]['anomalies']) for n in abnormal_ids);fn=140-tp
        fp=sum(bool(pred[n]['anomalies']) for n in normal_ids);tn=150-fp
        video.update(tp=tp,fn=fn,fp=fp,tn=tn,normal_false_positive_rate=fp/150)
        false_detections=sum(len(pred[n]['anomalies']) for n in normal_ids)
        normal=dict(videos=150,false_positive_videos=fp,false_positive_rate=fp/150,
            false_positive_detections=false_detections,
            false_positive_video_ids=sorted(n for n in normal_ids if pred[n]['anomalies']),
            false_positive_detections_by_category=dict(Counter(a['anomaly_type'] for n in normal_ids for a in pred[n]['anomalies'])))
        metrics[g]=dict(level1=frame,level2=temporal,level2_abnormal_only=abnormal_temporal,
                        level3=video,level4=category,normal=normal)
        summary[g]=dict(label=LABELS[g],global_auc_roc=frame['auc'],aupr=frame['aupr'],
            map_avg=temporal['mAP'],**temporal['values'],
            abnormal_only_map_avg=abnormal_temporal['mAP'],
            old_140_secondary_tie_sort_map_avg=old['summary'][g]['map_avg'],
            added_normal_delta_pp=100*(temporal['mAP']-abnormal_temporal['mAP']),
            video_accuracy=video['accuracy'],video_precision=video['precision'],video_recall=video['recall'],
            video_f1=video['f1'],video_auc=video['auc'],category_accuracy=category['accuracy'],
            normal_false_positive_rate=fp/150,normal_false_positive_videos=fp,
            normal_false_positive_detections=false_detections,
            total_predictions=sum(len(row['anomalies']) for row in pred.values()))
        # 只有排序变化；140 个异常视频的曲线和 frame-level 指标应完全保持不变。
        aframe=compute_frame_auc(agt,apred,fps)
        assert all(abs(aframe[k]-old['metrics'][g]['level1'][k])<1e-12 for k in ('auc','aupr'))
    assert all(metrics[g]['level3']==metrics[GROUPS[0]]['level3'] for g in GROUPS)
    assert all(metrics[g]['level4']==metrics[GROUPS[0]]['level4'] for g in GROUPS)
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),completed=290,abnormal_videos=140,normal_videos=150,
        groups=list(GROUPS),summary=summary,metrics=metrics,protocol=read(HERE/'protocol.json'),
        source_video_metadata=metadata,numpy_version=np.__version__,
        validation=dict(abnormal_results_reused_unchanged=140,normal_videos_new=150,
            normal_early_exit_videos=early,normal_six_branch_videos=150-early,failed=0,
            video_concurrency=inference_manifest['runtime_settings']['video_concurrency'],
            abnormal_frame_metrics_unchanged=True,normal_result_sha256=normal_source_hashes),
        normal_inference_fingerprint=inference_manifest['fingerprint'],manifest_fingerprint=manifest['fingerprint'],
        evaluation_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (
            ROOT/'evaluation_tools/temporal_ap.py',ROOT/'evaluation_tools/video_fps.py',
            ROOT/'vad_v3/video_anomaly_detection/evaluate.py',Path(__file__))},
        gt_sha256={p.name:sha(p) for p in label_dir.glob('test-*.parquet')})
    if (HERE/'resume_provenance.json').exists():
        report['resume_provenance']=read(HERE/'resume_provenance.json')
        report['protocol']['memory_optimization']='After 140 normal videos, unfinished videos resumed with VideoMAE streaming inputs. Same windows/preprocessing/model; actual CUDA reference scores verified exactly. Original completed results preserved with provenance.'
    write(HERE/'six_group_comparison.json',report)
    def pct(v):return f'{100*v:.3f}%'
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
    main_table=table(['组别','Global AUC-ROC','AUPR','AVG mAP（290 视频）'],[
        [LABELS[g],*[pct(summary[g][k]) for k in ('global_auc_roc','aupr','map_avg')]] for g in GROUPS])
    scope_table=table(['组别','原 140 · 旧并列规则','140 · NumPy 排序','290 · NumPy 排序','加入正常视频的变化 / pp'],[
        [LABELS[g],*[pct(summary[g][k]) for k in ('old_140_secondary_tie_sort_map_avg','abnormal_only_map_avg','map_avg')],
         f"{summary[g]['added_normal_delta_pp']:+.3f}"] for g in GROUPS])
    map_table=table(['组别','mAP@0.1','mAP@0.2','mAP@0.3','mAP@0.4','mAP@0.5','AVG'],[
        [LABELS[g],*[pct(summary[g][k]) for k in ('mAP@0.1','mAP@0.2','mAP@0.3','mAP@0.4','mAP@0.5','AVG')]] for g in GROUPS])
    video_table=table(['组别','Accuracy','Precision','Recall','F1','视频 AUC','正常视频误报率','FP 视频数 / 区间数'],[
        [LABELS[g],*[pct(summary[g][k]) for k in ('video_accuracy','video_precision','video_recall','video_f1','video_auc','normal_false_positive_rate')],
         f"{summary[g]['normal_false_positive_videos']} / {summary[g]['normal_false_positive_detections']}"] for g in GROUPS])
    category_table=table(['类别',*[LABELS[g] for g in GROUPS]],[
        [c,*[pct(metrics[g]['level4']['per_category'][c]['accuracy']) for g in GROUPS]]
        for c in sorted(metrics[GROUPS[0]]['level4']['per_category'])])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>290 视频 · 六组对照</title>
<style>body{{font:15px system-ui;max-width:1500px;margin:36px auto;padding:0 24px;color:#25382f;line-height:1.65}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{border:1px solid #d4ddd6;padding:9px;text-align:left}}th{{background:#edf3ef}}.note{{background:#f3f6f4;padding:16px}}a{{color:#216749}}</style>
<h1>140 异常 + 150 正常视频 · 六组 Level 1—4 对照</h1>
<p class="note">正常视频重新推理，视频并发度 2、API 并发度 2；140 个异常视频复用已核验的六组原始结果。AP 统一使用 np.argsort(-confidence)，没有视频 ID 或边界的额外并列排序键。原 140 视频正式报告保持不变。</p>
<h2>汇总</h2>{main_table}
<p>共同前序为 Stage 1—4（含 VideoMAE、CLIP 引导和 VLM 筛选）。G1—G4 保留固定 BUE→CBT；G5/G6 无后续边界处理，均跳过 Stage 5。Level 1 使用 290 个视频的所有帧；AUPR 是 PR 曲线积分。</p>
<h2>排序变化与新增正常视频分别带来的影响</h2>{scope_table}
<p>三个列使用相同的 140 个异常视频预测。第二到第三列仅新增正常视频预测作为误报；并列分数在数组长度变化后也可能被 NumPy 重新排列。</p>
<h2>Level 2 · 全 290 视频的定位 mAP</h2>{map_table}
<p>按 13 个异常类别分别进行同视频 GT 匹配；正常视频不生成异常 GT，其预测异常区间全部作为对应类别 FP 参与全局排序。AP 为非插值 rank AP，使用原视频实测 FPS、整数半开帧区间。AVG 为五个 IoU 阈值的均值；没有添加 Normal AP，因此本表不是官方默认 14 类平均。</p>
<h2>Level 3 · 视频级检测及正常视频误报</h2>{video_table}
<p>各分支只改变边界、不增删候选或修改类别/置信度，因此视频级指标一致。此处视频 AUC 有异常/正常两类样本支持。</p>
<h2>Level 4 · 异常类别准确率</h2>
<p>六组总体类别准确率均为 {pct(summary[GROUPS[0]]['category_accuracy'])}；仅在 140 个异常视频上计算，视频中有任一预测类别正确即记为正确，无检测计错。</p>{category_table}
<h2>可复现性与范围</h2>
<p>150 个正常视频中，{early} 个提前退出，{150-early} 个进入六组分支；缺失、失败结果不会被当作正确正常结果。全量逐类 AP、逐视频 AUC、误报视频清单、源视频时间基准及文件哈希均在 JSON 中。旧异常结果与新正常结果来自不同执行时间，继承既有测试集调参和 CBT 先验限制。</p>
<p>长视频补测期间，将 VideoMAE 一次性驻留全部窗口的方式改为逐窗处理以降低显存占用。实际 CUDA 对照的窗口中心与分数完全一致；已完成的 140 个正常视频保留，剩余 10 个重试，来源与源码变更均独立记录。</p>
<p><a href="six_group_comparison.json">完整对照 JSON</a> · <a href="protocol.json">协议</a> · <a href="{normal_dir.name}/metadata/run_status.json">150 视频完成状态</a> · <a href="streaming_validation.json">逐窗处理核验</a></p></html>'''
    (HERE/'six_group_comparison.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(summary=summary,normal_early_exit=early),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
