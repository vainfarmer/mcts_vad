"""Evaluate four live paired arms; incomplete runs never produce a final report."""
import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vad_v3/video_anomaly_detection'))
from dataset_index import parquet_directory
from evaluate import load_ground_truth,compute_vadclip_temporal_map,compute_frame_auc,compute_video_level
from evaluation_tools.temporal_ap import interval_iou
from evaluation_tools.video_fps import source_video_metadata

POLICIES=('off','strict','relaxed_node','force_best')
LABELS=dict(off='关闭 MCTS',strict='严格接收',relaxed_node='放宽接收',force_best='强制接收最佳提案')


def geometry(rows):return [(r['t_start'],r['t_end']) for r in rows]


def best_iou(segments,rows):
    return statistics.mean(max((interval_iou(gt,(r['t_start'],r['t_end'])) for r in rows),default=0.)
                           for gt in segments) if segments else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true')
    parser.add_argument('--dataset-dir',default='F:/dataset/ucf_crime');args=parser.parse_args()
    selected=json.loads((HERE/'selected_videos.json').read_text(encoding='utf-8'))
    outputs={};missing=[];status={p:Counter() for p in POLICIES};reasons={p:Counter() for p in POLICIES}
    cost=Counter();api={p:Counter() for p in POLICIES};paired_seconds=0.;early=Counter()
    for video in selected:
        name=video['video_id'];path=HERE/'results'/f'{name}.json'
        if not path.exists():missing.append(name);continue
        raw=json.loads(path.read_text(encoding='utf-8'))
        arms=dict(off=raw['paired_results']['original_off'],strict=raw,
            relaxed_node=raw['paired_results']['relaxed_node'],force_best=raw['paired_results']['force_best'])
        outputs[name]=arms;base=arms['off']['anomalies'];source_trace=raw.get('mcts_trace') or {}
        paired_seconds+=raw['processing_time']
        cost.update({k:raw.get('token_usage',{}).get(k,0) for k in (
            'requests','successful_requests','failed_requests','prompt_tokens','completion_tokens','total_tokens','unknown_usage_requests')})
        if source_trace.get('route')!='stage4_direct':early[source_trace.get('skipped_reason','shared_early_exit')]+=1
        for policy,result in arms.items():
            rows=result['anomalies'];trace=result.get('mcts_trace') or {};candidates=trace.get('candidates',{})
            assert len(rows)==len(base),(name,policy,'candidate count')
            assert [(r['confidence'],r['anomaly_type']) for r in rows]==[(r['confidence'],r['anomaly_type']) for r in base]
            if trace.get('route')=='stage4_direct':
                assert trace['stages']['post_stage4']==source_trace['stages']['post_stage4'],(name,policy)
                assert 'post_clip' not in trace['stages']
                if policy=='off':assert trace['mode']=='off' and not candidates and trace['logical_requests']==0
                else:
                    assert trace['acceptance_policy']==policy
                    assert trace['stages']['pre_bue']==trace['stages']['post_mcts']
                assert [r['coarse_score'] for r in trace['stages']['post_stage4']]==[r['confidence'] for r in rows]
            accepted=sum(c.get('applied',False) for c in candidates.values())
            overwritten=sum(c.get('applied',False) and geometry([rows[int(cid)]])==geometry([base[int(cid)]])
                            for cid,c in candidates.items())
            requests=sum(c.get('logical_requests',0) for c in candidates.values())
            status[policy].update(searched_candidates=sum(c.get('logical_requests',0)>0 for c in candidates.values()),
                searched_videos=int(requests>0),accepted_candidates=accepted,logical_search_requests=requests,
                overwritten_candidates=overwritten,final_changed_videos=int(geometry(rows)!=geometry(base)),
                total_predictions=len(rows),search_seconds=trace.get('search_seconds',0.),tail_seconds=trace.get('tail_seconds',0.))
            reasons[policy].update(c.get('reason','unknown') for c in candidates.values())
            api[policy].update(trace.get('search_api_metrics',{}))
            if video['category']=='Normal':
                status[policy].update(normal_false_positive_segments=len(rows),normal_false_positive_videos=int(bool(rows)))
    progress=dict(completed=len(outputs),expected=len(selected),remaining_count=len(missing),next_videos=missing[:3],
        policies={p:dict(status[p],reasons=dict(reasons[p])) for p in POLICIES},shared_early_exits=dict(early),
        paired_api_usage=dict(cost),paired_processing_seconds=paired_seconds)
    if missing:
        if not args.partial:raise RuntimeError('Incomplete run: '+str(progress))
        print(json.dumps(progress,ensure_ascii=False,indent=2));return
    full_gt=load_ground_truth(str(parquet_directory(args.dataset_dir)))
    gt={v['video_id']:full_gt[v['video_id']] for v in selected}
    metadata=source_video_metadata({v['video_id']:v['path'] for v in selected},video_root=args.dataset_dir)
    fps={name:row['fps'] for name,row in metadata.items()}
    pred={p:{name:arms[p] for name,arms in outputs.items()} for p in POLICIES}
    metrics={p:compute_vadclip_temporal_map(gt,pred[p],fps) for p in POLICIES}
    frames={p:compute_frame_auc(gt,pred[p],fps) for p in POLICIES}
    videos={p:compute_video_level(gt,pred[p]) for p in POLICIES}
    rows=[];rawpred={p:{} for p in POLICIES}
    for video in selected:
        name=video['video_id'];segments=[(s/fps[name],e/fps[name]) for s,e in gt[name]['segments']]
        row=dict(video=name,category=video['category'],gt_seconds=segments,arms={})
        base=outputs[name]['off']['anomalies'];base_iou=best_iou(segments,base)
        for policy,result in outputs[name].items():
            trace=result.get('mcts_trace') or {};candidates=trace.get('candidates',{})
            final=result['anomalies'];tiou=best_iou(segments,final)
            delta=None if tiou is None else tiou-base_iou
            if delta is not None:status[policy].update({('improved' if delta>1e-9 else 'worsened' if delta< -1e-9 else 'unchanged'):1})
            row['arms'][policy]=dict(mean_best_tiou=tiou,delta_tiou=delta,final_intervals=geometry(final),
                accepted=sum(c.get('applied',False) for c in candidates.values()),
                reasons=dict(Counter(c.get('reason','unknown') for c in candidates.values())))
            stages=trace.get('stages',{})
            rawpred[policy][name]=dict(anomalies=stages.get('post_mcts',stages.get('post_stage4',final)))
        rows.append(row)
    pre_metrics={p:compute_vadclip_temporal_map(gt,rawpred[p],fps) for p in POLICIES}
    previous=json.loads((HERE.parent/'stage4_fixed_vs_mcts/selected_videos.json').read_text(encoding='utf-8'))
    previous_ids={v['video_id'] for v in previous}
    assert previous_ids <= set(outputs)
    subgroup_metrics={}
    for label,ids in (('previous_14',previous_ids),('additional_51',set(outputs)-previous_ids)):
        subgroup_metrics[label]=dict(video_count=len(ids),metrics={p:compute_vadclip_temporal_map(
            {k:gt[k] for k in ids},{k:pred[p][k] for k in ids},{k:fps[k] for k in ids}) for p in POLICIES})
    summary={p:dict(status[p],reasons=dict(reasons[p]),search_api_metrics=dict(api[p]),
        mean_best_tiou=statistics.mean(r['arms'][p]['mean_best_tiou'] for r in rows if r['category']!='Normal')) for p in POLICIES}
    differences=[r['video'] for r in rows if r['arms']['relaxed_node']['final_intervals']!=r['arms']['force_best']['final_intervals']]
    report=dict(completed=len(outputs),expected=len(selected),summary=summary,metrics=metrics,
        metrics_before_fixed_tail=pre_metrics,frame_metrics=frames,video_metrics=videos,
        relaxed_force_different_videos=differences,per_video=rows,shared_early_exits=dict(early),
        subgroup_metrics=subgroup_metrics,
        paired_api_usage=dict(cost),paired_processing_seconds=paired_seconds,
        protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8')))
    (HERE/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    csv_rows=[dict(video=r['video'],category=r['category'],policy=p,gt_seconds=json.dumps(r['gt_seconds']),
        **{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r['arms'][p].items()}) for r in rows for p in POLICIES]
    with (HERE/'per_video.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    def pct(value):return f'{100*value:.2f}%'
    def score(value):return '—' if value is None else f'{value:.3f}'
    metric_rows=''.join(f'<tr><td>{LABELS[p]}<small>{p}</small></td>'+''.join(
        f'<td>{pct(metrics[p]["values"][k])}</td>' for k in ('mAP@0.1','mAP@0.2','mAP@0.3','mAP@0.4','mAP@0.5','AVG'))
        +f'<td>{summary[p]["accepted_candidates"]}/{summary[p]["searched_candidates"]}</td>'
        +f'<td>{summary[p]["final_changed_videos"]}</td><td>{summary[p].get("normal_false_positive_videos",0)}/5</td></tr>' for p in POLICIES)
    pre_rows=''.join(f'<tr><td>{p}</td><td>{pct(pre_metrics[p]["values"]["AVG"])}</td><td>{pct(metrics[p]["values"]["AVG"])}</td>'
        +f'<td>{pct(frames[p]["auc"])}</td><td>{pct(frames[p]["aupr"])}</td></tr>' for p in POLICIES)
    class_rows=''.join(f'<tr><td>{c}</td>'+''.join(f'<td>{pct(statistics.mean(metrics[p]["per_class"][c][f"AP@{t:.1f}"] for t in (.1,.2,.3,.4,.5)))}</td>'
        for p in POLICIES)+'</tr>' for c in metrics['off']['per_class'])
    subgroup_rows=''.join(f'<tr><td>{label}（{group["video_count"]}）</td>'+''.join(
        f'<td>{pct(group["metrics"][p]["values"]["AVG"])}</td>' for p in POLICIES)+'</tr>' for label,group in subgroup_metrics.items())
    delta=(metrics['relaxed_node']['values']['AVG']-metrics['off']['values']['AVG'])*100
    extra=subgroup_metrics['additional_51']['metrics']
    extra_delta=(extra['relaxed_node']['values']['AVG']-extra['off']['values']['AVG'])*100
    takeaway=(f'放宽接收相对关闭 MCTS 的 AVG 变化为 {delta:+.2f} 个百分点；'
        f'新增 51 个视频子集的变化为 {extra_delta:+.2f} 个百分点。'
        f'严格策略接受 {summary["strict"]["accepted_candidates"]} 个提案；'
        f'放宽与强制策略的最终区间在 {len(differences)} 个视频上不同。')
    video_rows=''.join(f'<tr><td>{r["category"]}<small>{r["video"]}</small></td><td>{html.escape(str(r["gt_seconds"]))}</td>'+''.join(
        f'<td>{score(r["arms"][p]["mean_best_tiou"])}<small>{html.escape(str(r["arms"][p]["final_intervals"]))}</small></td>' for p in POLICIES)+'</tr>' for r in rows)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>65 视频：MCTS 四组配对实验</title>
<style>body{{font:15px system-ui;max-width:1600px;margin:40px auto;padding:0 24px;color:#25382f}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{border:1px solid #d4ddd6;padding:9px;text-align:left}}th{{background:#edf3ef}}small{{display:block;color:#65726a}}pre{{white-space:pre-wrap}}a{{color:#216749}}</style>
<h1>65 视频：Stage 4 →（可选 MCTS）→ 固定 BUE→CBT</h1>
<p><strong>{html.escape(takeaway)}</strong></p>
<p>60 个异常视频、5 个正常视频，全部四组成功后生成此报告。上游只运行一次，原始候选、类别和置信度共享；off 在任何搜索前直接完成固定后处理。strict、relaxed_node、force_best 分别实时执行搜索与接收逻辑，并使用独立视频状态和逻辑预算。仅完全相同的模型参数、提示词及图像请求复用回复。</p>
<p>不是上一轮冻结搜索树回放。所有分支均跳过 Stage 5、中间精修、top-k、NMS、类别复核、置信度过滤和 QSCORE；后处理均为原有固定 BUE→CBT。</p>
<h2>最终定位指标</h2><p>当前 evaluate.py：VadCLIP 非插值 AP，13 类异常宏平均，原视频 FPS，整数半开帧区间；AVG 为 IoU 0.1—0.5 的均值。正常视频不进入主 mAP，误报单独列出。</p>
<table><tr><th>策略</th><th>@0.1</th><th>@0.2</th><th>@0.3</th><th>@0.4</th><th>@0.5</th><th>AVG</th><th>接受/搜索候选</th><th>相对 off 改变视频</th><th>正常视频误报</th></tr>{metric_rows}</table>
<h2>后处理与帧指标</h2><p>帧 AUC/AUPR 包含全部 65 个视频；后处理前后 mAP 仅比较对应阶段区间。</p>
<table><tr><th>策略</th><th>后处理前 AVG</th><th>最终 AVG</th><th>帧 AUC</th><th>帧 AUPR</th></tr>{pre_rows}</table>
<h2>各类别 AP（五阈值平均）</h2><table><tr><th>类别</th><th>off</th><th>strict</th><th>relaxed_node</th><th>force_best</th></tr>{class_rows}</table>
<h2>相对上一轮的样本扩展</h2><p>previous_14 为之前试验的同名视频本次重新推理结果；additional_51 为本轮增加的 51 个视频。分组只用于诊断，不声称独立留出验证，也不能对两组 mAP 简单加权得到整体 mAP。</p>
<table><tr><th>子集 / 视频数</th><th>off AVG</th><th>strict AVG</th><th>relaxed_node AVG</th><th>force_best AVG</th></tr>{subgroup_rows}</table>
<h2>逐视频定位</h2><p>表中数值为各 GT 最佳 tIoU 的均值，不计类别，作为边界诊断；下方为最终区间（秒）。</p>
<table><tr><th>视频</th><th>GT 秒</th><th>off</th><th>strict</th><th>relaxed_node</th><th>force_best</th></tr>{video_rows}</table>
<h2>配置、成本与范围</h2><p>动作 shrink；每候选 8 次模拟、最大深度 4；每视频最多搜索 2 个候选；每候选/每视频逻辑请求上限 10/20。strict 保留完整性、CLIP 质量及最终 A/B 比较；relaxed_node 保留目标、同一事件与 CLIP 质量；force_best 接受最高奖励有效非根节点。失败评价与非法区间始终不采用。</p>
<p>Token 与总耗时属于共享四组执行的实际用量；后执行分支会命中缓存，不能当作独立部署耗时/费用的公平比较。65 视频是子集，并包含先前试验视频，不是独立留出验证集。原有 CBT 时长先验未调整，历史统计来源未独立核实；GT 只用于评价。默认接收策略未改。</p>
<pre>{html.escape(json.dumps(dict(summary=summary,shared_early_exits=dict(early),paired_api_usage=dict(cost),paired_processing_seconds=paired_seconds,relaxed_force_different_videos=differences),ensure_ascii=False,indent=2))}</pre>
<p><a href="comparison.json">完整 JSON</a> · <a href="per_video.csv">逐视频 CSV</a> · <a href="protocol.json">协议</a> · <a href="selected_videos.json">输入清单及哈希</a> · <a href="COMMAND.json">复现命令</a> · <a href="executed_source.zip">执行源码</a></p></html>'''
    (HERE/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(completed=len(outputs),summary=summary,metrics={p:metrics[p]['percent'] for p in POLICIES},
        paired_api_usage=dict(cost),relaxed_force_different_videos=differences),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
