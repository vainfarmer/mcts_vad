"""Compare paired pilot arms using source-video FPS and class-aware temporal AP."""
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
from evaluate import load_ground_truth, compute_vadclip_temporal_map
from evaluation_tools.temporal_ap import compute_vadclip_map, interval_iou
from evaluation_tools.video_fps import source_video_metadata


def geometry(result):
    return [(x['t_start'],x['t_end'],x['anomaly_type'],x['confidence']) for x in result['anomalies']]


def mean_best_iou(segments,result):
    predictions=[(p['t_start'],p['t_end']) for p in result['anomalies']]
    return statistics.mean(max((interval_iou(gt,p) for p in predictions),default=0.) for gt in segments) if segments else None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true')
    parser.add_argument('--dataset-dir',default='F:/dataset/ucf_crime')
    args=parser.parse_args()
    selected=json.loads((HERE/'selected_videos.json').read_text(encoding='utf-8'))
    full_gt=load_ground_truth(str(parquet_directory(args.dataset_dir)))
    rows=[];predictions={'stage4_fixed':{},'stage4_mcts_fixed':{}}
    reasons=Counter();rejection_flags=Counter();invalid=Counter();cost=Counter();gt={};fps={};missing=[]
    evaluation_audit=[]
    metadata=source_video_metadata({v['video_id']:v['path'] for v in selected
        if (HERE/'results'/f"{v['video_id']}.json").exists()},video_root=args.dataset_dir)
    for video in selected:
        name=video['video_id'];path=HERE/'results'/f'{name}.json'
        if not path.exists():missing.append(name);continue
        result=json.loads(path.read_text(encoding='utf-8'))
        off=result['paired_results']['original_off']
        predictions['stage4_fixed'][name]=off
        predictions['stage4_mcts_fixed'][name]=result
        assert len(result['anomalies'])==len(off['anomalies'])
        assert [(x['confidence'],x['anomaly_type']) for x in result['anomalies']]==[(x['confidence'],x['anomaly_type']) for x in off['anomalies']]
        gt[name]=full_gt[name];fps[name]=metadata[name]['fps']
        segments=[(s/fps[name],e/fps[name]) for s,e in gt[name]['segments']]
        candidates=(result.get('mcts_trace') or {}).get('candidates',{})
        requests=sum(c.get('logical_requests',0) for c in candidates.values())
        accepted=sum(c.get('applied',False) for c in candidates.values())
        evaluated=sum(len(c.get('evaluations',[])) for c in candidates.values())
        candidate_reasons=Counter(c.get('reason','unknown') for c in candidates.values())
        reasons.update(candidate_reasons)
        overwritten=0
        for cid,candidate in candidates.items():
            if candidate.get('applied'):
                left=result['anomalies'][int(cid)];right=off['anomalies'][int(cid)]
                overwritten+=all(abs(left[k]-right[k])<1e-8 for k in ('t_start','t_end'))
            for ev in candidate.get('evaluations',[]):
                if ev.get('invalid_reason'):invalid[ev['invalid_reason']]+=1
                answer=ev.get('answer')
                if not ev.get('depth') or not answer:continue
                flags=[]
                if not answer.get('target_present_inside'):flags.append('target_absent')
                if answer.get('event_completeness')!='complete':flags.append('incomplete_event')
                for side in ('start_status','end_status'):
                    if answer.get(side)=='event_cut':flags.append(side+'_cut')
                rejection_flags.update(flags)
                evaluation_audit.append(dict(video=name,candidate_id=cid,interval=ev['interval'],flags=flags,answer=answer))
        before=mean_best_iou(segments,off);after=mean_best_iou(segments,result)
        usage=result.get('token_usage',{})
        cost.update({key:usage.get(key,0) for key in ('requests','successful_requests','failed_requests','prompt_tokens','completion_tokens','total_tokens','unknown_usage_requests')})
        rows.append(dict(category=video['category'],video_id=name,fps=fps[name],duration=metadata[name]['duration'],
            gt_seconds=segments,fixed_predictions=off['anomalies'],mcts_predictions=result['anomalies'],
            baseline_mean_best_tiou=before,mcts_mean_best_tiou=after,delta_tiou=None if before is None else after-before,
            baseline_count=len(off['anomalies']),mcts_count=len(result['anomalies']),
            candidates=len(candidates),searched_candidates=sum(c.get('logical_requests',0)>0 for c in candidates.values()),
            accepted_candidates=accepted,logical_search_requests=requests,evaluated_nodes=evaluated,
            reason_counts=dict(candidate_reasons),changed=geometry(result)!=geometry(off),
            accepted_overwritten_by_fixed_tail=overwritten,
            paired_elapsed_seconds=result.get('processing_time'),
            search_elapsed_seconds=sum(c.get('elapsed_seconds',0) for c in candidates.values()),
            paired_token_usage=usage))
    summary=dict(completed=len(rows),expected=len(selected),remaining=missing,
        searched_videos=sum(r['logical_search_requests']>0 for r in rows),
        searched_candidates=sum(r['searched_candidates'] for r in rows),
        accepted_candidates=sum(r['accepted_candidates'] for r in rows),
        logical_search_requests=sum(r['logical_search_requests'] for r in rows),
        reasons=dict(reasons),proposal_semantic_flags=dict(rejection_flags),invalid_evaluations=dict(invalid),
        final_changed_videos=sum(r['changed'] for r in rows),
        accepted_overwritten_by_fixed_tail=sum(r['accepted_overwritten_by_fixed_tail'] for r in rows),
        paired_api_usage=dict(cost))
    if missing:
        if not args.partial:raise RuntimeError('Missing results: '+', '.join(missing))
        print(json.dumps(summary,ensure_ascii=False,indent=2));return
    metrics={};audits={}
    for arm,preds in predictions.items():
        metrics[arm]=compute_vadclip_temporal_map(gt,preds,fps)
        checked,audits[arm]=compute_vadclip_map(gt,preds,fps)
        assert metrics[arm]['values']==checked['values']
    abnormal=[r for r in rows if r['category']!='Normal']
    summary.update(improved=sum(r['delta_tiou']>1e-9 for r in abnormal),
        worsened=sum(r['delta_tiou']< -1e-9 for r in abnormal),
        unchanged=sum(abs(r['delta_tiou'])<=1e-9 for r in abnormal),
        fixed_mean_best_tiou=statistics.mean(r['baseline_mean_best_tiou'] for r in abnormal),
        mcts_mean_best_tiou=statistics.mean(r['mcts_mean_best_tiou'] for r in abnormal),
        paired_total_seconds=sum(r['paired_elapsed_seconds'] for r in rows),
        search_total_seconds=sum(r['search_elapsed_seconds'] for r in rows))
    summary['normal_false_positives']={arm:sum(len(preds[r['video_id']]['anomalies']) for r in rows if r['category']=='Normal') for arm,preds in predictions.items()}
    protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    report=dict(summary=summary,metrics=metrics,per_video=rows,protocol=protocol)
    (HERE/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'matching_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'rejection_audit.json').write_text(json.dumps(dict(semantic_flags=dict(rejection_flags),invalid=dict(invalid),evaluations=evaluation_audit),ensure_ascii=False,indent=2),encoding='utf-8')
    with (HERE/'per_video.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    def score(value):return '—' if value is None else f'{value:.3f}'
    def intervals(values):return '<br>'.join(html.escape(f"{p['anomaly_type']} [{p['t_start']:.2f}, {p['t_end']:.2f}]") for p in values) or '无检测'
    metric_rows=''.join('<tr><td>'+html.escape(arm)+'</td>'+''.join(f'<td>{metrics[arm]["percent"][key]:.2f}%</td>' for key in ('mAP@0.1','mAP@0.3','mAP@0.5','AVG'))+'</tr>' for arm in metrics)
    video_rows=''.join(f'<tr><td>{r["category"]}<small>{r["video_id"]}</small></td><td>{html.escape(str([(round(s,2),round(e,2)) for s,e in r["gt_seconds"]]))}</td><td>{intervals(r["fixed_predictions"])}</td><td>{intervals(r["mcts_predictions"])}</td><td>{score(r["baseline_mean_best_tiou"])} → {score(r["mcts_mean_best_tiou"])}</td><td>{r["searched_candidates"]} / {r["accepted_candidates"]} / {r["logical_search_requests"]}</td><td>{html.escape(str(r["reason_counts"]))}</td></tr>' for r in rows)
    conclusion=(f'共 {summary["searched_videos"]} 个视频、{summary["searched_candidates"]} 个候选进入搜索，'
        f'{summary["logical_search_requests"]} 次逻辑评估，接受 {summary["accepted_candidates"]} 个修改。'
        +f'最终有 {summary["final_changed_videos"]} 个视频的输出改变；{summary["accepted_overwritten_by_fixed_tail"]} 个已接受修改被固定后处理恢复到对照边界。')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Stage 4：固定后处理与 MCTS 配对比较</title>
<style>body{{font:15px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;color:#26352f}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{border:1px solid #d4ddd6;padding:10px;text-align:left}}th{{background:#edf3ef}}small{{display:block;color:#65726a}}pre{{white-space:pre-wrap}}a{{color:#216749}}</style>
<h1>Stage 4 原始候选 →（可选 MCTS）→ 固定 BUE→CBT</h1>
<p><strong>{html.escape(conclusion)}</strong></p>
<p>13 类异常各一个测试视频，另加一个正常视频；与上一轮使用完全相同的文件。两组共享 Stage 4 的全部原始候选；置信度固定为发现分数，类别固定为上游预测，GT 不参与推理。</p>
<p>stage4_fixed：Stage 4 → BUE→CBT。stage4_mcts_fixed：Stage 4 → 原有左右边界收缩 MCTS → BUE→CBT。两组均跳过 Stage 5、中间精修、top-k、NMS、类别复核、置信度过滤及 QSCORE。BUE/CBT 是共同的固定后处理，不是搜索动作。</p>
<p>原有 CBT 时长先验的历史统计来源未重新核实，本次未使用所选视频标注调参。小样本对照不代表完整测试集表现。</p>
<h2>定位指标</h2><p>使用已修改的 evaluate.py：类别敏感、非插值 AP = sum(Precision@k × TP_k) / N_GT，依据 <a href="https://github.com/nwpu-zxr/VadCLIP/blob/main/src/utils/ucf_detectionMAP.py">VadCLIP 官方计算公式</a>。使用原视频 FPS 和整数半开区间；对 13 类求均值；AVG 为 IoU 0.1—0.5 的均值。正常视频单独检查误报。现有最终区间直接用于评价，不重新生成 VadCLIP 候选。</p>
<table><tr><th>方案</th><th>mAP@0.1</th><th>mAP@0.3</th><th>mAP@0.5</th><th>AVG</th></tr>{metric_rows}</table>
<h2>逐视频结果</h2><p>tIoU 列为各 GT 区间最佳检测重叠的均值，不计类别，漏检记 0；它用于诊断边界，不替代上面的类别敏感 AP。搜索/接受/请求分别为候选数、接受数、逻辑 VLM 评估数。</p>
<table><tr><th>类别 / 视频</th><th>GT 秒</th><th>直接 BUE→CBT</th><th>MCTS 后 BUE→CBT</th><th>最佳 tIoU 均值</th><th>搜索 / 接受 / 请求</th><th>搜索结果</th></tr>{video_rows}</table>
<h2>执行统计</h2><p>耗时和 Token 是整个配对流程的实际用量，不是某一个方案独立运行的费用。搜索耗时和逻辑请求另外记录；缓存命中和 HTTP 重试会使逻辑请求与实际请求数不同。</p><pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre>
<p><a href="comparison.json">完整 JSON</a> · <a href="per_video.csv">逐视频 CSV</a> · <a href="selected_videos.json">输入清单</a> · <a href="COMMAND.json">复现命令</a> · <a href="rejection_audit.json">拒绝原因明细</a> · <a href="executed_source.zip">本轮实际执行源码</a></p></html>'''
    (HERE/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(summary=summary,metrics={k:v['percent'] for k,v in metrics.items()}),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
