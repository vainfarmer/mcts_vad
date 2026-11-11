"""Report matched acceptance ablations, with labels accessed only after replay."""
import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'vad_v3/video_anomaly_detection'))
from dataset_index import parquet_directory
from evaluate import load_ground_truth, compute_vadclip_temporal_map
from evaluation_tools.temporal_ap import interval_iou
from evaluation_tools.video_fps import source_video_metadata

POLICIES = ('fixed', 'strict', 'relaxed_node', 'force_best')
LABELS = dict(fixed='直接 BUE→CBT', strict='严格接收 MCTS',
              relaxed_node='放宽接收 MCTS', force_best='强制接收最佳提案')


def best_iou(segments, rows):
    return statistics.mean(max((interval_iou(gt, (r['t_start'], r['t_end'])) for r in rows), default=0.)
                           for gt in segments) if segments else None


def geometry(rows):
    return [(r['t_start'], r['t_end']) for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--dataset-dir', default='F:/dataset/ucf_crime')
    args = parser.parse_args()
    selected = json.loads((HERE / 'selected_videos.json').read_text(encoding='utf-8'))
    outputs = {v['video_id']: HERE / 'results' / (v['video_id'] + '.json') for v in selected}
    missing = [name for name, path in outputs.items() if not path.exists()]
    if missing:
        if not args.partial:
            raise RuntimeError('Missing: ' + ', '.join(missing))
        print(json.dumps(dict(completed=len(selected)-len(missing), remaining=missing), indent=2)); return
    full_gt = load_ground_truth(str(parquet_directory(args.dataset_dir)))
    gt = {v['video_id']: full_gt[v['video_id']] for v in selected}
    metadata = source_video_metadata({v['video_id']:v['path'] for v in selected},video_root=args.dataset_dir)
    fps = {name:row['fps'] for name,row in metadata.items()}
    pred = {p: {} for p in POLICIES}; before = {p: {} for p in POLICIES}
    rows = []; status = {p: Counter() for p in POLICIES}; elapsed = 0.
    selected_diagnostics = {p: Counter() for p in ('relaxed_node', 'force_best')}
    for video in selected:
        name = video['video_id']; obj = json.loads(outputs[name].read_text(encoding='utf-8'))
        assert obj['baseline_reproduced'] and obj['new_vlm_requests'] == 0
        elapsed += obj['replay_seconds']
        segments = [(s / fps[name], e / fps[name]) for s, e in gt[name]['segments']]
        row = dict(video=name, category=video['category'], gt_seconds=segments, arms={})
        base = obj['arms']['fixed']['anomalies']
        for policy in POLICIES:
            arm = obj['arms'][policy]; final = arm['anomalies']; raw = arm['post_search']
            pred[policy][name] = dict(anomalies=final)
            before[policy][name] = dict(anomalies=raw)
            assert len(final) == len(raw) == len(base)
            assert [(r['confidence'], r['anomaly_type']) for r in final] == [
                (r['confidence'], r['anomaly_type']) for r in base]
            decisions = obj['decisions'].get(policy, {})
            accepted = sum(d['applied'] for d in decisions.values())
            if policy in selected_diagnostics:
                for decision in decisions.values():
                    if decision['applied']:
                        answer = decision['selected_answer']
                        selected_diagnostics[policy].update(
                            partial_selected=int(answer['event_completeness'] == 'partial'),
                            cut_selected=int('event_cut' in (answer['start_status'], answer['end_status'])),
                            below_root_reward=int(decision['selected_reward'] < decision['root_reward']))
                    else:
                        selected_diagnostics[policy].update({decision['source_reason']: 1})
            overwritten = sum(d['applied'] and geometry([final[int(cid)]]) == geometry([base[int(cid)]])
                              for cid, d in decisions.items())
            changed = geometry(final) != geometry(base)
            status[policy].update(accepted_candidates=accepted, overwritten_candidates=overwritten,
                                  final_changed_videos=int(changed), total_predictions=len(final))
            tiou = best_iou(segments, final); baseline = best_iou(segments, base)
            delta = tiou-baseline if tiou is not None else None
            if delta is not None:
                status[policy].update({('improved' if delta > 1e-9 else 'worsened' if delta < -1e-9 else 'unchanged'): 1})
            else:
                status[policy].update(normal_false_positives=len(final))
            row['arms'][policy] = dict(accepted=accepted, overwritten=overwritten, changed=changed,
                mean_best_tiou=tiou, delta_tiou=delta, before_fixed_tail=geometry(raw),
                final_intervals=geometry(final), final_predictions=final)
        rows.append(row)
    metrics = {p: compute_vadclip_temporal_map(gt, pred[p], fps) for p in POLICIES}
    pre_metrics = {p: compute_vadclip_temporal_map(gt, before[p], fps) for p in POLICIES}
    summary = {p: dict(status[p], mean_best_tiou=statistics.mean(
        r['arms'][p]['mean_best_tiou'] for r in rows if r['category'] != 'Normal')) for p in POLICIES}
    class_contributions = {c: sum(metrics['relaxed_node']['per_class'][c][f'AP@{t:.1f}'] -
                                 metrics['fixed']['per_class'][c][f'AP@{t:.1f}']
                                 for t in (.1,.2,.3,.4,.5)) / 5 / 13 * 100
                           for c in metrics['fixed']['per_class']}
    report = dict(completed=len(rows), new_vlm_requests=0, replay_seconds=elapsed,
        protocol=json.loads((HERE / 'protocol.json').read_text(encoding='utf-8')),
        summary=summary, selected_diagnostics=selected_diagnostics,
        class_contribution_to_avg_percentage_points=class_contributions,
        metrics=metrics, metrics_before_fixed_tail=pre_metrics, per_video=rows)
    (HERE / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    csv_rows = [dict(video=r['video'], category=r['category'], gt_seconds=json.dumps(r['gt_seconds']),
                    policy=p, **{k: json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v
                                 for k,v in r['arms'][p].items()}) for r in rows for p in POLICIES]
    with (HERE / 'per_video.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    def percent(x): return f'{100*x:.2f}%'
    def score(x): return '—' if x is None else f'{x:.3f}'
    metric_rows = ''.join('<tr><td>'+LABELS[p]+'</td>'+''.join(
        f'<td>{percent(metrics[p]["values"][k])}</td>' for k in ('mAP@0.1','mAP@0.3','mAP@0.5','AVG'))
        +f'<td>{summary[p]["accepted_candidates"]}</td><td>{summary[p]["final_changed_videos"]}</td>'
        +f'<td>{summary[p]["overwritten_candidates"]}</td></tr>' for p in POLICIES)
    pre_rows = ''.join(f'<tr><td>{LABELS[p]}</td><td>{percent(pre_metrics[p]["values"]["AVG"])}</td>'
                      f'<td>{percent(metrics[p]["values"]["AVG"])}</td></tr>' for p in POLICIES)
    video_rows = ''.join(f'<tr><td>{r["category"]}<small>{r["video"]}</small></td>'
        +''.join(f'<td>{score(r["arms"][p]["mean_best_tiou"])}<small>{html.escape(str(r["arms"][p]["final_intervals"]))}</small></td>'
                  for p in ('fixed','relaxed_node','force_best'))+'</tr>' for r in rows)
    conclusions = '；'.join(f'{LABELS[p]}接受 {summary[p]["accepted_candidates"]} 个提案，'
        f'最终改变 {summary[p]["final_changed_videos"]} 个视频，'
        f'AVG 相对直接后处理变化 {(metrics[p]["values"]["AVG"]-metrics["fixed"]["values"]["AVG"])*100:+.2f} 个百分点'
        for p in ('relaxed_node','force_best'))
    contributions = '、'.join(f'{c} {v:+.2f} 个百分点' for c,v in class_contributions.items() if abs(v)>1e-9)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MCTS 接收条件消融</title>
<style>body{{font:15px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;color:#25382f}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{border:1px solid #d4ddd6;padding:10px;text-align:left}}th{{background:#edf3ef}}small{{display:block;color:#65726a}}pre{{white-space:pre-wrap}}a{{color:#216749}}</style>
<h1>Stage 4 → MCTS 接收条件 → 固定 BUE→CBT</h1><p><strong>{html.escape(conclusions)}。</strong></p>
<p>同一批 14 个视频，复用上一轮 Stage 4 候选和完整 MCTS 搜索树、VLM 回答与奖励，仅改变接收规则。重建真实视频的 CLIP 曲线并重新执行原有 BUE→CBT；直接后处理和严格接收两条基线逐视频精确复现。新 VLM 请求：0。</p>
<ul><li>严格：原有完整性、未切断、CLIP 质量及最终 A/B 证据比较。</li>
<li>放宽 relaxed_node：仅保留目标存在、同一事件、CLIP 质量不明显下降，取消完整性/切断否决和最终 A/B 比较。</li>
<li>强制 force_best：每个已搜索候选直接采用评分最高的有效非根节点，允许分数低于原区间；取消语义和 CLIP 接收门槛及最终比较。</li></ul>
<p>每个候选只输出一个区间，不把搜索树所有节点变成检测结果。无合法已评分提案的候选保留原区间。几何合法性、证据覆盖、原搜索预算均保持；本次没有重新搜索未进入预算的候选。强制接收也不会采用无效 JSON 或失败评价。</p>
<h2>最终定位结果</h2><p>使用当前 evaluate.py：VadCLIP 非插值 AP，13 类异常，原视频 FPS，IoU 0.1—0.5；AVG 为五个阈值的平均。正常视频误报单独统计。</p>
<table><tr><th>策略</th><th>mAP@0.1</th><th>mAP@0.3</th><th>mAP@0.5</th><th>AVG</th><th>接受候选</th><th>最终改变视频</th><th>被固定尾流程恢复的候选</th></tr>{metric_rows}</table>
<p>放宽组 AVG 变化的类别贡献：{html.escape(contributions)}。这是各类别 AP 对宏平均的贡献，不是逐类 AP 本身。主要收益集中在少数样本，需要更大样本复核。</p>
<h2>固定后处理前后</h2><p>前列使用接收后的搜索区间直接评估；用于观察 BUE→CBT 的作用，类别与置信度保持相同。</p>
<table><tr><th>策略</th><th>后处理前 AVG</th><th>后处理后 AVG</th></tr>{pre_rows}</table>
<h2>逐视频边界</h2><p>数值为每个 GT 的最佳 tIoU 再求均值，不计类别；下方为最终区间（秒）。严格策略与直接后处理相同，完整数据见 JSON。</p>
<table><tr><th>视频</th><th>直接 BUE→CBT</th><th>放宽接收后 BUE→CBT</th><th>强制接收后 BUE→CBT</th></tr>{video_rows}</table>
<h2>范围与统计</h2><p>本次是固定搜索树的接收消融，不是重新执行整条推理流程；不能据此比较不同策略的新请求数或独立运行耗时。只有每类一个视频，不能代表完整数据集。原有 CBT 时长先验保持不变，历史统计来源未独立核实；未使用本轮 GT 调整接收条件。应用默认仍为严格接收。</p>
<pre>{html.escape(json.dumps(dict(completed=len(rows),new_vlm_requests=0,replay_seconds=elapsed,summary=summary,selected_diagnostics=selected_diagnostics),ensure_ascii=False,indent=2))}</pre>
<p><a href="comparison.json">完整 JSON</a> · <a href="per_video.csv">逐视频 CSV</a> · <a href="protocol.json">实验协议</a> · <a href="executed_source.zip">执行源码</a> · <a href="../stage4_fixed_vs_mcts/report.html">原始搜索报告</a></p></html>'''
    (HERE / 'report.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(summary=summary,metrics={p:metrics[p]['percent'] for p in POLICIES},
        before_fixed_tail={p:pre_metrics[p]['percent']['AVG'] for p in POLICIES}),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
