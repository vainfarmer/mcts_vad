"""离线分析共享上游/搜索的后处理消融，不调用模型。"""
from pathlib import Path
from collections import Counter
import json
import sys

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parents[1]/'video_anomaly_detection'))
from evaluate import load_ground_truth, compute_temporal_map, compute_frame_auc, compute_video_level

labels=load_ground_truth('F:/dataset/ucf_crime/data_parquet')
selected=json.loads((here/'selected_videos.json').read_text(encoding='utf-8'))
ids={Path(v['video_path']).stem for v in selected}
arms={k:{} for k in ('legacy_off','legacy_on','dedup_off','dedup_on','guard_on','skip_only')}
rows=[]; guard_reasons=Counter(); guard_stages=Counter(); applied=0; simulations=0

def boundaries(r):
    return sorted((a['t_start'],a['t_end']) for a in r['anomalies'])

for name in sorted(ids):
    raw=json.loads((here/'results'/f'{name}.json').read_text(encoding='utf-8'))
    paired=raw['paired_results']
    branch=dict(legacy_off=paired['legacy_off'],legacy_on=paired['legacy_on'],
                dedup_off=paired['original_off'],dedup_on=paired['dedup_on'],guard_on=raw,skip_only=paired['skip_only'])
    for arm,r in branch.items():
        value={k:v for k,v in r.items() if k not in ('paired_results','token_usage','token_usage_all_attempts')}
        arms[arm][name]=value
        dest=here/'comparison_arms'/arm; dest.mkdir(parents=True,exist_ok=True)
        (dest/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    trace=raw.get('mcts_trace') or {}
    for c in trace.get('candidates',{}).values():
        applied+=bool(c.get('applied'));simulations+=max(0,len(c.get('evaluations',[]))-1)
    for event in trace.get('boundary_guard_events',[]):
        if event.get('reverted'):
            guard_stages[event['stage']]+=1;guard_reasons.update(event['reasons'])
    rows.append(dict(video=name,category=labels[name]['category'],
                     boundaries={k:boundaries(v) for k,v in branch.items()},
                     guard_events=trace.get('boundary_guard_events',[])))

gt={k:labels[k] for k in ids}
metrics={k:dict(temporal=compute_temporal_map(gt,rs,30),
                frame=compute_frame_auc(gt,rs,30),video=compute_video_level(gt,rs),
                predictions=sum(len(r['anomalies']) for r in rs.values()),
                tail_seconds=sum((r.get('mcts_trace') or {}).get('tail_seconds',0) for r in rs.values()))
         for k,rs in arms.items()}
comparisons={}
for a,b in [('legacy_off','legacy_on'),('legacy_off','dedup_off'),('legacy_on','dedup_on'),('dedup_on','guard_on'),('dedup_off','guard_on')]:
    comparisons[f'{b}_minus_{a}']=dict(map_delta=metrics[b]['temporal']['mAP']-metrics[a]['temporal']['mAP'],
        boundary_changed_videos=sum(boundaries(arms[a][k])!=boundaries(arms[b][k]) for k in ids))
usage=json.loads((here/'results/metadata/token_usage.json').read_text(encoding='utf-8'))
report=dict(videos=len(ids),metrics=metrics,comparisons=comparisons,applied_candidates=applied,
            evaluated_simulations=simulations,guard_reasons=dict(guard_reasons),guard_stages=dict(guard_stages),
            token_usage=usage['usage'],per_video=rows)
(here/'ablation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# v216 后处理配对消融', '',
       '同一批14视频，共享上游和MCTS搜索；相同API请求复用。legacy为旧尾流程，dedup增加最终NMS，guard在dedup基础上增加边界保护。', '',
       '| 分支 | mAP | AP@0.3 | AP@0.5 | AP@0.7 | Frame AUC | 检测数 |',
       '|---|---:|---:|---:|---:|---:|---:|']
for arm,m in metrics.items():
    t=m['temporal'];lines.append(f"| {arm} | {t['mAP']:.6f} | {t['AP@0.3']:.6f} | {t['AP@0.5']:.6f} | {t['AP@0.7']:.6f} | {m['frame']['auc']:.6f} | {m['predictions']} |")
lines+=['','## 控制变量比较','']
for k,c in comparisons.items():lines.append(f"- {k}: mAP差值 {c['map_delta']:+.6f}，边界或数量变化视频 {c['boundary_changed_videos']}。")
lines+=['','## 保护与耗时','',f'接受候选数：{applied}；子节点评估次数：{simulations}。',
        f'保护回退阶段：{dict(guard_stages)}。原因计数（一次可有多个原因）：{dict(guard_reasons)}。',
        '', 'API时间均为所有请求的累计秒数，不等于墙钟耗时；请求等待已在CALL日志queue_wait_seconds独立记录。',
        '```json',json.dumps(usage['usage'],ensure_ascii=False,indent=2),'```','',
        'Frame scores沿用原有post_conf_gate计算位置；BUE/CBT之后的最终NMS不重新计算frame scores。因此边界或重复检测改善可能不改变Frame AUC。',
        '本实验验证14视频，不能据此确认65视频或完整测试集收益。总token属于整个多分支实验，不能视为单个on/off生产分支成本。',
        '最终NMS后的pre_bue/pre_cbt数组已按保留结果重新排列，但开启保护时历史边界不再能直接重构“关闭保护”分支；应使用明确保存的消融分支。',
        '', '完整逐视频区间见ablation.json；comparison_arms下保存每个分支的完整结果。']
lines+=['','## 当前配置建议','',
        '常规run_relaxed_search.ps1启用最终NMS，边界保护保持关闭。14视频最终NMS减少2条检测且mAP不变；v215冻结65视频的离线对照中，on mAP由0.155292升至0.155802，off由0.152019升至0.152725，详情见offline_nms_65.json。',
        '当前几何保护使mAP从0.169408降至0.161616，不应默认启用，也不在本次小样本上反复调阈值追求提升。',
        'Arson035的最终区间从4.5–45回退为22–33.5，最佳GT IoU从0.2469升至0.5926；但Vandalism015从55.5–90回退为55.5–67，IoU从0.6667降至0。说明已接受的局部区间可能仍缺失事件，不适合作为所有后处理的硬上限。',
        '后续应优先考虑条件式保护：大幅迁移时重新核对事件身份/核心证据，而非仅凭几何范围否定BUE补全。该方向尚未实现，当前保留关闭开关和明确的消融结果。']
(here/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','per_video','token_usage')},ensure_ascii=False,indent=2))
for name,m in metrics.items(): print(name,m['temporal'],m['frame']['auc'],m['predictions'])
