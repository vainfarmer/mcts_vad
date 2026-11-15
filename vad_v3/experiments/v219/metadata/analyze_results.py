"""Post-run artifact checks and readable report; never alters predictions."""
from pathlib import Path
import collections
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

report = read(ROOT/'evaluation_report.json')
manifest = read(ROOT/'metadata/manifest.json')
upstream_report = read(Path(manifest['runtime_settings']['source_dir'])/'evaluation_report.json')['evaluation']
assert report['arms']['baseline_off']['temporal_mAP_full'] == upstream_report['temporal_mAP_full']
assert report['arms']['baseline_off']['frame_AP'] == upstream_report['frame_AP']
selection = read(ROOT/'metadata/selection.json')
expected = sorted({n for rows in selection['slots'].values() for n in rows})
files = {p.stem:p for p in (ROOT/'result').glob('*.json')}
assert sorted(files) == expected and len(expected) == 35
results = {n:read(files[n]) for n in expected}
arm_names = ['mcts','mcts_qscore','mcts_qscore_cbt','mcts_qscore_cbt_bue','mcts_cbt_bue']
directions = collections.Counter()
searched_by_class = collections.Counter()
applied_by_class = collections.Counter()
rewards = collections.Counter()
geometry = {}
def signature(events):
    return sorted((e['anomaly_type'],round(e['t_start'],6),round(e['t_end'],6)) for e in events)
for n,r in results.items():
    assert r['experiment_fingerprint'] == manifest['fingerprint']
    assert set(r['arms']) == set(arm_names)
    assert r['upstream_frame_score_max_error'] <= 1e-4
    seed_ids = set(range(len(r['shared']['seed_events'])))
    for arm in r['arms'].values():
        assert len(arm['frame_scores']) == len(r['timestamps'])
        assert all(math.isfinite(x) and 0<=x<=1 for x in arm['frame_scores'])
        for e in arm['anomalies']:
            assert 0<=e['t_start']<e['t_end']<=r['duration'] and 0<=e['confidence']<=1
            assert set(e['source_candidate_ids']) <= seed_ids
            if 'qscore' in e:
                assert e['qscore']['evaluated_interval'] == [e['t_start'],e['t_end']]
        for i,a in enumerate(arm['anomalies']):
            for b in arm['anomalies'][i+1:]:
                if a['anomaly_type'] != b['anomaly_type']: continue
                overlap=max(0,min(a['t_end'],b['t_end'])-max(a['t_start'],b['t_start']))
                union=max(a['t_end'],b['t_end'])-min(a['t_start'],b['t_start'])
                assert overlap/union < .5+1e-8
    for item in r['shared']['mcts_trace']['candidates'].values():
        category=item['category']
        if item.get('logical_requests',0)>0: searched_by_class[category]+=1
        if not item.get('applied'):continue
        applied_by_class[category]+=1
        a,b=item['baseline'];c,d=item['proposed']
        if c>=a and d<=b: direction='shrink'
        elif c<=a and d>=b: direction='expand'
        elif abs((c-a)-(d-b))<1e-7: direction='shift'
        else: direction='mixed'
        directions[direction]+=1
        gain=item['reward_gain']
        rewards['gain' if gain>1e-8 else 'loss' if gain< -1e-8 else 'tie']+=1
for arm in arm_names:
    geometry[arm]=dict(
        videos_geometry_changed_from_off=sum(signature(r['arms'][arm]['anomalies'])!=signature(r['baseline_off']['anomalies']) for r in results.values()),
        videos_geometry_changed_from_mcts=sum(signature(r['arms'][arm]['anomalies'])!=signature(r['arms']['mcts']['anomalies']) for r in results.values()),
        median_predicted_duration=statistics.median(e['t_end']-e['t_start'] for r in results.values() for e in r['arms'][arm]['anomalies']))
for filename,digest in manifest['source_sha256'].items():
    assert hashlib.sha256((ROOT/'metadata/source_snapshot'/filename).read_bytes()).hexdigest()==digest
upstream=Path(manifest['runtime_settings']['source_dir'])
for name,digest in manifest['runtime_settings']['upstream_results_sha256'].items():
    assert hashlib.sha256((upstream/(name+'.json')).read_bytes()).hexdigest()==digest
usage=read(ROOT/'metadata/token_usage.json')
jobs=read(ROOT/'metadata/jobs.json')
probes=[read(p) for p in (ROOT/'metadata').glob('model_probe*.json')]
summary=dict(video_count=len(results), artifact_checks='passed',
    source_snapshots_verified=len(manifest['source_sha256']), upstream_results_unchanged=True,
    mcts=report['mcts'], applied_direction_counts=dict(directions), reward_gain_counts=dict(rewards),
    searched_by_predicted_class=dict(searched_by_class), applied_by_predicted_class=dict(applied_by_class),
    geometry=geometry, tokens=usage['usage'], wall_seconds=jobs['wall_seconds'],
    mean_video_seconds=statistics.mean(r['processing_time'] for r in results.values()),
    max_upstream_frame_score_error=max(r['upstream_frame_score_max_error'] for r in results.values()),
    probe_requests=sum(p['token_usage']['requests'] for p in probes),
    probe_known_tokens=sum(p['token_usage']['total_tokens'] for p in probes),
    probe_unknown_usage_requests=sum(p['token_usage']['unknown_usage_requests'] for p in probes))
write(ROOT/'analysis_summary.json',summary)
labels={'baseline_off':'v218 / MCTS off','mcts':'MCTS 强制接收','mcts_qscore':'MCTS + Q-score',
        'mcts_qscore_cbt':'MCTS + Q-score + CBT','mcts_qscore_cbt_bue':'MCTS + Q-score + CBT + BUE',
        'mcts_cbt_bue':'MCTS + CBT + BUE'}
lines=['# v219：XD 35 视频五组配对消融','',
    '35/35 个视频完成。六类异常和 Normal 各指定 5 个视频，与 v218 完全相同；多类别视频仍保留所有预测类别。五组共用 v218 上游结果及一次 MCTS 搜索。','',
    '| 组别 | 帧 AP % | mAP@0.1 % | @0.2 % | @0.3 % | @0.4 % | @0.5 % | AVG % | 区间数 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for arm,metric in report['arms'].items():
    p=metric['temporal_mAP_full']['percent']
    values=[100*metric['frame_AP']]+[p[f'mAP@{t:.1f}'] for t in (.1,.2,.3,.4,.5)]+[p['AVG']]
    lines.append('| '+labels[arm]+' | '+' | '.join(f'{x:.3f}' for x in values)+f" | {metric['total_predictions']} |")
lines += ['', '时间 IoU 使用每个源视频的真实 FPS，预测转换为半开帧区间；六类分别计算 AP 后取宏平均。Normal 不作为异常类别参与宏平均，其误报仍计入对应类别 FP。帧 AP 与时间 IoU mAP 是不同指标。','',
    '## 本轮结论','',
    '时间定位 mAP 最好的是 **MCTS + CBT + BUE：21.624%**，相对 v218 off 的 18.018% 提高 **3.606 个百分点**；相对仅 MCTS 的 19.133% 提高 2.491 个百分点。5 个 Normal 视频在所有分支均无误报。',
    '本轮 XD Q-score 重排降低了 mAP：仅 MCTS 加 Q-score 从 19.133% 降到 13.824%；CBT+BUE 组合加 Q-score 从 21.624% 降到 17.881%。因此，当前这版 Q-score 不宜直接作为 XD 默认项。这是本次固定尺度原始 CLIP 质量分的结果，不代表所有 Q-score 设计都无效。',
    '在 Q-score+CBT 的基础上加 BUE，mAP 从 12.861% 升到 17.881%。外扩在本批短候选上有帮助；但没有单独的 MCTS+CBT 和 MCTS+BUE 两组，不能把最佳组合的提升全部归于某一个模块。',
    '仅 MCTS 的帧 AP 为 64.579%，略高于 CBT+BUE 组合的 64.465%；选择配置应区分时间定位 mAP 与帧级 AP 的目标。','',
    '## MCTS 与后处理实际执行情况','',
    f"- 种子 {report['mcts']['total_candidates']} 个，执行搜索 {report['mcts']['searched']} 个，采用 {report['mcts']['applied']} 个。未搜索候选仍保留，并参与各组后处理。",
    '- 接受的几何变化：'+json.dumps(dict(directions),ensure_ascii=False)+'。',
    '- 已接受提案相对根节点奖励：'+json.dumps(dict(rewards),ensure_ascii=False)+'。force_best 允许奖励下降，applied 不等于指标改善。',
    '- 搜索/未采用原因：'+json.dumps(report['mcts']['reasons'],ensure_ascii=False)+'。','',
    '| 组别 | CBT 改变/检查 | BUE 改变/检查 | Q-score 改分/检查 | 相对 off 几何改变的视频数 |',
    '|---|---:|---:|---:|---:|']
for arm in arm_names:
    operators=report['arms'][arm]['operators']
    vals=[f"{operators[k]['changed']}/{operators[k]['checked']}" if k in operators else '—' for k in ('cbt','bue','qscore')]
    lines.append('| '+labels[arm]+' | '+' | '.join(vals)+f" | {geometry[arm]['videos_geometry_changed_from_off']} |")
lines += ['', 'CBT/BUE 是 XD 类别曲线的实验适配，没有套用 UCF 的类别时长上限。执行顺序是 MCTS → 可选 CBT → 可选 BUE → 同类相邻合并 → 可选 Q-score → 同类 NMS。Q-score 使用最终几何的原始类别曲线并重排置信度，不借用输出分数进行自我评分。','',
    '## 耗时与用量','',
    f"批次处理耗时 {jobs['wall_seconds']:.2f} 秒（不含 CLIP 初始化）；单视频含预处理排队平均 {summary['mean_video_seconds']:.2f} 秒。诊断并发 2，API 上限 4。",
    f"实际主测请求 {usage['usage']['requests']} 次，失败 {usage['usage']['failed_requests']} 次，已知 token {usage['usage']['total_tokens']:,}。五组共享同一批调用，不能将此用量乘以五；也不含 v218 已发生的分类成本。",
    f"另外模型可用性探测 {summary['probe_requests']} 次，已知 token {summary['probe_known_tokens']}，其中 {summary['probe_unknown_usage_requests']} 次失败请求未返回用量。两个替代模型返回 403，主测采用可用的 qwen3-vl-plus。",'',
    '## 结果路径与检查','',
    '- `result/<video_id>.json`：每视频一个文件，五组最终结果在 `arms.<组名>.anomalies`。共享种子、MCTS 后边界和完整树轨迹在 `shared`。',
    '- `evaluation_report.json`：完整指标和每类 AP；`analysis_summary.json`：搜索、后处理及用量汇总。',
    '- `metadata/matching_<组名>.json`：逐预测匹配审计；`metadata/video_fps.json`：源 FPS；`metadata/manifest.json`：固定配置、上游哈希。',
    '- 所有输出边界、分数、同类 NMS、五组完整性、Q-score 边界一致性均已校验；源码快照哈希和 v218 输入哈希一致。',
    f"- CLIP/frame_scores 复现最大误差：{summary['max_upstream_frame_score_error']:.8g}。",'',
    '此结果只反映已观察过的 35 视频上的探索性实验，未运行完整 800 视频测试集。强制接收只能改已有候选的边界，无法自动修复上游类别混淆或完全漏检。']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
