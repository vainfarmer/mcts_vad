"""Validation and read models; never rewrite the detector's scores or evidence."""
from __future__ import annotations
import csv
import html
import io
import json
import math
from pathlib import Path
from .settings import CATEGORIES


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_detection(row, duration):
    if not isinstance(row, dict):
        raise ValueError('区间必须为对象。')
    start, end = row.get('t_start'), row.get('t_end')
    if not finite(start) or not finite(end) or not 0 <= start < end <= duration + 1e-6:
        raise ValueError('区间必须满足 0 ≤ 开始 < 结束 ≤ 视频时长。')
    if row.get('anomaly_type') not in (*CATEGORIES, 'unknown'):
        raise ValueError('未知异常类别。')
    if not finite(row.get('confidence', 1)):
        raise ValueError('检测评分必须为有限数值。')


def validate_result(result, duration=None):
    if not isinstance(result, dict) or not finite(result.get('duration')) or result['duration'] <= 0:
        raise ValueError('检测结果缺少有效视频时长。')
    if duration is not None and abs(result['duration'] - duration) > max(.5, duration * .01):
        raise ValueError('结果时长与输入视频不一致。')
    if not isinstance(result.get('anomalies'), list):
        raise ValueError('检测结果缺少区间列表。')
    if type(result.get('is_normal')) is not bool or result['is_normal'] != (not result['anomalies']):
        raise ValueError('检测结果的正常标记与区间列表不一致。')
    for row in result['anomalies']:
        validate_detection(row, result['duration'])
    scores, times = result.get('frame_scores'), result.get('timestamps')
    if scores is not None or times is not None:
        if not isinstance(scores, list) or not isinstance(times, list) or len(scores) != len(times):
            raise ValueError('曲线和时间轴长度不一致。')
        if any(not finite(x) for x in scores + times):
            raise ValueError('曲线包含无效数值。')
        if any(a >= b for a, b in zip(times, times[1:])) or any(t < 0 or t > result['duration'] + .1 for t in times):
            raise ValueError('曲线时间轴无效。')
    return result


def analysis(store, job_id):
    job = store.job(job_id)
    result = store.result(job_id)
    attempt = store.attempt(job['result_attempt_id'])
    directory = store.path(attempt['artifact_dir'])
    research = directory / 'research.json'
    detail = json.loads(research.read_text(encoding='utf-8')) if research.exists() else {}
    trace = result.get('mcts_trace') or {}
    stages=detail.get('stages',trace.get('stages',{}))
    known=['coarse_candidates','discovery_candidates','post_clip','post_mcts',
           'post_stage5b','post_merge','post_nms','post_tighten','post_short','post_classify',
           'post_curve','pre_conf_gate','post_conf_gate','pre_qscore','post_qscore',
           'post_bue','post_cbt','final']
    order=detail.get('stage_order') or [k for k in known if k in stages]+[k for k in stages if k not in known]
    return dict(schema_version=1, job_id=job_id, attempt_id=attempt['id'],
                duration=result['duration'], is_normal=result.get('is_normal'),
                detections=[dict(row, detection_id=f'det-{i}', score_semantics='ranking_score_uncalibrated',
                                 description_stage='discovery') for i, row in enumerate(result['anomalies'])],
                stage_trace=stages,stage_order=order,
                stage_seconds=trace.get('stage_seconds', {}), mcts=trace,
                evidence=detail.get('evidence', []), classification=detail.get('classification'),
                observations=detail.get('observations',{}),
                exit_reason=detail.get('exit_reason', '原因未记录'),
                reviews=store.reviews(job_id), usage=result.get('token_usage'),
                runtime=attempt.get('metrics'), config=job['config'],
                recorded=bool(detail), raw_result_url=f'/api/jobs/{job_id}/result')


def curves(store, job_id):
    job = store.job(job_id)
    result = store.result(job_id)
    a = store.attempt(job['result_attempt_id'])
    path = store.path(a['artifact_dir'] + '/curves.json')
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    if result.get('frame_scores') is not None:
        data = [r for r in data if r['name'] != 'frame_scores'] + [
            dict(name='frame_scores', timestamps=result['timestamps'], values=result['frame_scores'],
                 semantics='融合帧级评分；在末尾 Q-score/BUE/CBT 之前生成')]
    return data


def review_projection(result, reviews):
    rows = {f'det-{i}': dict(row, source='model') for i, row in enumerate(result['anomalies'])}
    for r in reviews:
        if r['after'] is None:
            rows.pop(r['subject_id'], None)
        else:
            rows[r['subject_id']] = dict(r['after'], source='review', revision=r['revision'])
    return rows


def export_csv(result, reviews):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['source', 'event_id', 'class', 'start_seconds', 'end_seconds', 'score', 'description'])

    def cell(value):
        text = str(value or '')
        return "'" + text if text.startswith(('=', '+', '-', '@')) else text

    for i, row in enumerate(result['anomalies']):
        writer.writerow(['model', f'det-{i}', row['anomaly_type'], row['t_start'], row['t_end'],
                         row['confidence'], cell(row.get('description'))])
    if reviews:
        for key, row in review_projection(result, reviews).items():
            writer.writerow(['reviewed', key, row['anomaly_type'], row['t_start'], row['t_end'],
                             row.get('confidence', ''), cell(row.get('description'))])
    return '\ufeff' + stream.getvalue()


def export_html(data, filename):
    esc = lambda v: html.escape(str(v))
    rows = ''.join(f"<tr><td>{esc(d['anomaly_type'])}</td><td>{d['t_start']:.3f}</td>"
                   f"<td>{d['t_end']:.3f}</td><td>{d['confidence']:.4f}</td>"
                   f"<td>{esc(d.get('description') or '未记录')}</td></tr>" for d in data['detections'])
    reviews = ''.join(f"<li>{esc(r['subject_id'])} · v{r['revision']} · {esc(r['action'])}"
                      f" · {esc(r['reason'])}<pre>{esc(json.dumps(r['after'], ensure_ascii=False))}</pre></li>"
                      for r in data['reviews'])
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
    <title>视频分析报告</title><style>body{{font:15px/1.7 system-ui;max-width:1050px;margin:40px auto;color:#183831}}
    table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:10px;text-align:left}}
    pre{{white-space:pre-wrap;overflow-wrap:anywhere}}@media print{{body{{margin:15mm}}}}</style>
    <h1>视频异常检测与时序定位分析报告</h1><p>{esc(filename)} · {data['duration']:.3f} 秒</p>
    <p>任务 {esc(data['job_id'])} / 执行 {esc(data['attempt_id'])}</p>
    <p>原始模型结果。检测评分未校准为概率；描述来自发现阶段，原区间可能与最终边界不同。</p>
    <table><tr><th>类别</th><th>开始（秒）</th><th>结束（秒）</th><th>评分</th><th>原始描述</th></tr>{rows}</table>
    <h2>人工复核记录</h2><ul>{reviews or '<li>无</li>'}</ul>
    <h2>执行配置</h2><pre>{esc(json.dumps(data['config'], ensure_ascii=False, indent=2))}</pre>
    <h2>请求用量</h2><pre>{esc(json.dumps(data['usage'], ensure_ascii=False, indent=2))}</pre>
    <p>无独立 GT 的上传视频不报告准确率。可使用浏览器打印保存为 PDF。</p></html>"""
