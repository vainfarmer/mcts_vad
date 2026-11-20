"""Create a compact reviewable report from completed, frozen v218 metrics."""
import html
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
LABELS={'baseline':'原 MCTS+BUE/CBT 基线','yesno_probability':'独立 Yes/No 概率',
        'fusion_50':'v218：等权融合（预先确定的主结果）','yesno_hard_diagnostic':'离散 Yes/No 对照'}


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    reports={s:read(HERE/(s+'_metrics.json')) for s in ['full290','pilot65','remaining225']}
    full=reports['full290'];manifest=read(HERE/'manifest.json');progress=read(HERE/'progress.json')
    assert progress['status']=='completed' and full['counts']['videos']==290 and full['counts']['normal_videos']==150
    primary=manifest['primary_arm'];base=full['metrics']['baseline'];chosen=full['metrics'][primary]
    summary=dict(version='v218',primary_arm=primary,completed_videos=290,abnormal_videos=140,normal_videos=150,
                 reused_pilot_videos=65,new_inference_videos=225,video_concurrency=2,api_concurrency=4,
                 primary_metrics={k:chosen[k] for k in ['global_auc','ano_auc','aupr']},
                 primary_auc_delta_pp=100*(chosen['global_auc']-base['global_auc']),
                 all_arms={k:{m:v[m] for m in ['global_auc','ano_auc','aupr']} for k,v in full['metrics'].items()},
                 counts=full['counts'],max_observed_video_workers=progress['counts']['max_observed_video_workers'],
                 max_observed_api_requests=progress['counts']['max_observed_api_requests'],
                 failed_videos=progress['failed_videos'],limitations=full['limitations'])
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    sections=[]
    for scope,title in [('full290','完整 290 视频'),('pilot65','先前 65 视频'),('remaining225','新增 225 视频')]:
        report=reports[scope]
        sections.append('<h2>'+title+'</h2><p>'+str(report['counts']['videos'])+' 个视频；Normal '+str(report['counts']['normal_videos'])+' 个。</p>'+table(
            ['方法','全局帧级 AUC','Ano-AUC','AUPR','相对本范围基线 ΔAUC'],
            [[LABELS[k],*[f'{100*v[m]:.3f}%' for m in ['global_auc','ano_auc','aupr']],
              f'{100*(v["global_auc"]-report["metrics"]["baseline"]["global_auc"]):+.3f} pp'] for k,v in report['metrics'].items()]))
    sections.append('<h2>类别内部帧级 AUC（140 个异常视频）</h2>'+table(
        ['类别','基线','独立概率','v218 融合'],
        [[c,*[f'{100*full["metrics"][k]["per_category_auc"][c]:.2f}%' for k in ['baseline','yesno_probability','fusion_50']]]
         for c in sorted(base['per_category_auc'])]))
    sections.append('<h2>实验完整性</h2><ul><li>固定 5 秒目标间隔、10 秒视觉上下文、最多 10 帧；类别盲测；不向模型提供文件名、真实类别或原分类器标签。</li>'
        '<li>主结果在全量测试前固定为等权融合；其余分支作为对照，不按全量测试结果重新选择权重。</li>'
        '<li>复用的 65 个视频保留原始结果哈希和来源，新增 225 个视频独立视觉推理。完整集合包含先前的小样本集合。</li>'
        '<li>全量 AUC 拼接原始视频帧，包含 150 个纯正常视频。Ano-AUC 仅使用异常视频内部的正常和异常帧。</li>'
        '<li>本实验改变帧级分数，不重新运行 MCTS 边界搜索，不声称定位 mAP 提升。基线及融合仍有之前记录的 VideoMAE 训练划分来源限制。</li></ul>')
    sections.append('<p>总窗口数 '+str(full['counts']['windows'])+'；完整 Yes/No 概率对 '+str(full['counts']['exact_pairs'])+'；top-5 截断近似 '+str(full['counts']['bounded_pairs'])+'。</p>')
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>v218 · UCF-Crime 全量 AUC 测试</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#192733;background:#fafbfc}h1{font-size:28px}h2{font-size:21px;margin-top:36px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px 12px;border-bottom:1px solid #dce2e8;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eaf0f6}p,li{max-width:1000px}.lead{padding:18px;background:#e8f3ef;border-left:4px solid #287b5e}</style><h1>v218 · UCF-Crime 全量 AUC 测试</h1>'
    doc+='<p class="lead">预先确定的主结果：等权融合。全量 AUC '+f'{100*chosen["global_auc"]:.3f}%'+', 相对同范围基线 '+f'{summary["primary_auc_delta_pp"]:+.3f} 个百分点'+'。视频并发 2，API 并发上限 4。</p>'
    doc+=''.join(sections)+'</html>'
    (HERE/'report.html').write_text(doc,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
