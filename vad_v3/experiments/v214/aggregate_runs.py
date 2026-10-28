from pathlib import Path
import json,re
here=Path(__file__).resolve().parent
report=json.loads((here/'summary.json').read_text(encoding='utf-8'))
runs=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((here/'results/metadata').glob('token_usage_*.json'))]
fields=('requests','successful_requests','failed_requests','unknown_usage_requests','prompt_tokens','completion_tokens','total_tokens')
combined={k:sum(r['usage'][k] for r in runs) for k in fields}
calls=[json.loads(line) for r in runs for line in (here/'results/metadata'/r['calls_file']).read_text(encoding='utf-8').splitlines()]
assert combined['requests']==len(calls)
assert combined['total_tokens']==sum(c['total_tokens'] or 0 for c in calls)
logs=[here/'v214_console.log',here/'v214_resume1_console.log']
times=[float(re.search(r'Total wall-clock time:\s+([\d.]+)s',p.read_text(encoding='utf-8')).group(1)) for p in logs]
report['last_execution_timing']=report['timing']
report['timing']={'execution_count':len(times),'batch_wall_seconds_by_run':times,'sum_batch_wall_seconds':sum(times),'concurrency':2,'note':'Excludes model loading and gaps between runs; first run contains failed attempt.'}
report['token_usage']=combined
report['run_ids']=[r['run_id'] for r in runs]
accepted=[];changed=[];searched=0;compared=0;repairs=0
for p in sorted((here/'results').glob('*.json')):
 r=json.loads(p.read_text(encoding='utf-8'))
 if 'paired_results' not in r:continue
 off=r['paired_results']['original_off']
 if r['anomalies']!=off['anomalies']:changed.append(p.stem)
 for c in (r.get('mcts_trace') or {}).get('candidates',{}).values():
  searched+=int(c.get('logical_requests',0)>0)
  compared+=int(bool(c.get('pairwise_attempts')))
  repairs+=max(0,len(c.get('pairwise_attempts',[]))-1)
  if c.get('applied'):accepted.append({'video':p.stem,'id':c['candidate_id'],'baseline':c['baseline'],'proposed':c['proposed'],'verdict':c['pairwise_verdict'],'final_off':off['anomalies'],'final_on':r['anomalies']})
boundary_changed=[]
for x in accepted:
 off=sorted((a['t_start'],a['t_end']) for a in x['final_off'])
 on=sorted((a['t_start'],a['t_end']) for a in x['final_on'])
 if off!=on and x['video'] not in boundary_changed:boundary_changed.append(x['video'])
report['acceptance_audit']={'boundary_changed_count':len(boundary_changed),'videos_with_changed_final_boundaries':boundary_changed,'searched':searched,'compared':compared,'repair_attempts':repairs,'accepted':accepted,'videos_with_changed_final_anomalies':changed}
(here/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ('refinement','token_usage','timing','bootstrap_on_minus_off')},indent=2))
for name,arm in report['arms'].items():print(name,arm['temporal'],arm['frame']['auc'])
print('accepted',[(x['video'],x['id']) for x in accepted],'changed',changed,'searched',searched)
