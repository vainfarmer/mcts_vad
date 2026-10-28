from pathlib import Path
import os,sys,subprocess,json
here=Path(__file__).resolve().parent;code=here.parents[1]/'video_anomaly_detection'
for line in (code/'.env').read_text(encoding='utf-8-sig').splitlines():
 if line.strip() and not line.lstrip().startswith('#') and '=' in line:
  k,v=line.split('=',1)
  if k.strip()=='DASHSCOPE_API_KEY':os.environ[k.strip()]=v.strip().strip('\"\x27')
assert os.environ.get('DASHSCOPE_API_KEY'),'Missing API key'
os.environ['PYTHONIOENCODING']='utf-8';os.environ['HF_HUB_OFFLINE']='1'
args=[sys.executable,'-u',str(code/'batch_evaluate.py'),'--dataset_dir',str(here/'dataset'),'--output_dir',str(here/'results'),'--detect_model','qwen3-vl-plus','--classify_model','qwen3-vl-plus','--max_per_category','1','--concurrency','1','--clip_model','ViT-L-14','--device','cuda','--mcts_mode','refine','--mcts_paired_eval','--mcts_trace']
(here/'COMMAND.json').write_text(json.dumps(args,indent=2),encoding='utf-8')
with (here/'v210_console.log').open('w',encoding='utf-8') as log:
 result=subprocess.run(args,cwd=code,stdout=log,stderr=subprocess.STDOUT)
(here/'exit_status.json').write_text(json.dumps({'returncode':result.returncode}),encoding='utf-8')
sys.exit(result.returncode)
