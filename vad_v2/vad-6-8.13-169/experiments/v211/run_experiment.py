from pathlib import Path
import os,sys,subprocess,json,hashlib
here=Path(__file__).resolve().parent;code=here.parents[1]/'video_anomaly_detection'
for line in (code/'.env').read_text(encoding='utf-8-sig').splitlines():
 if line.startswith('DASHSCOPE_API_KEY='):os.environ['DASHSCOPE_API_KEY']=line.split('=',1)[1].strip().strip('\"\x27')
assert os.environ.get('DASHSCOPE_API_KEY')
os.environ['PYTHONIOENCODING']='utf-8';os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,str(code))
from batch_evaluate import select_videos
selected=select_videos('F:/dataset/ucf_crime',1)
(here/'selected_videos.json').write_text(selected.to_json(orient='records',indent=2),encoding='utf-8')
args=[sys.executable,'-u',str(code/'batch_evaluate.py'),'--dataset_dir','F:/dataset/ucf_crime','--output_dir',str(here/'results'),'--max_per_category','1','--concurrency','1','--clip_model','ViT-L-14','--device','cuda','--detect_model','qwen3-vl-plus','--classify_model','qwen3-vl-plus']
(here/'COMMAND.json').write_text(json.dumps(args,indent=2),encoding='utf-8')
(here/'manifest.json').write_text(json.dumps({'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')},'index':'F:/dataset/ucf_crime/data_parquet/test-00000-of-00001.parquet','index_sha256':hashlib.sha256(Path('F:/dataset/ucf_crime/data_parquet/test-00000-of-00001.parquet').read_bytes()).hexdigest(),'notes':'V2 pipeline. VideoMAE state-key compatibility fix only; original CLIP activation configuration retained. Existing parquet read directly; no index generated.'},indent=2),encoding='utf-8')
with (here/'v211_console.log').open('w',encoding='utf-8') as log:
 result=subprocess.run(args,cwd=code,stdout=log,stderr=subprocess.STDOUT)
(here/'exit_status.json').write_text(json.dumps({'returncode':result.returncode}),encoding='utf-8');sys.exit(result.returncode)
