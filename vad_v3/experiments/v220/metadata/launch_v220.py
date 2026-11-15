"""Local background launcher. Contains no credentials; never prints their values."""
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import sys

PROJECT=Path(__file__).resolve().parents[4]
OUTPUT=Path(__file__).resolve().parents[1]
os.chdir(PROJECT)
status='failed';error_type=None
try:
    if not os.environ.get('DASHSCOPE_API_KEY'):
        # Reuse the user's existing local configuration without executing it.
        tree=ast.parse((PROJECT/'test.py').read_text(encoding='utf-8-sig'))
        credentials={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant)
                     and isinstance(n.value,str) and n.value.startswith('sk-') and len(n.value)>20}
        if len(credentials)!=1:raise RuntimeError('Set DASHSCOPE_API_KEY before running')
        os.environ['DASHSCOPE_API_KEY']=credentials.pop()
        del credentials,tree
    entry=PROJECT/'vad_v3/video_anomaly_detection/xd_full_ablation.py'
    sys.argv=[str(entry),'--dataset_dir','F:/dataset/xd-violence','--output_dir',str(OUTPUT),
              '--config',str(PROJECT/'vad_v3/experiments/v219/metadata/config.json'),
              '--expected_videos','800','--concurrency','4','--api_concurrency','8',
              '--detect_model','qwen3-vl-plus','--classify_model','qwen3-vl-plus']
    runpy.run_path(str(entry),run_name='__main__')
    status='completed'
except BaseException as error:
    error_type=type(error).__name__
    raise
finally:
    (OUTPUT/'metadata/process_exit.json').write_text(json.dumps(dict(status=status,error_type=error_type,
        finished_at_utc=datetime.now(timezone.utc).isoformat()),indent=2),encoding='utf-8')
