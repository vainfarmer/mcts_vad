"""Checks run outside the web process so CUDA DLL errors do not stop the UI."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .settings import redact, ROOT


def check_environment():
    packages={}
    for package in ('Flask','waitress','python-dotenv','psutil','imageio-ffmpeg',
                    'torch','torchvision','open_clip_torch','transformers','numpy',
                    'pandas','scikit-learn','scipy','openai','safetensors'):
        try: packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: packages[package]=None
    result=dict(checked_at=time.time(),python=sys.executable,python_version=sys.version.split()[0],
                packages=packages,api_key_configured=bool(os.getenv('DASHSCOPE_API_KEY')))
    command=[sys.executable,'-c',
        "import torch,json;print(json.dumps({'available':torch.cuda.is_available(),"
        "'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'torch':torch.__version__,'cuda':torch.version.cuda}))"]
    try:
        proc=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',
            timeout=45,env=dict(os.environ,PYTHONUTF8='1'),cwd=ROOT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        result['cuda']=json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode==0 else dict(
            available=False,error=redact(proc.stderr[-3000:]))
    except Exception as exc:
        result['cuda']=dict(available=False,error=redact(str(exc)))
    try:
        import imageio_ffmpeg
        executable=imageio_ffmpeg.get_ffmpeg_exe()
        version=subprocess.run([executable,'-version'],capture_output=True,text=True,timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        result['ffmpeg']=dict(available=version.returncode==0,version=version.stdout.splitlines()[0])
    except Exception as exc:
        result['ffmpeg']=dict(available=False,error=str(exc))
    from huggingface_hub import try_to_load_from_cache
    assets={}
    for name,repo,filename in (
        ('videomae_config','OPear/videomae-large-finetuned-UCF-Crime','config.json'),
        ('videomae','OPear/videomae-large-finetuned-UCF-Crime','model.safetensors'),
        ('clip','timm/vit_large_patch14_clip_224.openai','open_clip_model.safetensors')):
        cached=try_to_load_from_cache(repo,filename)
        path=Path(cached) if isinstance(cached,str) else None
        assets[name]=dict(cached=bool(path and path.is_file()),
            bytes=path.stat().st_size if path and path.is_file() else None)
    mil=ROOT/'vad_v3/video_anomaly_detection/model_mil.pth'
    assets['mil']=dict(cached=mil.exists(),bytes=mil.stat().st_size if mil.exists() else None)
    result['model_assets']=assets
    result['ready_for_cuda']=result['cuda'].get('available',False) and result['api_key_configured']
    return result
