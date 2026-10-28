"""Reproducible manifests and per-detect exact-request replay (no shared video state)."""
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import hashlib
import json

# ContextVar按视频异步上下文隔离，避免不同视频或并发任务串用缓存。
request_cache = ContextVar('vad_v3_request_cache', default=None)
request_metrics = ContextVar('vad_v3_request_metrics', default=None)


# 一个detect调用内建立缓存；早退视频无尾流程可运行，配对结果等于该早退结果。
def paired_scope(func):
    @wraps(func)
    async def wrapped(self, *args, **kwargs):
        token=request_cache.set({} if self.config.mcts.paired_eval else None)
        counters={'logical_calls':0,'cache_hits':0,'sdk_attempts':0,'prompt_tokens':0,'completion_tokens':0}
        metric_token=request_metrics.set(counters)
        try:
            result=await func(self,*args,**kwargs)
            if result.mcts_trace is not None:
                result.mcts_trace['video_api_metrics']=dict(counters)
            if self.config.mcts.paired_eval and result.paired_results is None:
                raw=result.to_dict(); raw.pop('mcts_trace',None)
                result.paired_results={'original_off':raw,'skip_only':dict(raw)}
                if self.config.mcts.postprocess_ablation:
                    result.paired_results.update({k:dict(raw) for k in ('legacy_off','legacy_on','dedup_on')})
            return result
        finally:
            request_cache.reset(token)
            request_metrics.reset(metric_token)
    return wrapped


# 只有模型参数、提示词及图像内容完全一致才复用成功回复，减少配对对照中的API随机性。
# 该缓存会使未改变边界的on/off尾流程得到相同回复，这是实验控制，不是复制指标。
def cached_request(func):
    @wraps(func)
    async def wrapped(self, prompt, images=None, **kwargs):
        cache=request_cache.get()
        metrics=request_metrics.get()
        if metrics is not None: metrics['logical_calls']+=1
        if cache is None:
            return await func(self,prompt,images,**kwargs)
        meta=[str(self.client.base_url), self.model,self.temperature,self.max_tokens,self.max_frames,prompt,kwargs]
        digest=hashlib.sha256(json.dumps(meta,sort_keys=True).encode())
        if images is not None:
            digest.update(str((images.shape,str(images.dtype))).encode()); digest.update(images.tobytes())
        key=digest.hexdigest()
        if key not in cache:
            cache[key]=await func(self,prompt,images,**kwargs)
        elif metrics is not None:
            metrics['cache_hits']+=1
        return cache[key]
    return wrapped


# 固化配置、输入列表、模型和源码哈希，阻止不兼容结果被断点续跑混入同一实验。
# 不记录Key，也不为了生成manifest联网下载模型。
def prepare_manifest(path, config, clients, videos, runtime_settings=None):
    path=Path(path)
    source=Path(__file__).parent
    model_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in source.glob('*.pth')}
    weights=source/'prompt_weights.json'
    if weights.exists(): model_files[weights.name]=hashlib.sha256(weights.read_bytes()).hexdigest()
    payload={'schema':'vad-v3-1','config':config.to_dict(),
             'models':[{k:getattr(c,k,None) for k in ('model','temperature','max_tokens','max_frames')}
                       | {'base_url':str(c.client.base_url)} for c in clients],
             'videos':sorted(videos), 'local_model_sha256':model_files,
             'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))}}
    payload['runtime_settings']=runtime_settings or {}
    payload['api_settings']=[getattr(c,'api_settings',{}) for c in clients]
    from importlib.metadata import version
    payload['packages']={n:version(n) for n in ('torch','transformers','open-clip-torch','numpy')}
    # Only record cached revisions; do not download models while preparing metadata.
    from huggingface_hub import scan_cache_dir
    payload['cached_model_revisions']={r.repo_id:sorted(x.commit_hash for x in r.revisions)
        for r in scan_cache_dir().repos if r.repo_type=='model' and
        (r.repo_id.startswith('timm/vit_') or r.repo_id=='OPear/videomae-large-finetuned-UCF-Crime')}
    fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    payload['fingerprint']=fingerprint
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')).get('fingerprint')!=fingerprint:
            raise ValueError('Experiment manifest mismatch; use a new output directory')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return payload
