"""Actual API usage, isolated by asynchronous video and batch contexts."""
import asyncio
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import inspect
import json
from pathlib import Path
import time
import uuid

try:
    from . import vad_logger
except ImportError:
    import vad_logger

logger = vad_logger.get('vad.tokens')
_video = ContextVar('llm_usage_video', default=None)
_batch = ContextVar('llm_usage_batch', default=None)


def counter():
    return dict(requests=0, successful_requests=0, failed_requests=0,
                unknown_usage_requests=0, prompt_tokens=0, completion_tokens=0,
                total_tokens=0, by_model={})


def add(stats, event, group=True):
    stats['requests'] += 1
    stats['successful_requests' if event['status'] == 'success' else 'failed_requests'] += 1
    stats['unknown_usage_requests'] += int(not event['usage_complete'])
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        stats[key] += event[key] or 0
    if group:
        add(stats['by_model'].setdefault(event['model'], counter()), event, False)


def field(value, key):
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def emit(kind, event):
    logger.info('[LLM-TOKENS-%s] %s', kind, json.dumps(event, ensure_ascii=False))


def append_event(path, event):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')


async def tracked_completion(client, **kwargs):
    """One SDK attempt; no prompt, image, API key or error text is logged."""
    video, batch = _video.get(), _batch.get()
    event = dict(request_id=uuid.uuid4().hex, timestamp=datetime.now(timezone.utc).isoformat(),
                 model=kwargs.get('model', 'unknown'), video_path=video['video_path'] if video else None,
                 run_id=batch['run_id'] if batch else None, status='error',
                 prompt_tokens=None, completion_tokens=None, total_tokens=None, usage_complete=False)
    started = time.perf_counter()
    try:
        response = await client.chat.completions.create(**kwargs)
        event['status'] = 'success'
        event['provider_request_id'] = field(response, 'id')
        usage = field(response, 'usage')
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            value = field(usage, key)
            event[key] = value if isinstance(value, int) and value >= 0 else None
        event['usage_complete'] = all(event[k] is not None for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
        # Detail counts are subsets of the totals, never added again.
        event['cached_tokens'] = field(field(usage, 'prompt_tokens_details'), 'cached_tokens')
        event['reasoning_tokens'] = field(field(usage, 'completion_tokens_details'), 'reasoning_tokens')
        return response
    except BaseException as exc:
        event['status'] = 'cancelled' if isinstance(exc, asyncio.CancelledError) else 'error'
        event['error_type'] = type(exc).__name__
        raise
    finally:
        event['elapsed_seconds'] = round(time.perf_counter() - started, 4)
        if video is not None:
            add(video, event)
        if batch is not None:
            add(batch['usage'], event)
        emit('CALL', event)
        if batch is not None:
            append_event(batch['calls_path'], event)


def track_video_usage(func):
    @wraps(func)
    async def wrapped(*args, **kwargs):
        bound = inspect.signature(func).bind(*args, **kwargs)
        stats = counter()
        stats.update(video_path=str(bound.arguments.get('video_path', 'unknown')), status='error')
        token = _video.set(stats)
        try:
            result = await func(*args, **kwargs)
            stats['status'] = 'success'
            if isinstance(result, dict):
                result['token_usage'] = stats
            else:
                result.token_usage = stats
            return result
        finally:
            _video.reset(token)
            emit('VIDEO', stats)
            batch = _batch.get()
            if batch is not None:
                batch['videos'].append(stats)
    return wrapped


def track_batch_usage(func):
    @wraps(func)
    async def wrapped(args, *rest, **kwargs):
        directory = Path(args.output_dir) / 'metadata'
        directory.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
        batch = dict(run_id=run_id, usage=counter(), videos=[],
                     calls_path=directory / f'llm_calls_{run_id}.jsonl')
        batch['calls_path'].touch()
        token = _batch.set(batch)
        status = 'error'
        try:
            result = await func(args, *rest, **kwargs)
            status = 'completed'
            return result
        finally:
            _batch.reset(token)
            summary = dict(run_id=run_id, status=status, scope='current_execution_only',
                           usage=batch['usage'], videos=batch['videos'],
                           calls_file=batch['calls_path'].name,
                           note='Known API usage only; unknown_usage_requests may incur unreported usage. Skipped results are excluded.')
            emit('BATCH', summary)
            payload = json.dumps(summary, ensure_ascii=False, indent=2)
            (directory / f'token_usage_{run_id}.json').write_text(payload, encoding='utf-8')
            (directory / 'token_usage.json').write_text(payload, encoding='utf-8')
    return wrapped
