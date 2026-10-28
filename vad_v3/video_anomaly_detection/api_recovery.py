"""有界API重试、同视频并行请求清理，以及防止失败被降级逻辑吞掉。"""
import asyncio
from contextvars import ContextVar
from functools import wraps
import math
import random
import time

from openai import APIConnectionError, APITimeoutError, APIStatusError

try:
    from . import vad_logger
    from .llm_usage import api_queue_wait, record_wait
except ImportError:
    import vad_logger
    from llm_usage import api_queue_wait, record_wait

log = vad_logger.get('vad.retry')
_terminal_errors = ContextVar('vad_terminal_api_errors', default=None)


class VideoAPIError(RuntimeError):
    def __init__(self, errors):
        self.retryable = all(is_retryable(e) for e in errors)
        super().__init__('Video contains exhausted API failures: ' +
                         ', '.join(type(e).__name__ for e in errors))


def is_retryable(error):
    if isinstance(error, VideoAPIError):
        return error.retryable
    if isinstance(error, (APITimeoutError, APIConnectionError)):
        return True
    return isinstance(error, APIStatusError) and (
        error.status_code in (408, 409, 429) or 500 <= error.status_code < 600)


def validate_delay(low, high):
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high <= 60):
        raise ValueError('Retry delay must satisfy 0 <= min <= max <= 60 seconds')


async def retry_api(operation, semaphore, attempts=3, delay_min=2., delay_max=8., model='unknown'):
    """每次实际调用都独立计token；随机等待时释放API并发槽，不阻塞其他视频。"""
    if type(attempts) is not int or attempts < 1:
        raise ValueError('API attempts must be a positive integer')
    validate_delay(delay_min, delay_max)
    for attempt in range(1, attempts+1):
        try:
            queued = time.perf_counter()
            try:
                await semaphore.acquire()
            except asyncio.CancelledError:
                wait = time.perf_counter() - queued
                record_wait(model, 'api_cancelled_queue_seconds', wait)
                raise
            wait = time.perf_counter() - queued
            token = api_queue_wait.set(wait)
            try:
                return await operation()
            finally:
                api_queue_wait.reset(token)
                semaphore.release()
        except Exception as error:
            # CancelledError不属于Exception，取消/单候选超时不能被重试吞掉。
            if not is_retryable(error) or attempt == attempts:
                errors = _terminal_errors.get()
                if errors is not None:
                    errors.append(error)
                raise
            delay = random.uniform(delay_min, delay_max)
            log.warning('[API-RETRY] model=%s failed_attempt=%s/%s error=%s status=%s wait=%.2fs',
                        model, attempt, attempts, type(error).__name__,
                        getattr(error, 'status_code', None), delay)
            started = time.perf_counter()
            try:
                await asyncio.sleep(delay)
            finally:
                record_wait(model, 'api_backoff_seconds', time.perf_counter()-started)


def ensure_api_success(func):
    """MCTS/分类内部可能捕获API异常并回退；这类不完整诊断不能写成成功结果。"""
    @wraps(func)
    async def wrapped(*args, **kwargs):
        failures = []
        token = _terminal_errors.set(failures)
        try:
            result = await func(*args, **kwargs)
            if failures:
                raise VideoAPIError(failures)
            return result
        finally:
            _terminal_errors.reset(token)
    return wrapped


async def gather_video_tasks(*awaitables):
    """任一片段失败后先取消并回收同组任务，防止重诊断与旧请求重叠及用量串账。"""
    tasks = [asyncio.ensure_future(a) for a in awaitables]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def add_api_arguments(parser):
    parser.add_argument('--api_retry_attempts', type=int, default=3,
                        help='Total SDK attempts per logical request, including first attempt')
    parser.add_argument('--api_retry_min_seconds', type=float, default=2.)
    parser.add_argument('--api_retry_max_seconds', type=float, default=8.)
    parser.add_argument('--api_concurrency', type=int, default=4,
                        help='Shared limit across detection/classification API requests')
    parser.add_argument('--api_timeout_seconds', type=float, default=60.)
    parser.add_argument('--api_connect_timeout_seconds', type=float, default=15.)


def client_retry_options(args):
    return dict(api_retry_attempts=getattr(args,'api_retry_attempts',3),
                api_retry_min_seconds=getattr(args,'api_retry_min_seconds',2.),
                api_retry_max_seconds=getattr(args,'api_retry_max_seconds',8.),
                api_concurrency=getattr(args,'api_concurrency',4),
                api_timeout_seconds=getattr(args,'api_timeout_seconds',60.),
                api_connect_timeout_seconds=getattr(args,'api_connect_timeout_seconds',15.))


async def run_video_jobs(jobs, execute, rounds=1, delay_min=5., delay_max=15., record=None):
    """首轮全部结束后只重跑可恢复失败；成功/跳过及永久错误不会再发请求。"""
    if type(rounds) is not int or rounds < 0:
        raise ValueError('video_retry_rounds must be a nonnegative integer')
    validate_delay(delay_min,delay_max)
    latest = {}
    history = []
    async def one(index, attempt):
        if attempt > 1:
            delay=random.uniform(delay_min,delay_max)
            log.warning('[VIDEO-RETRY] video=%s attempt=%s wait=%.2fs',jobs[index]['video_id'],attempt,delay)
            await asyncio.sleep(delay)
        result=await execute(jobs[index],attempt)
        latest[index]=result
        entry=dict(result,video_id=jobs[index]['video_id'],attempt=attempt)
        history.append(entry)
        if record is not None:
            record(entry)
    pending=list(range(len(jobs)))
    for attempt in range(1,rounds+2):
        await gather_video_tasks(*(one(index,attempt) for index in pending))
        pending=[index for index in pending if latest[index]['status']=='failed' and latest[index]['retryable']]
        if not pending:
            break
    return [latest[i] for i in range(len(jobs))], history
