import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, APIStatusError

import api_recovery as recovery
from llm_usage import track_video_usage, track_batch_usage, tracked_completion, video_usage_all_attempts


def timeout():
    return APITimeoutError(request=httpx.Request('POST','https://example.test'))


def status(code):
    response=httpx.Response(code,request=httpx.Request('POST','https://example.test'))
    return APIStatusError('test',response=response,body={})


@pytest.mark.parametrize('code,retryable',[(400,False),(401,False),(403,False),(404,False),
                                        (408,True),(409,True),(429,True),(500,True),(503,True)])
def test_status_policy(code,retryable):
    assert recovery.is_retryable(status(code))==retryable
    assert not recovery.is_retryable(ValueError('rate limit words are not an API status'))


def test_jitter_retry_succeeds_and_releases_slot(monkeypatch):
    delays=[];calls=0;sem=asyncio.Semaphore(1)
    async def operation():
        nonlocal calls
        calls+=1
        if calls<3:raise timeout()
        return 'ok'
    async def sleep(delay):
        assert not sem.locked()
        delays.append(delay)
    monkeypatch.setattr(recovery.asyncio,'sleep',sleep)
    monkeypatch.setattr(recovery.random,'uniform',lambda low,high:(low+high)/2)
    assert asyncio.run(recovery.retry_api(operation,sem,3,2,8))=='ok'
    assert calls==3 and delays==[5,5]


def test_no_retry_permanent_error_or_cancellation():
    for error in (status(403),asyncio.CancelledError()):
        calls=[]
        async def operation():
            calls.append(1);raise error
        with pytest.raises(type(error)):
            asyncio.run(recovery.retry_api(operation,asyncio.Semaphore(1),3,0,0))
        assert len(calls)==1


def test_swallowed_exhausted_error_marks_video_failed():
    @recovery.ensure_api_success
    async def detect():
        async def operation():raise timeout()
        try:await recovery.retry_api(operation,asyncio.Semaphore(1),2,0,0)
        except Exception:pass  # 模拟MCTS已有的降级回退
        return 'apparently normal'
    with pytest.raises(recovery.VideoAPIError) as error:asyncio.run(detect())
    assert error.value.retryable


def test_failed_gather_drains_siblings():
    async def exercise():
        started=asyncio.Event();finished=[]
        async def sibling():
            started.set()
            try:await asyncio.sleep(3600)
            finally:finished.append(True)
        async def failing():
            await started.wait();raise timeout()
        with pytest.raises(APITimeoutError):await recovery.gather_video_tasks(sibling(),failing())
        assert finished==[True]
    asyncio.run(exercise())


def test_batch_only_retries_transient_failed_video():
    calls=[];records=[]
    async def execute(job,attempt):
        calls.append((job['video_id'],attempt))
        failed=job['video_id'] in ('transient','permanent') and (attempt==1 or job['video_id']=='permanent')
        return dict(status='failed' if failed else 'succeeded',elapsed=1.,
                    retryable=job['video_id']=='transient',error_type='test' if failed else None)
    jobs=[dict(video_id=x) for x in ('success','transient','permanent')]
    results,history=asyncio.run(recovery.run_video_jobs(jobs,execute,1,0,0,records.append))
    assert calls==[('success',1),('transient',1),('permanent',1),('transient',2)]
    assert [r['status'] for r in results]==['succeeded','succeeded','failed']
    assert len(history)==len(records)==4


def test_video_retry_is_bounded():
    async def execute(job,attempt):return dict(status='failed',elapsed=0,retryable=True,error_type='timeout')
    results,history=asyncio.run(recovery.run_video_jobs([dict(video_id='x')],execute,2,0,0))
    assert len(history)==3 and results[0]['status']=='failed'


def test_real_client_attempts_accounted(monkeypatch):
    from llm_client import DashScopeLLMClient
    from experiment import request_metrics
    client=DashScopeLLMClient(api_key='test',api_retry_min_seconds=0,api_retry_max_seconds=0)
    calls=[]
    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls)<3:raise timeout()
        return SimpleNamespace(id='r',usage=SimpleNamespace(prompt_tokens=5,completion_tokens=1,total_tokens=6),
                               choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))])
    monkeypatch.setattr(client.client.chat.completions,'create',create)
    async def exercise():
        metrics=dict(logical_calls=0,cache_hits=0,sdk_attempts=0,prompt_tokens=0,completion_tokens=0)
        token=request_metrics.set(metrics)
        try:
            assert await client.generate('test')=='ok'
            assert metrics['sdk_attempts']==3 and metrics['prompt_tokens']==5
        finally:
            request_metrics.reset(token);await client.client.close()
    asyncio.run(exercise());assert len(calls)==3


def test_all_attempt_usage_includes_failure_and_final_success(tmp_path):
    @track_video_usage
    @recovery.ensure_api_success
    async def detect(video_path,fail):
        async def create(**kwargs):
            if fail:raise timeout()
            return SimpleNamespace(id='r',usage=dict(prompt_tokens=5,completion_tokens=1,total_tokens=6))
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        await recovery.retry_api(lambda:tracked_completion(client,model='m'),asyncio.Semaphore(1),1,0,0)
        return {}
    @track_batch_usage
    async def batch(args):
        with pytest.raises(APITimeoutError):await detect('v',True)
        await detect('v',False)
        return video_usage_all_attempts('v')
    stats=asyncio.run(batch(SimpleNamespace(output_dir=tmp_path)))
    assert stats['diagnosis_attempts']==2 and stats['requests']==2
    assert stats['unknown_usage_requests']==1 and stats['total_tokens']==6
