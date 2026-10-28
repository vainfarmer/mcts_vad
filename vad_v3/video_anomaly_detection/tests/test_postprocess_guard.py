import asyncio
import copy
from types import SimpleNamespace

import pytest
from boundary_guard import BoundaryGuard
from mcts_config import MCTSConfig
from test_mcts_fusion import mock_pipeline, data
from api_recovery import retry_api
from llm_usage import track_video_usage, tracked_completion


def seed(s, e, sid):
    return dict(t_start=s,t_end=e,confidence=.9,anomaly_type='Shoplifting',source_candidate_ids=[sid])


def test_guard_reverts_drift_and_restores_confidence():
    original=seed(10.,20.,0)
    protection={0:dict(accepted_interval=[10.,20.],anchor_timestamp=15.,evidence_bounds=[9.,21.])}
    guard=BoundaryGuard([original],protection,MCTSConfig())
    rows=[dict(original,t_start=10.5,t_end=19.5)]
    assert not guard.apply(rows,'post_curve')
    guard.remember(rows)
    rows=[dict(original,t_start=40.,t_end=42.,confidence=.1)]
    events=guard.apply(rows,'post_cbt')
    assert rows[0]['t_start']==10.5 and rows[0]['confidence']==.9
    assert events[0]['reverted'] and 'lost_curve_anchor' in events[0]['reasons']


def test_guard_does_not_truncate_unprotected_merged_source():
    seeds=[seed(10.,20.,0),seed(19.,30.,1)]
    guard=BoundaryGuard(seeds,{0:dict(accepted_interval=[10.,20.],anchor_timestamp=15.,evidence_bounds=[9.,21.])},MCTSConfig())
    merged=dict(seeds[0],t_end=30.,source_candidate_ids=[0,1]);guard.remember([merged])
    rows=[dict(merged,t_end=29.)]
    assert not guard.apply(rows,'post_curve')
    rows=[dict(merged,t_start=22.)]
    assert guard.apply(rows,'post_cbt')[0]['reverted']
    assert rows[0]['t_end']==30.


def test_final_nms_after_cbt_preserves_auxiliary_alignment():
    vad=mock_pipeline();vad.config.mcts.final_nms_enabled=True;vad.config.scoring.cbt_enabled=True
    seeds=[seed(2.,4.,0),seed(8.,10.,1)]
    def cbt(rows,*args):
        return [dict(a,t_start=5.,t_end=7.,confidence=.8+i*.1) for i,a in enumerate(rows)]
    vad._cbt_category_boundary_tighten=cbt
    result=asyncio.run(vad._finish_from_stage5(seeds,data(),'x','Shoplifting',.9,set(),set(),{}))
    assert len(result.anomalies)==1
    assert result.pre_cbt_boundaries==[[8.,10.]]
    assert len(result.mcts_trace['final_nms_suppression'])==1


def test_full_tail_guard_can_be_disabled():
    vad=mock_pipeline();vad.config.scoring.cbt_enabled=True
    vad._cbt_category_boundary_tighten=lambda rows,*a:[dict(x,t_start=12.,t_end=14.) for x in rows]
    protections={0:dict(accepted_interval=[2.,8.],anchor_timestamp=5.,evidence_bounds=[1.,9.])}
    async def run(enabled):
        return await vad._finish_from_stage5([seed(2.,8.,0)],data(),'x','Shoplifting',.9,set(),set(),{},
                                            guard_enabled=enabled,protections=protections)
    off=asyncio.run(run(False));on=asyncio.run(run(True))
    assert off.anomalies[0].t_start==12. and on.anomalies[0].t_start==2.
    assert on.mcts_trace['boundary_guard_events'][0]['stage']=='post_cbt'


def test_ablation_arms_are_independent_and_early_return_complete():
    for early in (False,True):
        vad=mock_pipeline('refine',True)
        cfg=vad.config.mcts
        cfg.postprocess_ablation=cfg.final_nms_enabled=cfg.boundary_guard_enabled=True
        if early:
            vad.videomae.classify=lambda *a,**kw:{'category':'Normal','confidence':.9,'top_k':[('Normal',.9)]}
        result=asyncio.run(vad.detect('x'))
        assert {'legacy_off','legacy_on','dedup_on','original_off','skip_only'}==set(result.paired_results)
        assert result.paired_results['legacy_off'] is not result.paired_results['legacy_on']
        if not early:
            assert result.paired_results['legacy_on']['mcts_trace']['postprocess_config']==dict(final_nms=False,boundary_guard=False)
            assert result.mcts_trace['postprocess_config']==dict(final_nms=True,boundary_guard=True)


def test_queue_time_is_separate_from_actual_request_time():
    async def exercise():
        sem=asyncio.Semaphore(1)
        await sem.acquire()
        async def create(**kwargs):
            return SimpleNamespace(id='r',usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2))
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        @track_video_usage
        async def detect(video_path):
            await retry_api(lambda:tracked_completion(client,model='mock'),sem)
            return {}
        task=asyncio.create_task(detect('x'))
        await asyncio.sleep(.02)
        sem.release()
        result=await task
        usage=result['token_usage']
        assert usage['requests']==1 and usage['api_queue_seconds']>=.01
        assert usage['api_request_seconds']<usage['api_queue_seconds']
    asyncio.run(exercise())
