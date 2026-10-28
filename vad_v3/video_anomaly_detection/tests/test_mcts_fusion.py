import asyncio
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest

from mcts_config import MCTSConfig, add_mcts_arguments
from mcts_boundary_scorer import BoundaryEvidenceScorer, EvidenceError, parse_answer
from boundary_search import LocalBoundarySearch, Node
from mcts_refiner import MCTSBoundaryRefiner
from config import VADConfig
from pipeline import MultiAnomalyVAD
from post_processor import global_merge, temporal_nms


def data(step=.5):
    t=np.arange(0,10,step)
    return {'timestamps':t,'duration':10.,'all_frames':np.zeros((len(t),4,4,3),np.uint8), 'video_path':'mock.mp4'}


def answer():
    return {'target_present_inside':True,'target_confidence':.9,'same_event_as_reference':True,
            'start_status':'well_bounded','end_status':'well_bounded',
            'event_completeness':'complete','evidence_image_indices':[0]}


class Client:
    max_frames=32
    def __init__(self,fail=False): self.calls=0; self.fail=fail
    async def generate(self,prompt,images):
        self.calls+=1
        if self.fail: raise RuntimeError('mock network failure')
        if 'Compare A and B' in prompt:
            payload=json.loads(next(line for line in prompt.splitlines() if line.startswith('{"target"')))
            regions=payload['changed_regions']
            return json.dumps({'winner':'B','same_event':True,'complete_without_loss':True,
                'removed_contains_target':False,'added_same_event':True,
                'evidence_image_indices':[0], 'removed_evidence_indices':sorted({i for r in regions if r['kind']=='removed' for i in r['eligible_image_indices']}),
                'added_evidence_indices':sorted({i for r in regions if r['kind']=='added' for i in r['eligible_image_indices']})})
        return json.dumps(answer())


def run_refiner(mode='refine',client=None,**kwargs):
    c=MCTSConfig(mode=mode,trace_enabled=True,max_images_per_evaluation=32,**kwargs)
    vd=data(); curve=np.ones(len(vd['timestamps']))*.5
    seeds={0:{'t_start':2.,'t_end':8.,'confidence':.8,'coarse_score':.9,'anomaly_type':'Arrest',
              'boundary_contrast':.1,'frame_indices':[1,2]}}
    original=copy.deepcopy(seeds); cli=client or Client()
    fs=SimpleNamespace(get_category_scores=lambda d,c:curve)
    result=asyncio.run(MCTSBoundaryRefiner(c,cli).refine_candidates(seeds,seeds,vd,'Arrest',fs))
    assert seeds==original
    return result,cli,original


@pytest.mark.parametrize('strategy',['mcts','greedy','uniform'])
def test_accept_preserves_score_and_budget(strategy):
    out,cli,seeds=run_refiner(search_strategy=strategy)
    assert out.applied_ids=={0}
    a=out.candidates_by_id[0]
    assert a['confidence']==seeds[0]['confidence'] and a['coarse_score']==seeds[0]['coarse_score']
    assert 'boundary_contrast' not in a and 'frame_indices' not in a
    assert a['t_end']-a['t_start']<6
    assert cli.calls<=10 and out.trace['logical_requests']==cli.calls


def test_shadow_and_off_preserve_payload():
    shadow,c,seeds=run_refiner('shadow')
    assert shadow.candidates_by_id==seeds and not shadow.applied_ids
    assert shadow.trace['candidates']['0']['would_accept'] and c.calls>0
    off,c,seeds=run_refiner('off')
    assert off.candidates_by_id==seeds and c.calls==0 and off.trace is None


def test_exception_budget_and_gain_fallback():
    out,c,seeds=run_refiner(client=Client(fail=True))
    assert out.candidates_by_id==seeds and not out.applied_ids
    out,c,seeds=run_refiner(max_vlm_evaluations_per_candidate=2)
    assert not out.applied_ids and c.calls<=2
    out,c,seeds=run_refiner(comparison_gate='reward_gain',min_reward_gain=1.)
    assert not out.applied_ids and out.trace['candidates']['0']['reason']=='reward_gain_gate'


def test_timeout_and_cancellation():
    class Slow(Client):
        async def generate(self,*args,**kwargs): await asyncio.sleep(10)
    out,c,seeds=run_refiner(client=Slow(),timeout_seconds_per_candidate=.01)
    assert out.candidates_by_id==seeds and out.trace['candidates']['0']['reason']=='timeout'
    class Cancel(Client):
        async def generate(self,*args,**kwargs): raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError): run_refiner(client=Cancel())


def test_sparse_boundary_and_large_changed_region():
    vd=data(2); c=MCTSConfig()
    sc=BoundaryEvidenceScorer(vd,np.ones(5)*.5,c)
    with pytest.raises(EvidenceError,match='boundary_gap'): sc.evidence((2,8),[(2,8)],[1],16)
    vd=data(.5); sc=BoundaryEvidenceScorer(vd,np.ones(20)*.5,c)
    with pytest.raises(EvidenceError,match='image_budget'): sc.evidence((2,8),[(2,8),(3,7)],[4],2)


def test_weighted_mean_and_geometric_limits():
    vd=data(); sc=BoundaryEvidenceScorer(vd,vd['timestamps']/10,MCTSConfig())
    assert sc.mean(2,8)==pytest.approx(.5)
    search=LocalBoundarySearch((2,8),sc,MCTSConfig(action_set='shrink_shift'))
    nodes=search.enumerate_nodes()
    assert all(search.legal(n.interval) and n.interval[1]-n.interval[0]>=3 for n in nodes)
    child=Node((2.5,8),1,search.root)
    assert search.seed not in search.successors(child)
    assert search.successors(Node((2,8),4))==[]


@pytest.mark.parametrize('field,value',[('target_present_inside','true'),('target_confidence',float('nan')),
    ('start_status','yes'),('evidence_image_indices',[-1])])
def test_strict_response(field,value):
    a=answer(); a[field]=value
    with pytest.raises(ValueError): parse_answer(json.dumps(a),2)


def test_config_all_cli_fields_and_validation():
    import argparse
    from dataclasses import fields,asdict
    p=argparse.ArgumentParser(); add_mcts_arguments(p)
    args=p.parse_args(['--mcts_mode','refine','--mcts_search_strategy','uniform','--mcts_trace'])
    c=MCTSConfig.from_args(args)
    assert c.mode=='refine' and c.search_strategy=='uniform' and c.trace_enabled
    assert set(asdict(c))=={f.name for f in fields(MCTSConfig)}
    with pytest.raises(ValueError): MCTSConfig(max_outward_shift_ratio=.9)
    with pytest.raises(ValueError): MCTSConfig(timeout_seconds_per_candidate=float('nan'))


def test_source_ids_merge_and_nms():
    a={'t_start':0.,'t_end':4.,'confidence':.8,'anomaly_type':'A','source_candidate_ids':[0]}
    b=dict(a,t_start=5.,t_end=8.,source_candidate_ids=[1])
    merged=global_merge([a,b]); assert merged[0]['source_candidate_ids']==[0,1]
    suppressed=dict(a,confidence=.7,source_candidate_ids=[2]); audit=[]
    out=temporal_nms([a,suppressed],audit=audit)
    assert out[0]['source_candidate_ids']==[0] and audit[0]['suppressed']==[2]


def test_full_tail_uses_frozen_stage5b_and_trace():
    vad=MultiAnomalyVAD.__new__(MultiAnomalyVAD); vad.config=VADConfig()
    vad.config.scoring.qscore_rerank=False; vad.config.scoring.bue_enabled=False; vad.config.scoring.cbt_enabled=False
    calls=[]
    async def boundary(a,*args): calls.append(a['t_start']); return dict(a,t_start=a['t_start']+.1)
    async def classify(*args): return 'Shoplifting'
    vad._vlm_boundary_estimation=boundary; vad._ensemble_classify=classify
    vad._post_merge_boundary_tightening=lambda a,*args:a
    vad.frame_scorer=SimpleNamespace(clip_score_curve_correction=lambda a,*args,**kw:a,compute_frame_scores=lambda *a:None)
    seeds=[{'t_start':2.,'t_end':4.,'confidence':.9,'anomaly_type':'Shoplifting','source_candidate_ids':[0]}]
    trace={}
    result=asyncio.run(vad._finish_from_stage5(copy.deepcopy(seeds),data(),'x','Shoplifting',.9,{0},{0},trace))
    assert not calls and result.anomalies[0].t_start==2.
    assert trace['stages']['final'][0]['source_candidate_ids']==[0]
    result=asyncio.run(vad._finish_from_stage5(copy.deepcopy(seeds),data(),'x','Shoplifting',.9,{0},set()))
    assert calls==[2.] and result.anomalies[0].t_start==2.1


def mock_pipeline(mode='off', paired=False):
    vad=MultiAnomalyVAD.__new__(MultiAnomalyVAD)
    vad.config=VADConfig(); cfg=vad.config
    cfg.mcts=MCTSConfig(mode=mode,trace_enabled=True,paired_eval=paired,max_images_per_evaluation=32)
    cfg.scoring.videomae_weight=0; cfg.scoring.qscore_rerank=False
    cfg.scoring.bue_enabled=False; cfg.scoring.cbt_enabled=False
    vd=data(); client=Client(); vad.llm_client=client; vad.classify_llm_client=client
    vad.preprocessor=SimpleNamespace(process=lambda path:copy.deepcopy(vd))
    vad.anomaly_expert=SimpleNamespace(model=None,clip_call_count=0)
    vad.videomae=SimpleNamespace(model=object(),classify=lambda *a,**k:{'category':'Shoplifting','confidence':.9,'top_k':[('Shoplifting',.9)]})
    vad._compute_motion_energy=lambda vd:np.zeros(len(vd['timestamps']))
    vad._clip_temporal_localization=lambda *args:[]
    seed={'t_start':2.,'t_end':8.,'confidence':.8,'coarse_score':.9,'anomaly_type':'Shoplifting','boundary_contrast':.1}
    async def discover(*args): return [copy.deepcopy(seed)]
    vad.discovery=SimpleNamespace(discover=discover,scan_segments_count=0)
    vad._clip_boundary_detection=lambda *args:[copy.deepcopy(seed)]
    vad.frame_sampler=None; vad.mil_model=None
    async def boundary(a,*args): return dict(a,t_start=a['t_start']+.1)
    async def classify(*args): return 'Shoplifting'
    vad._vlm_boundary_estimation=boundary; vad._ensemble_classify=classify
    vad._post_merge_boundary_tightening=lambda a,*args:a
    vad.frame_scorer=SimpleNamespace(get_category_scores=lambda vd,c:np.ones(len(vd['timestamps']))*.5,
        clip_score_curve_correction=lambda a,*args,**kw:a,compute_frame_scores=lambda *args:None)
    return vad


def test_actual_pipeline_off_shadow_paired_branches():
    off=asyncio.run(mock_pipeline().detect('x')).to_dict()
    shadow=asyncio.run(mock_pipeline('shadow').detect('x')).to_dict()
    shadow.pop('mcts_trace')
    assert off==shadow
    on=asyncio.run(mock_pipeline('refine',True).detect('x')).to_dict()
    off.pop('token_usage')  # Paired arms share actual requests; usage belongs to the whole video.
    assert on['paired_results']['original_off']==off
    assert on['paired_results']['skip_only']['anomalies'][0]['t_start']==2.
    assert on['mcts_trace']['candidates']['0']['applied']


def test_paired_early_return_and_concurrent_video_isolation():
    vad=mock_pipeline('refine',True)
    vad.videomae.classify=lambda *a,**kw:{'category':'Normal','confidence':.9,'top_k':[('Normal',.9)]}
    result=asyncio.run(vad.detect('x')).to_dict()
    assert result['paired_results']['original_off']['is_normal']
    assert result['mcts_trace']['skipped_reason']=='normal_early_return'
    async def simultaneous():
        shared=mock_pipeline('refine',True)
        return await asyncio.gather(shared.detect('a'),shared.detect('b'))
    a,b=asyncio.run(simultaneous())
    assert a.mcts_trace is not b.mcts_trace
    assert a.mcts_trace['logical_requests']==b.mcts_trace['logical_requests']<=10


def test_video_budget_and_candidate_reordering():
    cfg=MCTSConfig(mode='refine',trace_enabled=True,max_images_per_evaluation=32,
                   max_vlm_evaluations_per_video=5,max_candidates_per_video=2)
    vd=data(); seeds={7:{'t_start':2.,'t_end':8.,'coarse_score':.9,'confidence':.8},
                      2:{'t_start':1.,'t_end':7.,'coarse_score':.6,'confidence':.5}}
    fs=SimpleNamespace(get_category_scores=lambda *a:np.ones(20)*.5); cli=Client()
    out=asyncio.run(MCTSBoundaryRefiner(cfg,cli).refine_candidates(seeds,seeds,vd,'Arrest',fs))
    assert set(out.candidates_by_id)=={7,2} and cli.calls<=5
    assert out.candidates_by_id[7]['confidence']==.8 and out.candidates_by_id[2]['confidence']==.5


def test_videomae_failure_cannot_leave_initialized_model(monkeypatch):
    from videomae_classifier import VideoMAEClassifier
    holder=[]
    def fail(self,*args):
        self.model=object(); holder.append(self); raise ValueError('bad checkpoint')
    monkeypatch.setattr(VideoMAEClassifier,'_load_model',fail)
    with pytest.raises(RuntimeError): VideoMAEClassifier()
    assert holder[0].model is None and not holder[0].loaded


def test_off_tail_equivalent_to_v2_source():
    # Execute the unmodified v2 tail with the same fixed mocks, independently of v3 wiring.
    from pathlib import Path
    import pipeline
    original=Path(__file__).resolve().parents[3]/'vad_v2/vad-6-8.13-169/video_anomaly_detection/pipeline.py'
    text=original.read_text(encoding='utf-8')
    start=text.index('        # Stage 5b: VLM boundary estimation')
    end=text.index('    # ================================================================\n    # v27-RIG',start)
    body=text[start:end]
    code='async def original_tail(self,all_anomalies,video_data,video_path,videomae_category,videomae_confidence):\n'+''.join(line[4:] if line.startswith('    ') else line for line in body.splitlines(True))
    namespace=dict(vars(pipeline)); exec(compile(code,'v2_original_tail','exec'),namespace)
    seeds=[{'t_start':2.,'t_end':8.,'confidence':.8,'coarse_score':.9,'anomaly_type':'Shoplifting','boundary_contrast':.1}]
    vad=mock_pipeline()
    a=asyncio.run(namespace['original_tail'](vad,copy.deepcopy(seeds),data(),'x','Shoplifting',.9))
    b=asyncio.run(vad._finish_from_stage5(copy.deepcopy(seeds),data(),'x','Shoplifting',.9,{0},set()))
    assert a.to_dict()==b.to_dict()


def test_exact_request_cache_scope():
    from experiment import cached_request,request_cache
    class Fake:
        client=SimpleNamespace(base_url='mock'); model='m'; temperature=0; max_tokens=512; max_frames=16
        def __init__(self): self.calls=0
        @cached_request
        async def generate(self,prompt,images=None,**kwargs): self.calls+=1; return prompt
    async def exercise():
        f=Fake(); token=request_cache.set({})
        try:
            await f.generate('x',np.zeros((1,2,2,3))); await f.generate('x',np.zeros((1,2,2,3)))
            assert f.calls==1
            await f.generate('x',np.ones((1,2,2,3))); assert f.calls==2
        finally: request_cache.reset(token)
        await f.generate('x'); assert f.calls==3
    asyncio.run(exercise())
