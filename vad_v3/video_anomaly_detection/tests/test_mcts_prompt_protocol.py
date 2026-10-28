"""证据协议测试：提高有效回复率不能以放松完整性或引用检查为代价。"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from boundary_search import LocalBoundarySearch
from mcts_boundary_scorer import BoundaryEvidenceScorer
from mcts_config import MCTSConfig
from mcts_prompts import build_prompt, changed_regions, citation_failures, semantic_failures
from test_mcts_fusion import Client, data, run_refiner


def valid_pair():
    return dict(winner='B', same_event=True, complete_without_loss=True,
                removed_contains_target=False, added_same_event=True,
                evidence_image_indices=[0], removed_evidence_indices=[], added_evidence_indices=[])


def test_half_open_mapping_covers_both_removed_sides_and_added_regions():
    evidence=SimpleNamespace(timestamps=[1.,2.,3.,4.,5.,6.,7.,8.,9.])
    regions=changed_regions((2,8),(3,7),evidence)
    assert [r['eligible_image_indices'] for r in regions]==[[1],[6]]
    response=valid_pair(); response['removed_evidence_indices']=[1]
    assert citation_failures(response,regions)==[
        {'code':'missing_changed_region_citation','region_id':'removed_1'}]
    response['removed_evidence_indices']=[1,6]
    assert not citation_failures(response,regions)
    response['removed_evidence_indices']=[1,6,7]  # t=8 is outside [7,8).
    assert citation_failures(response,regions)[0]['code']=='citation_outside_changed_region'
    added=changed_regions((3,7),(2,8),evidence)
    assert [r['kind'] for r in added]==['added','added']
    response=valid_pair();response['added_evidence_indices']=[1,6]
    assert not citation_failures(response,added)


@pytest.mark.parametrize('proposal_label',['A','B'])
def test_prompt_labels_field_definitions_and_no_filled_answer_example(proposal_label):
    vd=data();sc=BoundaryEvidenceScorer(vd,np.ones(20)*.5,MCTSConfig())
    ev=sc.evidence((2,8),[(2,8),(3,7)],sc.core_indices((2,8)),32)
    prompt=build_prompt('Arrest','reference only',ev,(2,8),(3,7),proposal_label)
    payload=json.loads(next(line for line in prompt.splitlines() if line.startswith('{"target"')))
    assert payload['label_roles']['proposal']==proposal_label
    assert payload['intervals'][proposal_label]==[3,7]
    assert len(payload['changed_regions'])==2
    assert all(r['eligible_image_indices'] for r in payload['changed_regions'])
    assert 'PROPOSAL ONLY' in prompt and 'an empty array does NOT mean no action' in prompt
    assert '"removed_evidence_indices": []' not in prompt
    assert '"complete_without_loss": true' not in prompt


class RepairClient(Client):
    def __init__(self, incomplete=False, keep_missing=False, change_semantics=False):
        super().__init__();self.pair_calls=0;self.incomplete=incomplete
        self.keep_missing=keep_missing;self.change_semantics=change_semantics

    async def generate(self,prompt,images):
        text=await super().generate(prompt,images)
        if 'Compare A and B' not in prompt:return text
        self.pair_calls+=1;response=json.loads(text)
        if self.pair_calls==1 or self.keep_missing:
            response['removed_evidence_indices']=[]
        if self.incomplete:response['complete_without_loss']=False
        if self.pair_calls>1 and self.change_semantics:response['winner']='A'
        return json.dumps(response)


def test_single_citation_repair_can_accept_without_changing_semantics():
    client=RepairClient();out,client,_=run_refiner(client=client)
    item=out.trace['candidates']['0']
    assert out.applied_ids=={0} and client.pair_calls==2
    assert len(item['pairwise_attempts'])==2
    assert item['pairwise_attempts'][0]['citation_failures']
    assert not item['rejection_reasons']
    assert client.calls<=10 and out.trace['logical_requests']==client.calls


def test_semantic_rejection_is_never_reasked_and_all_failures_are_recorded():
    client=RepairClient(incomplete=True);out,client,seeds=run_refiner(client=client)
    item=out.trace['candidates']['0']
    assert client.pair_calls==1 and out.candidates_by_id==seeds
    assert item['reason']=='comparison_rejected'
    assert 'proposal_incomplete' in item['rejection_reasons']
    assert 'missing_changed_region_citation' in item['rejection_reasons']


@pytest.mark.parametrize('kwargs',[
    {'citation_repair_attempts':0},
    {'max_vlm_evaluations_per_candidate':3,'max_vlm_evaluations_per_video':3},
])
def test_citation_repair_respects_disable_and_total_request_budget(kwargs):
    client=RepairClient();out,client,_=run_refiner(client=client,**kwargs)
    assert client.pair_calls==1 and not out.applied_ids
    assert client.calls<=kwargs.get('max_vlm_evaluations_per_candidate',10)


def test_still_missing_citation_rejected_after_one_repair():
    client=RepairClient(keep_missing=True);out,client,_=run_refiner(client=client)
    assert client.pair_calls==2 and not out.applied_ids
    assert out.trace['candidates']['0']['reason']=='missing_changed_region_citation'


def test_repair_cannot_change_semantic_verdict():
    client=RepairClient(change_semantics=True);out,_,seeds=run_refiner(client=client)
    assert not out.applied_ids and out.candidates_by_id==seeds
    assert out.trace['candidates']['0']['reason']=='repair_changed_semantics'


def test_all_semantic_constraints_reported_together():
    response=valid_pair();response.update(winner='A',same_event=False,
        complete_without_loss=False,removed_contains_target=True,added_same_event=False)
    assert len(semantic_failures(response,'B'))==5


def test_optional_expansion_stays_inside_seed_relative_limits():
    vd=data();cfg=MCTSConfig(action_set='shrink_expand_shift',max_depth=2)
    sc=BoundaryEvidenceScorer(vd,np.ones(20)*.5,cfg)
    search=LocalBoundarySearch((2,8),sc,cfg);successors=search.successors(search.root)
    assert any(s<2 and e==8 for s,e in successors)
    assert any(s==2 and e>8 for s,e in successors)
    assert all(search.legal(n.interval) for n in search.enumerate_nodes())
    legacy=LocalBoundarySearch((2,8),sc,MCTSConfig())
    assert all(s>=2 and e<=8 for s,e in legacy.successors(legacy.root))


def test_expansion_can_preserve_event_when_all_shrinks_cut_it():
    class WholeEventClient(Client):
        async def generate(self,prompt,images):
            text=await super().generate(prompt,images)
            if 'Compare A and B' in prompt:return text
            payload=json.loads(next(line for line in prompt.splitlines() if line.startswith('{"target"')))
            start,end=payload['intervals']['evaluated'];response=json.loads(text)
            if start>2 or end<8:
                response['event_completeness']='partial'
                response['start_status']='event_cut' if start>2 else 'well_bounded'
                response['end_status']='event_cut' if end<8 else 'well_bounded'
            return json.dumps(response)
    rejected,_,_=run_refiner(client=WholeEventClient(),max_depth=1)
    assert not rejected.applied_ids
    accepted,_,_=run_refiner(client=WholeEventClient(),max_depth=1,action_set='shrink_expand_shift')
    assert accepted.applied_ids=={0}
    proposal=accepted.candidates_by_id[0]
    assert proposal['t_start']<=2 and proposal['t_end']>=8
    assert proposal['t_start']<2 or proposal['t_end']>8


@pytest.mark.parametrize('value',[-1,2,True])
def test_repair_count_must_be_zero_or_one(value):
    with pytest.raises(ValueError):MCTSConfig(citation_repair_attempts=value)


def test_all_valid_searches_high_contrast_without_removing_evidence_checks():
    import asyncio
    from mcts_refiner import MCTSBoundaryRefiner
    vd=data();curve=np.where((vd['timestamps']>=2)&(vd['timestamps']<8),.9,.1)
    seed={0:{'t_start':2.,'t_end':8.,'coarse_score':.9,'confidence':.8}}
    scorer=SimpleNamespace(get_category_scores=lambda *args:curve)
    outcomes={}
    for policy in ('low_contrast','all_valid'):
        cfg=MCTSConfig(mode='refine',trace_enabled=True,trigger_policy=policy,max_images_per_evaluation=32)
        client=Client()
        outcomes[policy]=asyncio.run(MCTSBoundaryRefiner(cfg,client).refine_candidates(seed,seed,vd,'Arrest',scorer))
        if policy=='low_contrast':assert client.calls==0
        else:assert 0<client.calls<=10
    assert outcomes['low_contrast'].trace['candidates']['0']['reason']=='contrast_not_selected'
    cfg=MCTSConfig(mode='refine',trace_enabled=True,trigger_policy='all_valid',max_images_per_evaluation=2)
    client=Client()
    out=asyncio.run(MCTSBoundaryRefiner(cfg,client).refine_candidates(seed,seed,vd,'Arrest',scorer))
    assert client.calls==0 and out.trace['candidates']['0']['reason']=='image_budget_insufficient'


def test_relaxed_temporal_limits_still_reject_unsupported_gaps():
    from mcts_boundary_scorer import EvidenceError
    cfg=MCTSConfig(max_boundary_gap_seconds=1.5,max_evidence_gap_seconds=3.0)
    vd=data(1.5);sc=BoundaryEvidenceScorer(vd,np.ones(len(vd['timestamps']))*.5,cfg)
    sc.evidence((3,6),[(3,6)],sc.core_indices((3,6)),32)
    strict=BoundaryEvidenceScorer(vd,np.ones(len(vd['timestamps']))*.5,MCTSConfig())
    with pytest.raises(EvidenceError,match='boundary_gap'):strict.evidence((3,6),[(3,6)],[2],32)
    sparse=data(3.0);sc=BoundaryEvidenceScorer(sparse,np.ones(len(sparse['timestamps']))*.5,cfg)
    with pytest.raises(EvidenceError,match='boundary_gap'):sc.evidence((3,6),[(3,6)],[1],32)


@pytest.mark.parametrize('preserves,winner,accepted',[(True,'tie',True),(False,'tie',False),(True,'A',False)])
def test_relative_nonloss_requires_retention_but_allows_partial_tie(preserves,winner,accepted):
    class PartialClient(Client):
        async def generate(self,prompt,images):
            response=json.loads(await super().generate(prompt,images))
            if 'Compare A and B' in prompt:
                response.update(winner=winner,complete_without_loss=False,preserves_reference_content=preserves)
            else:response['event_completeness']='partial'
            return json.dumps(response)
    out,_,_=run_refiner(client=PartialClient(),acceptance_policy='relative_nonloss')
    assert bool(out.applied_ids)==accepted
    strict,_,_=run_refiner(client=PartialClient())
    assert not strict.applied_ids


def test_relative_policy_rejects_complete_to_partial_and_new_cut():
    from mcts_prompts import proposal_eligible
    from test_mcts_fusion import answer
    old=answer();new=answer();new['event_completeness']='partial'
    assert not proposal_eligible(new,old,'relative_nonloss')
    old['event_completeness']='partial'
    assert proposal_eligible(new,old,'relative_nonloss')
    new['start_status']='event_cut'
    assert not proposal_eligible(new,old,'relative_nonloss')
    old['start_status']='event_cut'
    assert proposal_eligible(new,old,'relative_nonloss')


def test_relative_policy_missing_retention_is_not_assumed_true():
    out,_,_=run_refiner(client=Client(),acceptance_policy='relative_nonloss')
    assert not out.applied_ids
    assert out.trace['candidates']['0']['reason']=='invalid_preserves_reference_content'
