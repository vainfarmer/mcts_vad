"""Per-video refinement orchestration. Search scores never replace detection scores."""
import asyncio
import copy
from dataclasses import dataclass
import hashlib
import json
import time
import numpy as np

try:
    from .boundary_search import LocalBoundarySearch
    from .mcts_boundary_scorer import BoundaryEvidenceScorer, EvidenceError, parse_answer, interval_difference
except ImportError:
    from boundary_search import LocalBoundarySearch
    from mcts_boundary_scorer import BoundaryEvidenceScorer, EvidenceError, parse_answer, interval_difference

try:
    from .mcts_prompts import (PROMPT_VERSION, SEMANTIC_FIELDS, build_prompt,
                               changed_regions, semantic_failures, citation_failures, repair_prompt, proposal_eligible)
    from . import vad_logger
except ImportError:
    from mcts_prompts import (PROMPT_VERSION, SEMANTIC_FIELDS, build_prompt,
                              changed_regions, semantic_failures, citation_failures, repair_prompt, proposal_eligible)
    import vad_logger

_log = vad_logger.get('vad.mcts')


@dataclass
class RefinementOutcome:
    candidates_by_id: dict
    applied_ids: set
    trace: object


class MCTSBoundaryRefiner:
    def __init__(self, config, llm_client, frame_sampler=None, category_priors=None):
        self.cfg=config; self.client=llm_client; self.priors=category_priors or {}
        try:
            from .token_compressor import SemanticFrameSampler
        except ImportError:
            from token_compressor import SemanticFrameSampler
        self.select_evidence = SemanticFrameSampler.select_boundary_evidence

    # 输入是 Stage 5 CLIP 边界及原始候选，输出始终保留 candidate_id 映射。
    # 深拷贝保证任何失败都能回退；本模块只修改边界，不用搜索奖励替换检测置信度。
    async def refine_candidates(self, seeds_by_id, source_candidates_by_id, video_data, category, frame_scorer):
        cfg=self.cfg
        output=copy.deepcopy(seeds_by_id); applied=set()
        trace={'mode':cfg.mode,'acceptance_policy':cfg.acceptance_policy,'strategy':cfg.search_strategy,'prompt_version':PROMPT_VERSION,
               'candidates':{},'logical_requests':0,'images_sent':0,'frame_scores_stage':'post_conf_gate'}
        if cfg.mode=='off': return RefinementOutcome(output,applied,None)
        cfg.__post_init__()
        remaining=cfg.max_vlm_evaluations_per_video
        # 必须同时遵守证据预算和客户端图片上限，避免客户端二次抽帧破坏证据对应关系。
        limit=min(cfg.max_images_per_evaluation,getattr(self.client,'max_frames',cfg.max_images_per_evaluation))
        try:
            curve=frame_scorer.get_category_scores(video_data,category)
            if curve is None: raise EvidenceError('missing_category_curve')
            data=video_data
            if cfg.dense_fps:
                try:
                    from .boundary_dense import local_dense_pool
                except ImportError:
                    from boundary_dense import local_dense_pool
                data,curve=local_dense_pool(video_data,seeds_by_id,category,frame_scorer,cfg)
            scorer=BoundaryEvidenceScorer(data,curve,cfg,self.priors.get(category,{}).get('expected_duration',(3.,30.)))
        except Exception as exc:
            trace['skipped_reason']=str(exc)
            return RefinementOutcome(output,applied,trace if cfg.trace_enabled or cfg.paired_eval else None)
        trace['evidence_pool']=data.get('evidence_pool','global')
        trace['decode_seconds']=data.get('decode_seconds',0.)
        trace['encode_seconds']=data.get('encode_seconds',0.)
        # 第一层门槛：先检查原边界和至少一个动作的证据覆盖，再筛低对比候选。
        # 因此 reason 是首个失败项，不是该候选所有问题的完整列表。
        eligible=[]
        for sid,seed in seeds_by_id.items():
            item=trace['candidates'][str(sid)]={'candidate_id':sid,'applied':False,'would_accept':False,
                'baseline':[seed['t_start'],seed['t_end']], 'boundary_method':seed.get('_boundary_method','unknown')}
            try:
                if not np.isfinite(seed.get('coarse_score',np.nan)): raise EvidenceError('missing_coarse_score')
                interval=(float(seed['t_start']),float(seed['t_end']))
                q=scorer.quality(interval); core=scorer.core_indices(interval)
                evidence=self.select_evidence(scorer,interval,[interval],core,limit)
                search=LocalBoundarySearch(interval,scorer,cfg,sid)
                valid_actions=[]
                for x in search.successors(search.root):
                    try:
                        scorer.quality(x); self.select_evidence(scorer,interval,[x],core,limit); valid_actions.append(x)
                    except EvidenceError: pass
                item.update(baseline_contrast_raw=q['contrast_raw'],baseline_curve_quality=q['quality'],
                            boundary_measurement_valid=True,evidence_coverage=evidence.coverage,
                            legal_evidence_actions=len(valid_actions),core_evidence_ids=core)
                if not valid_actions: raise EvidenceError('no_evidenced_action')
                if cfg.trigger_policy=='low_contrast' and q['contrast_raw']>=cfg.contrast_threshold:
                    raise EvidenceError('contrast_not_selected')
                eligible.append((q['contrast_raw'],interval[0],sid,search,core))
            except Exception as exc:
                item['reason']=str(exc)
        # 按对比值、开始时间、ID 排序；每视频只取预算允许的前几个候选。
        for rank,(_,_,sid,search,core) in enumerate(sorted(eligible)):
            item=trace['candidates'][str(sid)]
            if rank>=cfg.max_candidates_per_video or remaining<2:
                item['reason']='candidate_or_video_budget'; continue
            _log.info('[MCTS] search candidate=%s video=%s',sid,video_data.get('video_path'))
            allowance=min(cfg.max_vlm_evaluations_per_candidate,remaining)
            used=0; cache={}; evaluation_log=[]; started=time.monotonic()
            description=source_candidates_by_id.get(sid,{}).get('description') or category
            # logical_requests 是逻辑评价次数；实际 HTTP 次数/token 由 llm_usage 统计。
            # 配对缓存命中仍占逻辑预算，但不会重复产生 HTTP 请求。
            async def request(prompt,evidence,pair=False):
                nonlocal used,remaining
                if used>=allowance or remaining<=0: raise EvidenceError('request_budget')
                used+=1; remaining-=1; trace['logical_requests']+=1; trace['images_sent']+=len(evidence.indices)
                text=await self.client.generate(prompt,evidence.frames)
                answer=parse_answer(text,len(evidence.indices),pair)
                if pair and cfg.acceptance_policy=='relative_nonloss' and type(answer.get('preserves_reference_content')) is not bool:
                    raise EvidenceError('invalid_preserves_reference_content')
                return answer
            async def evaluate(node):
                evaluation_started=time.monotonic()
                simulation=len(evaluation_log)  # 根为0，其后每个扩展评估占一轮（包含缓存/无效）。
                try:
                    evidence=self.select_evidence(scorer,search.seed,[node.interval],core,limit)
                    prompt=build_prompt(category,description,evidence,search.seed,node.interval)
                    key=hashlib.sha256(prompt.encode()).hexdigest() # Candidate-local immutable frame pool.
                    cached=key in cache
                    # 相同候选内相同证据请求可以复用；失败也缓存，避免耗尽预算重复请求。
                    if not cached:
                        # 至少为最终原边界/新边界 A/B 比较保留一次请求。
                        if used>=allowance-1 or remaining<=1: raise EvidenceError('comparison_budget_reserved')
                        cache[key]=None  # Failed evaluations are not requested repeatedly.
                        cache[key]=await request(prompt,evidence)
                    answer=cache[key]
                    if answer is None: raise EvidenceError('cached_invalid_evaluation')
                    reward,parts=scorer.reward(node.interval,answer)
                    ev={'reward':reward,'parts':parts,'answer':answer,'interval':node.interval}
                    evaluation_log.append({'interval':node.interval,'depth':node.depth,'reward':reward,'cache_hit':cached,'parts':parts,'answer':answer})
                    if node is search.root:
                        # 沿用v1展示格式；异常存在性来自VLM，reward使用v3实际评分。
                        _log.info('[MCTS] Root eval: %.1f-%.1fs, anomaly_present=%s, reward=%.3f | candidate=%s video=%s',
                                  *node.interval, answer['target_present_inside'], reward,
                                  sid, video_data.get('video_path'))
                    else:
                        _log.info('[MCTS] Sim %s/%s: candidate=%s video=%s depth=%s interval=%.1f-%.1fs anomaly=%s reward=%.3f cache_hit=%s elapsed=%.3fs',
                                  simulation,cfg.simulations,sid,video_data.get('video_path'),node.depth,*node.interval,
                                  answer['target_present_inside'],reward,cached,time.monotonic()-evaluation_started)
                    return ev
                except asyncio.CancelledError:
                    evaluation_log.append({'interval':node.interval,'depth':node.depth,'invalid_reason':'cancelled'})
                    _log.info('[MCTS] Evaluation cancelled: candidate=%s video=%s simulation=%s depth=%s',
                              sid,video_data.get('video_path'),simulation,node.depth)
                    raise
                except Exception as exc:
                    evaluation_log.append({'interval':node.interval,'depth':node.depth,'invalid_reason':str(exc)})
                    if node is search.root:
                        # 无效评估没有可靠分数，不能伪装为anomaly_present=False或reward=0。
                        _log.warning('[MCTS] Root eval: %.1f-%.1fs, anomaly_present=unknown, reward=unavailable | candidate=%s video=%s error=%s',
                                     *node.interval, sid, video_data.get('video_path'), type(exc).__name__)
                    else:
                        _log.info('[MCTS] Sim %s/%s: candidate=%s video=%s depth=%s interval=%.1f-%.1fs reward=unavailable reason=%s',
                                  simulation,cfg.simulations,sid,video_data.get('video_path'),node.depth,*node.interval,str(exc))
                    return None
            async def work():
                try:
                    from .temporal_mcts import TemporalBoundaryMCTS
                except ImportError:
                    from temporal_mcts import TemporalBoundaryMCTS
                await TemporalBoundaryMCTS.refine(search,evaluate,lambda:used<allowance-1 and remaining>1)
                if search.root.evaluation is None: raise EvidenceError('invalid_root')
                root=search.root.evaluation
                # 第二层门槛：搜索评分高不等于可采用。先要求同一事件、完整且未切断，
                # 再限制 CLIP 质量下降；best_valid 模式不要求奖励一定高于根节点。
                good=[]
                for n in search.nodes[1:]:
                    ev=n.evaluation
                    if ev is None: continue
                    a=ev['answer']
                    if not proposal_eligible(a,root['answer'],cfg.acceptance_policy): continue
                    if scorer.quality(n.interval)['quality'] < item['baseline_curve_quality']-cfg.max_curve_quality_drop: continue
                    good.append(n)
                if not good: raise EvidenceError('no_valid_proposal')
                best=max(good,key=lambda n:n.evaluation['reward'])
                gain=best.evaluation['reward']-root['reward']
                item.update(proposed=list(best.interval),baseline_reward=root['reward'],proposal_reward=best.evaluation['reward'],reward_gain=gain)
                if cfg.comparison_gate=='reward_gain' and gain<cfg.min_reward_gain: raise EvidenceError('reward_gain_gate')
                evidence=self.select_evidence(scorer,search.seed,[search.seed,best.interval],core,limit)
                proposal_label='A' if (cfg.random_seed+sid)%2 else 'B'
                regions=changed_regions(search.seed,best.interval,evidence)
                prompt=build_prompt(category,description,evidence,search.seed,best.interval,proposal_label,cfg.acceptance_policy)
                item.update(pairwise_mapping={'proposal':proposal_label},changed_regions=regions,
                            changed_region_coverage=evidence.coverage,pairwise_attempts=[])
                answer=await request(prompt,evidence,True)
                for attempt in range(cfg.citation_repair_attempts+1):
                    semantic=semantic_failures(answer,proposal_label,cfg.acceptance_policy)
                    citations=citation_failures(answer,regions)
                    item['pairwise_attempts'].append({'answer':copy.deepcopy(answer),
                        'semantic_failures':semantic,'citation_failures':citations})
                    item.update(pairwise_verdict=answer,semantic_failures=semantic,citation_failures=citations)
                    if semantic or not citations:
                        break
                    # 只在所有语义条件已通过时纠正引用。继续使用原图和原预算，不重搜提案。
                    if attempt>=cfg.citation_repair_attempts:
                        item['repair_stop_reason']='attempt_limit'; break
                    if used>=allowance or remaining<=0:
                        item['repair_stop_reason']='request_budget'; break
                    corrected=await request(repair_prompt(prompt,answer,citations),evidence,True)
                    frozen=SEMANTIC_FIELDS+(('preserves_reference_content',) if cfg.acceptance_policy=='relative_nonloss' else ())
                    if any(corrected[k]!=answer[k] for k in frozen):
                        item['pairwise_attempts'].append({'answer':corrected,
                            'semantic_failures':semantic_failures(corrected,proposal_label,cfg.acceptance_policy),
                            'citation_failures':citation_failures(corrected,regions),
                            'repair_error':'repair_changed_semantics'})
                        item['rejection_reasons']=['repair_changed_semantics']+semantic+[x['code'] for x in citations]
                        raise EvidenceError('repair_changed_semantics')
                    answer=corrected
                item['rejection_reasons']=semantic+[x['code'] for x in citations]
                if semantic:
                    raise EvidenceError('comparison_rejected')
                if citations:
                    raise EvidenceError(citations[0]['code'])
                item['would_accept']=True
                # 可追溯的几何保护契约；曲线峰值是启发式锚点，不声称是VLM逐帧认证。
                inside=np.flatnonzero((scorer.times>=best.interval[0]) & (scorer.times<best.interval[1]))
                item['boundary_protection']={'accepted_interval':list(best.interval),
                    'anchor_timestamp':float(scorer.times[inside[np.argmax(scorer.curve[inside])]]),
                    'evidence_bounds':[min(evidence.timestamps),max(evidence.timestamps)]}
                # shadow 只记录 would_accept；只有 refine 才改边界并允许绕过 Stage 5b。
                if cfg.mode=='refine':
                    result=copy.deepcopy(seeds_by_id[sid]); result['t_start'],result['t_end']=best.interval
                    result.pop('boundary_contrast',None); result.pop('frame_indices',None)
                    result['depth']=best.depth; result['visits']=best.visits
                    output[sid]=result; applied.add(sid); item['applied']=True
                item['reason']='accepted' if cfg.mode=='refine' else 'shadow_would_accept'
            try:
                await asyncio.wait_for(work(),timeout=cfg.timeout_seconds_per_candidate)
            except asyncio.TimeoutError: item['reason']='timeout'
            except Exception as exc: item['reason']=str(exc)
            item.update(logical_requests=used,elapsed_seconds=time.monotonic()-started,stop_reason=search.stop_reason,
                        evaluations=evaluation_log,tree_nodes=len(search.nodes))
            _log.info('[MCTS] Search end: candidate=%s video=%s simulations=%s evaluations=%s stop_reason=%s outcome=%s elapsed=%.3fs',
                      sid,video_data.get('video_path'),max(0,len(evaluation_log)-1),len(evaluation_log),search.stop_reason,
                      item.get('reason'),item['elapsed_seconds'])
        for sid,item in trace['candidates'].items():
            _log.info('[MCTS] candidate=%s video=%s requests=%s applied=%s reason=%s failures=%s',
                      sid,video_data.get('video_path'),item.get('logical_requests',0),
                      item['applied'],item.get('reason'),item.get('rejection_reasons',[]))
        _log.info('[MCTS] summary video=%s mode=%s candidates=%s applied=%s logical_requests=%s',
                  video_data.get('video_path'),cfg.mode,len(trace['candidates']),len(applied),trace['logical_requests'])
        return RefinementOutcome(output,applied,trace if cfg.trace_enabled or cfg.paired_eval or cfg.boundary_guard_enabled else None)
