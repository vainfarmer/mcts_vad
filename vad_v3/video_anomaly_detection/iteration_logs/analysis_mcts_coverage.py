"""Offline evidence coverage; seeds must be predicted Stage-5 intervals, not GT.

No VLM requests. A development-only GT file optionally diagnoses geometric reachability.
"""
# 离线覆盖诊断：GT仅用于开发集可达性分析，不能反馈给线上MCTS搜索或测试集调参。
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import VADConfig
from mcts_config import add_mcts_arguments
from video_preprocessor import VideoPreprocessor
from anomaly_expert import VideoAnomalyExpert
from frame_scorer import FrameScorer
from boundary_dense import local_dense_pool
from boundary_search import LocalBoundarySearch
from mcts_boundary_scorer import BoundaryEvidenceScorer, EvidenceError
from pipeline import CATEGORY_DURATION_PRIORS, MultiAnomalyVAD


def diagnose(vd,curve,seeds,cfg,category,gt_intervals=None):
    sc=BoundaryEvidenceScorer(vd,curve,cfg,CATEGORY_DURATION_PRIORS.get(category,{}).get('expected_duration',(3.,30.)))
    records=[]
    for sid,seed in enumerate(seeds):
        interval=(float(seed['t_start']),float(seed['t_end']))
        row={'candidate_id':sid,'baseline':interval,'root_evidence_valid':False,'legal_evidence_actions':0,'selected':False}
        try:
            q=sc.quality(interval); row.update(q)
            row['selected']=q['contrast_raw']<cfg.contrast_threshold
            core=sc.core_indices(interval)
            ev=sc.evidence(interval,[interval],core,cfg.max_images_per_evaluation)
            row['root_evidence_valid']=True; row['root_image_count']=len(ev.indices)
            tree=LocalBoundarySearch(interval,sc,cfg,sid)
            for x in tree.successors(tree.root):
                try:
                    sc.quality(x); sc.evidence(interval,[x],core,cfg.max_images_per_evaluation)
                    row['legal_evidence_actions']+=1
                except EvidenceError: pass
            if gt_intervals is not None:
                nodes=tree.enumerate_nodes()
                def iou(a,b):
                    intersection=max(0.,min(a[1],b[1])-max(a[0],b[0]))
                    return intersection/(a[1]-a[0]+b[1]-b[0]-intersection)
                row['development_only_reachable_best_iou']=max((iou(n.interval,g) for n in nodes for g in gt_intervals),default=0.)
                row['reachable_tree_states']=len(nodes)
        except ValueError as exc:
            row['reason']=str(exc)
        records.append(row)
    return {'candidates':records,'all_candidates':len(records),
            'selected_candidates':sum(r['selected'] for r in records),
            'valid_roots':sum(r['root_evidence_valid'] for r in records),
            'candidates_with_action':sum(r['legal_evidence_actions']>0 for r in records)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-path',required=True); p.add_argument('--seeds-json',required=True)
    p.add_argument('--category',required=True); p.add_argument('--config',help='VADConfig JSON')
    p.add_argument('--output',required=True); p.add_argument('--development-gt-json',help='Optional list of [start,end] in seconds, development data ONLY')
    a=p.parse_args(); cfg=VADConfig.load(a.config) if a.config else VADConfig()
    seeds=json.loads(Path(a.seeds_json).read_text(encoding='utf-8'))
    if not isinstance(seeds,list): raise ValueError('seeds-json must be a list of predicted Stage-5 candidates')
    pp=cfg.preprocess
    vd=VideoPreprocessor(pp.fps,pp.segment_duration,pp.motion_threshold,pp.min_motion_interval,pp.segment_overlap).process(a.video_path)
    cats=cfg.anomaly_categories or MultiAnomalyVAD.DEFAULT_ANOMALY_CATEGORIES
    expert=VideoAnomalyExpert(cats,cfg.clip.model_name,cfg.clip.pretrained,device=cfg.device)
    if expert.model is None: raise RuntimeError('CLIP load failed')
    vd['clip_features']=expert.precompute_video_features(vd['all_frames'])
    fs=FrameScorer(expert,cfg,cats); curve=fs.get_category_scores(vd,a.category)
    if cfg.mcts.dense_fps: vd,curve=local_dense_pool(vd,dict(enumerate(seeds)),a.category,fs,cfg.mcts)
    gt=json.loads(Path(a.development_gt_json).read_text()) if a.development_gt_json else None
    report=diagnose(vd,curve,seeds,cfg.mcts,a.category,gt)
    report.update(config=cfg.to_dict(),category=a.category,video_path=a.video_path)
    Path(a.output).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__': main()
