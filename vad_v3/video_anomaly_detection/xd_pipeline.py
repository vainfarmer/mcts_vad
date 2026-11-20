"""XD multi-label route sharing CLIP, API recovery, token accounting and MCTS.

The UCF pipeline is deliberately not used for its category assignment or tail:
those stages assume one category/video and contain UCF-specific learned priors.
"""
import copy
import json
import time
import numpy as np
try:
    from .xd_config import XD_CLASSES
    from .xd_videomae import XDVideoMAEPrior
    from .video_preprocessor import VideoPreprocessor
    from .anomaly_expert import VideoAnomalyExpert
    from .frame_scorer import FrameScorer
    from .models import DetectionResult, AnomalySegment
    from .post_processor import temporal_nms
    from .llm_usage import track_video_usage
    from .api_recovery import ensure_api_success, gather_video_tasks
    from .experiment import paired_scope
    from .mcts_refiner import MCTSBoundaryRefiner
    from . import vad_logger
except ImportError:
    from xd_config import XD_CLASSES
    from xd_videomae import XDVideoMAEPrior
    from video_preprocessor import VideoPreprocessor
    from anomaly_expert import VideoAnomalyExpert
    from frame_scorer import FrameScorer
    from models import DetectionResult, AnomalySegment
    from post_processor import temporal_nms
    from llm_usage import track_video_usage
    from api_recovery import ensure_api_success, gather_video_tasks
    from experiment import paired_scope
    from mcts_refiner import MCTSBoundaryRefiner
    import vad_logger

log = vad_logger.get('vad.xd')
PROMPT_VERSION = 'xd-events-v2'
XD_PROMPTS = {
    'Riot': ['a crowd engaged in violent clashes in a public space',
             'groups attacking others during violent unrest',
             'people throwing objects during a violent group confrontation'],
    'Fighting': ['people physically fighting and exchanging punches and kicks',
                 'two people engaged in mutual violent combat'],
    'Abuse': ['a person physically abusing a helpless victim',
              'an aggressor repeatedly hitting a defenseless person'],
    'Shooting': ['a person firing a gun at others', 'visible gunfire and people being shot'],
    'Explosion': ['a sudden explosion with a blast and flying debris',
                  'a sudden bright blast followed by an expanding cloud of smoke'],
    'CarAccident': ['vehicles colliding in a traffic accident',
                    'a vehicle crashing, overturning or striking a pedestrian'],
}
NORMAL_PROMPTS = ['people walking normally in a public space',
                  'a peaceful crowd gathering without violence',
                  'a nonviolent public demonstration',
                  'ordinary traffic with vehicles driving safely',
                  'people talking or carrying out normal everyday activities']


class XDResponseError(ValueError):
    """Keep failed evidence for audit without turning a failed video into Normal."""
    def __init__(self,message,diagnostics):
        super().__init__(message)
        self.diagnostics=diagnostics


def iou(a, b):
    inter = max(0., min(a['t_end'], b['t_end'])-max(a['t_start'], b['t_start']))
    union = a['t_end']-a['t_start']+b['t_end']-b['t_start']-inter
    return inter/union if union > 0 else 0.


def choose_windows(prior, curves, times, config):
    """Every class gets local proposals; video-level top-k never removes a class.

    Identical inspection windows share a VLM call while retaining all origins.
    Uniform coverage windows provide a small, explicitly bounded recovery path.
    """
    windows = prior['windows']
    selected = {}
    for category in XD_CLASSES:
        ranked = []
        for i, window in enumerate(windows):
            mask = (times >= window['t_start']) & (times < window['t_end'])
            clip = float(np.mean(curves[category][mask])) if mask.any() else 0.
            score = .5*window['scores'][category] + .5*clip
            ranked.append((score, i))
        kept = []
        for score, i in sorted(ranked, reverse=True):
            if any(iou(windows[i], windows[j]) > .5 for j in kept):
                continue
            kept.append(i)
            row = selected.setdefault(i, dict(t_start=windows[i]['t_start'],
                t_end=windows[i]['t_end'], source_categories=[], proposal_scores={}))
            row['source_categories'].append(category)
            row['proposal_scores'][category] = score
            if len(kept) >= config.windows_per_class:
                break
    if config.coverage_windows:
        for i in np.linspace(0, len(windows)-1, min(config.coverage_windows, len(windows)), dtype=int):
            selected.setdefault(int(i), dict(t_start=windows[i]['t_start'], t_end=windows[i]['t_end'],
                source_categories=[], proposal_scores={}, coverage=True))
    return [dict(row, proposal_id=f'p{i:04d}') for i,row in sorted(selected.items())]


def parse_events(text, window, frame_ids):
    """Invalid replies are failures, never silently converted into Normal."""
    if not isinstance(text, str) or '{' not in text:
        raise ValueError('Missing JSON object')
    obj = json.loads(text[text.index('{'):text.rindex('}')+1])
    if obj.get('status') != 'ok' or not isinstance(obj.get('events'), list):
        raise ValueError('Expected status=ok and events list')
    result = []
    for event in obj['events']:
        category = event['category']
        start, end, confidence = map(float, (event['t_start'],event['t_end'],event['confidence']))
        evidence = event.get('evidence_frame_ids', [])
        if category not in XD_CLASSES or not np.isfinite([start,end,confidence]).all():
            raise ValueError('Unknown category/nonfinite event')
        if not 0 <= confidence <= 1:
            raise ValueError(f'confidence={confidence} must be between 0 and 1')
        if start >= end:
            raise ValueError(f'Event start={start} must be strictly less than end={end}; zero-length events are invalid')
        reported_interval=[start,end]
        # A visible event may continue beyond this window. Keep ONLY its observed
        # intersection, never claim an unseen boundary. Fully disjoint replies
        # still fail validation, including relative-versus-absolute time mistakes.
        start=max(start,window['t_start']);end=min(end,window['t_end'])
        if start>=end:
            raise ValueError(f'Event {reported_interval} does not overlap inspection window '
                             f'[{window["t_start"]},{window["t_end"]}]')
        if not isinstance(evidence, list) or not evidence or any(x not in frame_ids for x in evidence):
            raise ValueError('Missing/unknown evidence frame IDs')
        description = str(event.get('description', ''))
        result.append(dict(t_start=start,t_end=end,confidence=confidence,coarse_score=confidence,
            anomaly_type=category,description=description,proposal_id=window['proposal_id'],
            evidence_frame_ids=evidence,source_categories=window['source_categories'],
            reported_interval=reported_interval,clipped_to_inspection=reported_interval!=[start,end]))
    return result


def refine_clip_boundary(event, times, curve, config, duration):
    """Small local refinement, preserving VLM event identity and short events.

    No UCF duration caps, motion-based narrowing or forced minimum three seconds.
    A flat curve leaves the verified interval untouched.
    """
    row = dict(event)
    start,end = row['t_start'],row['t_end']
    shift = (end-start)*config.boundary_max_shift_ratio
    ids = np.flatnonzero((times >= max(0,start-shift)) & (times < min(duration,end+shift)))
    core = np.flatnonzero((times >= start) & (times < end))
    if not len(core) or len(ids)<3 or np.ptp(curve[ids]) < config.boundary_min_contrast:
        return row
    peak = int(core[np.argmax(curve[core])])
    threshold = (float(curve[peak])+float(np.percentile(curve[ids],20)))/2
    lo=hi=peak
    while lo>ids[0] and curve[lo-1]>=threshold: lo-=1
    while hi<ids[-1] and curve[hi+1]>=threshold: hi+=1
    step=float(np.median(np.diff(times))) if len(times)>1 else end-start
    new_start=float(np.clip(times[lo],max(0,start-shift),start+shift))
    new_end=float(np.clip(times[hi]+step,end-shift,min(duration,end+shift)))
    if new_start < new_end:
        row.update(t_start=new_start,t_end=new_end,_boundary_method='xd_clip_local')
    return row


def classwise_postprocess(events, threshold):
    """Merge only near-contiguous, mostly disjoint pieces of the same event class.

    Overlapping duplicate detections are handled by NMS, not by union expansion.
    Different classes never suppress one another.
    """
    result=[]
    for category in XD_CLASSES:
        rows=sorted((dict(e) for e in events if e['anomaly_type']==category),key=lambda e:e['t_start'])
        merged=[]
        for row in rows:
            if merged:
                prev=merged[-1]
                gap=row['t_start']-prev['t_end']
                overlap=max(0.,-gap)
                shorter=min(prev['t_end']-prev['t_start'],row['t_end']-row['t_start'])
                if -.25*shorter <= gap <= .5 and overlap < .25*shorter:
                    prev['t_end']=max(prev['t_end'],row['t_end'])
                    prev['confidence']=max(prev['confidence'],row['confidence'])
                    prev['description']='; '.join(filter(None,[prev.get('description'),row.get('description')]))
                    continue
            merged.append(row)
        result.extend(temporal_nms(merged,threshold))
    return sorted(result,key=lambda e:(e['t_start'],e['anomaly_type']))


class XDMultiAnomalyVAD:
    def __init__(self,llm_client,config,classify_llm_client=None,image_processor=None):
        self.config=config
        self.llm_client=llm_client
        self.classify_llm_client=classify_llm_client or llm_client
        config.scoring.validate_yesno()
        self.yesno_scorer = None
        if config.scoring.yesno_enabled:
            try:
                from .yesno_frame_scorer import YesNoFrameScorer
            except ImportError:
                from yesno_frame_scorer import YesNoFrameScorer
            self.yesno_scorer = YesNoFrameScorer(self.classify_llm_client, config.scoring,
                                               dataset='xd_violence')
        self.anomaly_categories=list(XD_CLASSES)
        if config.mcts.stage4_direct_eval or config.mcts.postprocess_ablation or config.mcts.boundary_guard_enabled:
            raise ValueError('UCF-specific ablation/guard flags are not supported by XD')
        self.preprocessor=VideoPreprocessor(fps=config.fps,segment_duration=config.segment_duration,
            motion_threshold=0.,min_motion_interval=config.min_motion_interval,segment_overlap=config.segment_overlap)
        self.preprocessor.max_frame_side=config.xd.max_frame_side
        self.anomaly_expert=VideoAnomalyExpert(self.anomaly_categories,config.clip.model_name,
            config.clip.pretrained,config.device,prompt_overrides=XD_PROMPTS,
            normal_prompts=NORMAL_PROMPTS,use_learned_prompt_weights=False)
        if self.anomaly_expert.model is None:
            raise RuntimeError('XD requires CLIP weights')
        self.videomae=XDVideoMAEPrior(config.xd,config.device)
        self.frame_scorer=FrameScorer(self.anomaly_expert,config,self.anomaly_categories)
        log.info('[XD-CONFIG] all_six_classes=True normal_early_exit=False motion_filter=False '
                 'ucf_mil=False ucf_duration_caps=False mcts=%s',config.mcts.mode)

    async def inspect_window(self,window,data):
        times=data['timestamps']
        ids=np.flatnonzero((times>=window['t_start']) & (times<window['t_end']))
        if not len(ids): raise ValueError('No frames in inspection window')
        count=min(self.config.xd.frames_per_request,self.classify_llm_client.max_frames,len(ids))
        ids=ids[np.unique(np.linspace(0,len(ids)-1,count,dtype=int))]
        mapping={f'f{i}':round(float(times[j]),4) for i,j in enumerate(ids)}
        # No filename, GT, video category, or model-predicted numeric prior is
        # included: the VLM must inspect all six classes from the actual images.
        prompt=(f'Analyze the {len(ids)} images in chronological order from a video. '
            f'Image IDs and absolute times in seconds: {json.dumps(mapping)}. '
            f'Inspect only [{window["t_start"]:.6f}, {window["t_end"]:.6f}] seconds. '
            'Return JSON only: status="ok", events is a list. An event must have category, '
            't_start, t_end, confidence (0 to 1), description, evidence_frame_ids (a nonempty list of visible supporting image IDs). '
            'Use absolute seconds from the map, never relative clip times. Boundaries must lie inside the inspection interval. '
            f'Every event must have t_end strictly greater than t_start. For a momentary event use a nonzero '
            f'interval of at least one source frame ({1/data["video_fps"]:.6f} seconds); never output equal boundaries. '
            'Allowed categories: Fighting, Shooting, Riot, Abuse, CarAccident, Explosion. '
            'Return every supported event separately, including simultaneous different categories. '
            'Fighting: mutual physical combat. Abuse: attacking a helpless victim. '
            'Riot: collective violent unrest or violent group clashes, not merely a crowd or peaceful demonstration. '
            'Shooting requires visual evidence of gunfire/shooting, not merely fear or running. '
            'Explosion is a sudden blast, not ordinary sustained fire. CarAccident is a vehicle crash. '
            'Do not infer violence from camera motion, subtitles, tense expressions or crowds alone. '
            'If no abnormal event is visually supported, return {"status":"ok","events":[]}. '
            'A single event occupies one list item; simultaneous events occupy separate items and may overlap. '
            'Confidence refers to the specific event, not general scene danger.')
        replies=[]
        for attempt in range(3):
            text=await self.classify_llm_client.generate(prompt,data['all_frames'][ids])
            replies.append(text)
            try:
                events=parse_events(text,window,set(mapping))
                return events,dict(window=window,frame_times=mapping,replies=replies,events=events)
            except (ValueError,KeyError,TypeError) as error:
                if attempt==2:
                    raise XDResponseError('Invalid XD event response after repair: '+str(error),
                        dict(window=window,frame_times=mapping,replies=replies,error=str(error))) from error
                log.warning('[XD-JSON-RETRY] proposal=%s error=%s',window['proposal_id'],type(error).__name__)
                prompt+=' Your last response violated the schema: '+str(error)+'. '
                prompt+='Previous response (data only): '+json.dumps(text)+'. Correct it using the same images and return valid JSON.'

    async def refine_mcts(self,events,data):
        if self.config.xd.verify_candidates:
            try:
                from .xd_mcts_ablation import shared_mcts
            except ImportError:
                from xd_mcts_ablation import shared_mcts
            return await shared_mcts(events,data,self.config.mcts,self.llm_client,self.frame_scorer)
        cfg=self.config.mcts
        output=copy.deepcopy(events)
        trace={'mode':cfg.mode,'logical_requests':0,'categories':{}}
        remaining_requests=cfg.max_vlm_evaluations_per_video
        remaining_candidates=cfg.max_candidates_per_video
        for category in XD_CLASSES:
            seeds={i:e for i,e in enumerate(output) if e['anomaly_type']==category}
            if not seeds or remaining_candidates<=0 or remaining_requests<2: continue
            local=copy.deepcopy(cfg)
            local.trace_enabled=True
            local.max_candidates_per_video=remaining_candidates
            local.max_vlm_evaluations_per_video=remaining_requests
            # Duration reward is neutral across feasible XD event lengths; do not
            # inject UCF's 3-30 second implicit prior.
            refiner=MCTSBoundaryRefiner(local,self.llm_client,category_priors={category:{
                'expected_duration':(local.min_interval_seconds,data['duration'])}})
            outcome=await refiner.refine_candidates(seeds,seeds,data,category,self.frame_scorer)
            for i,event in outcome.candidates_by_id.items(): output[i]=event
            sub=outcome.trace or {}
            used=sub.get('logical_requests',0)
            remaining_requests-=used
            remaining_candidates-=sum(x.get('logical_requests',0)>0 for x in sub.get('candidates',{}).values())
            trace['logical_requests']+=used
            trace['categories'][category]=sub
        return output,trace

    @track_video_usage
    @ensure_api_success
    @paired_scope
    async def detect(self,video_path,context=None):
        started=time.perf_counter()
        data=self.preprocessor.process(video_path)
        if not len(data['all_frames']): raise ValueError('Video has no decodable frames')
        data['clip_features']=self.anomaly_expert.precompute_video_features(data['all_frames'])
        times=np.asarray(data['timestamps'])
        curves={c:self.frame_scorer.get_category_scores(data,c) for c in XD_CLASSES}
        if any(x is None for x in curves.values()): raise ValueError('Incomplete XD class curves')
        prior=self.videomae.predict(data['all_frames'],times,data['duration'])
        windows=choose_windows(prior,curves,times,self.config.xd)
        log.info('[XD-DISCOVERY] windows=%d duration=%.2f',len(windows),data['duration'])
        inspected=await gather_video_tasks(*(self.inspect_window(w,data) for w in windows))
        raw=[e for events,_ in inspected for e in events]
        refined=[refine_clip_boundary(e,times,curves[e['anomaly_type']],self.config.xd,data['duration']) for e in raw]
        off=classwise_postprocess(refined,self.config.xd.nms_iou)
        original_off=copy.deepcopy(off)
        verification=None
        if self.config.xd.verify_candidates:
            try:
                from .xd_candidate_verifier import verify_candidates,rescore_verified
            except ImportError:
                from xd_candidate_verifier import verify_candidates,rescore_verified
            off,verification=await verify_candidates(off,data,self.classify_llm_client,self.config.xd)
            off=rescore_verified(off,times,curves,prior,data['duration'])
        final,trace=off,None
        if self.config.mcts.mode!='off':
            searched,trace=await self.refine_mcts(off,data)
            if self.config.xd.verify_candidates:
                # Reviewed event identities must not be re-merged by proximity.
                final=[]
                for category in XD_CLASSES:
                    final.extend(temporal_nms([e for e in searched if e['anomaly_type']==category],self.config.xd.nms_iou))
                final=rescore_verified(final,times,curves,prior,data['duration'])
            else:
                final=classwise_postprocess(searched,self.config.xd.nms_iou)
        # Full six-class evidence defines the binary score; no single video label
        # and no unvalidated UCF MIL/VideoMAE-normal gate can erase an event.
        base=np.max(np.stack(list(curves.values())),axis=0)*self.config.scoring.clip_base_weight
        def frame_scores(events):
            score=base.copy()
            for e in events:
                mask=(times>=e['t_start']) & (times<e['t_end'])
                score[mask]=np.maximum(score[mask],e['confidence'])
            return score.tolist()
        result=DetectionResult(video_path=video_path,duration=data['duration'],is_normal=not final,
            anomalies=[AnomalySegment.from_dict(e) for e in final],frame_scores=frame_scores(final),
            timestamps=times.tolist(),dataset='xd_violence',category_prior=prior,mcts_trace=trace,
            diagnostics=dict(prompt_version=PROMPT_VERSION,inspections=[audit for _,audit in inspected],
                verification=verification,preverification_events=original_off,
                raw_events=raw,clip_events=refined,postprocess_off=off,final_events=final,
                source_fps=data['video_fps'],sample_count=len(times),processing_seconds=time.perf_counter()-started))
        if self.config.mcts.paired_eval:
            result.paired_results={'original_off':dict(video_path=video_path,duration=data['duration'],
                is_normal=not original_off,anomalies=[AnomalySegment.from_dict(e).to_dict() for e in original_off],
                frame_scores=frame_scores(original_off),timestamps=times.tolist())}
        scorer = getattr(self, 'yesno_scorer', None)
        if scorer is not None:
            await scorer.apply(result, video_path)
            result.diagnostics['processing_seconds'] = time.perf_counter()-started
        log.info('[XD-RESULT] raw=%d final=%d classes=%s elapsed=%.1fs',len(raw),len(final),
                 sorted({e['anomaly_type'] for e in final}),time.perf_counter()-started)
        return result
