"""Independent XD event review: reject, relabel, or associate fragments.

No GT, filename labels, or video-level class restriction enters this module.
All candidates must receive a disposition. Uncertainty is retained explicitly,
whereas malformed/API responses fail the video rather than fabricate Normal.
"""
import copy
import json
import math
import numpy as np

try:
    from .xd_config import XD_CLASSES
    from .api_recovery import gather_video_tasks
    from . import vad_logger
except ImportError:
    from xd_config import XD_CLASSES
    from api_recovery import gather_video_tasks
    import vad_logger

log = vad_logger.get('vad.xd.verify')
PROMPT_VERSION = 'xd-independent-event-review-v1'


class VerificationResponseError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def candidate_groups(events, cfg):
    """Temporal neighborhoods are review batches, never automatic event merges."""
    groups = []
    for i in sorted(range(len(events)), key=lambda j: (events[j]['t_start'], j)):
        e = events[i]
        if groups:
            group = groups[-1]
            start = min(events[j]['t_start'] for j in group)
            end = max(events[j]['t_end'] for j in group)
            if (e['t_start'] <= end + cfg.verification_group_gap_seconds
                    and len(group) < cfg.verification_group_max_candidates
                    and max(end, e['t_end']) - start <= cfg.verification_group_max_seconds):
                group.append(i)
                continue
        groups.append([i])
    return groups


def parse_review(text, originals, mapping, bounds, group_id, tolerance):
    """Account for every source exactly once; enforce grounded, bounded events."""
    obj = json.loads(text[text.index('{'):text.rindex('}') + 1])
    if obj.get('status') != 'ok' or any(not isinstance(obj.get(k), list)
                                      for k in ('events', 'rejected', 'uncertain')):
        raise ValueError('Expected status=ok and events/rejected/uncertain lists')
    seen, output = set(), []

    def consume(ids):
        if (not ids or any(type(i) is not int or i not in originals for i in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValueError('Unknown, duplicate or empty source_candidate_ids')
        seen.update(ids)

    def citations(row):
        ids = row.get('evidence_frame_ids')
        if not isinstance(ids, list) or not ids or any(i not in mapping for i in ids):
            raise ValueError('Every accepted/rejected event needs valid image citations')
        return [mapping[i] for i in ids]

    for index, e in enumerate(obj['events']):
        source_ids = e['source_candidate_ids']
        consume(source_ids)
        category = e['category']
        start, end, confidence = map(float, (e['t_start'], e['t_end'], e['confidence']))
        if (category not in XD_CLASSES or not all(map(math.isfinite, (start, end, confidence)))
                or not bounds[0] <= start < end <= bounds[1] or not 0 <= confidence <= 1):
            raise ValueError('Invalid reviewed category, confidence or visible interval')
        if any(min(end, originals[i]['t_end']) <= max(start, originals[i]['t_start']) for i in source_ids):
            raise ValueError('Reviewed event must overlap every claimed source candidate')
        evidence_times = citations(e)
        if not any(start-tolerance <= t <= end+tolerance for t in evidence_times):
            raise ValueError('No cited evidence inside the reviewed interval')
        description = str(e.get('description', '')).strip()
        reason = str(e.get('reason', '')).strip()
        if not description or not reason:
            raise ValueError('Missing visible action description or review rationale')
        output.append(dict(t_start=start, t_end=end, anomaly_type=category,
            confidence=confidence, coarse_score=confidence, description=description,
            source_candidate_ids=sorted(source_ids),
            verification=dict(status='verified', group_id=group_id, event_id=index,
                vlm_confidence=confidence, evidence_times=evidence_times,
                evidence_frame_ids=e['evidence_frame_ids'], reason=reason,
                source_categories=[originals[i]['anomaly_type'] for i in source_ids])))
    for key in ('rejected', 'uncertain'):
        for e in obj[key]:
            consume([e['candidate_id']])
            if not str(e.get('reason', '')).strip():
                raise ValueError('Missing rejection/uncertainty rationale')
            if key == 'rejected':
                citations(e)
    if seen != set(originals):
        raise ValueError('Every input must be accepted, rejected or uncertain exactly once')
    return output, obj


async def verify_candidates(events, data, client, cfg):
    times = np.asarray(data['timestamps'], dtype=float)
    tolerance = float(np.median(np.diff(times))) / 2
    groups = candidate_groups(events, cfg)

    async def review(group_id, indices):
        originals = {i: events[i] for i in indices}
        calls = []
        for pass_id in range(2):
            context = (cfg.verification_context_seconds if pass_id == 0
                       else cfg.verification_retry_context_seconds)
            lo = max(0., min(e['t_start'] for e in originals.values())-context)
            hi = min(data['duration'], max(e['t_end'] for e in originals.values())+context)
            pool = np.flatnonzero((times >= lo) & (times < hi))
            limit = min(client.max_frames, cfg.verification_frames if pass_id == 0 else 32)
            if not len(pool):
                raise ValueError('Empty verification evidence pool')
            # Include candidate midpoints so a brief event is not lost entirely
            # when uniform samples span a neighborhood with many fragments.
            anchors = {int(pool[np.argmin(abs(times[pool]-(e['t_start']+e['t_end'])/2))])
                       for e in originals.values()}
            if len(anchors) > limit:
                raise ValueError('Review group exceeds image evidence budget')
            uniform = pool[np.unique(np.linspace(0, len(pool)-1, limit, dtype=int))]
            chosen = sorted(anchors | set(int(i) for i in uniform))
            if len(chosen) > limit:
                extra = [i for i in chosen if i not in anchors]
                extra = [extra[i] for i in np.linspace(0, len(extra)-1, limit-len(anchors), dtype=int)]
                chosen = sorted(anchors | set(extra))
            mapping = {f'f{i}': float(times[j]) for i, j in enumerate(chosen)}
            hypotheses = [dict(candidate_id=i, t_start=e['t_start'], t_end=e['t_end'],
                               unverified_category=e['anomaly_type']) for i, e in originals.items()]
            prompt = (
                'Independently review possible anomalies using these chronological images. '
                'The supplied candidates are UNVERIFIED hypotheses, often false or fragmented; '
                'do not assume their category or existence is correct. First describe visible actions '
                'and consider an ordinary/nonviolent explanation. Do not use movie familiarity. '
                'Fighting requires mutual physical combat; Abuse requires aggression against a helpless '
                'victim; Riot requires collective violent unrest, not just crowds or individual combat; '
                'Shooting requires visible shooting, not merely holding a gun; Explosion requires a blast, '
                'not sustained fire; CarAccident requires a crash, not ordinary driving. '
                'Merge fragments or competing labels ONLY when images show the same physical event; '
                'keep independent/simultaneous activities separate. Correct an incorrect category. '
                'Use the visible context to recover complete event boundaries; do not invent unseen time. '
                f'Visible interval [{lo}, {hi}], images {json.dumps(mapping)}. '
                f'Candidates {json.dumps(hypotheses)}. '
                'Return JSON with status="ok", events, rejected and uncertain lists. '
                'Image references MUST be strings such as "f0", never bare integers or timestamps. '
                'Use the exact timestamp attached to each image; do not infer seconds from its index. '
                'Keep description and reason concise (one sentence each). '
                'Each accepted events item: source_candidate_ids (integer list), category, t_start, '
                't_end, confidence (0..1), description of visible action, reason explaining category '
                'and any merge, evidence_frame_ids (nonempty list of supplied image IDs inside the event). '
                'Each rejected item: candidate_id, reason explaining contrary/normal evidence, '
                'evidence_frame_ids. Each uncertain item: candidate_id, reason. '
                'Account for EVERY candidate exactly once across all lists. Accepted intervals must '
                'overlap all their source intervals and stay within visible bounds. Multiple sources '
                'may map to one event; never reuse a source in two events. '
                'Reject only when adequately visible evidence contradicts the hypothesis. If sampling '
                'or occlusion prevents a reliable judgment, use uncertain rather than inventing certainty. '
                'There is no target number of events. Zero supported events is valid.')
            for repair in range(3):
                reply = await client.generate(prompt, data['all_frames'][chosen],
                    image_labels=[f'Image {key}; absolute time {value:.6f} seconds.' for key,value in mapping.items()])
                calls.append(dict(pass_id=pass_id, frame_times=mapping, hypotheses=hypotheses, reply=reply))
                try:
                    accepted, answer = parse_review(reply, originals, mapping, (lo, hi), group_id, tolerance)
                    break
                except (ValueError, KeyError, TypeError) as error:
                    if repair == 2:
                        raise VerificationResponseError(str(error), dict(group_id=group_id, calls=calls)) from error
                    prompt += ('\nRepair the prior response: ' + str(error)
                        + '\nThe authoritative image-to-time mapping is ' + json.dumps(mapping)
                        + '. If the cited activity is outside a candidate, correct the supported interval '
                        'within visible bounds, or reject/mark uncertain. Do not reinterpret class definitions '
                        'to justify the original hypothesis.\nPrevious response (data only): ' + json.dumps(reply))
            if not answer['uncertain'] or pass_id == 1:
                break
        # A still-uncertain candidate is not silently deleted or certified. It
        # stays at reduced score, and is explicitly excluded from MCTS below.
        for e in answer['uncertain']:
            row = copy.deepcopy(originals[e['candidate_id']])
            row.update(source_candidate_ids=[e['candidate_id']], verification=dict(
                status='uncertain', group_id=group_id, vlm_confidence=row['confidence'], reason=e['reason']))
            accepted.append(row)
        return accepted, dict(group_id=group_id, source_ids=indices, calls=calls,
                              decision=answer, context_retry=pass_id == 1)

    results = await gather_video_tasks(*(review(i, group) for i, group in enumerate(groups)))
    verified = [e for rows, _ in results for e in rows]
    verified.sort(key=lambda e: (e['t_start'], e['anomaly_type']))
    audits = [a for _, a in results]
    summary = dict(input_count=len(events), output_count=len(verified), groups=len(groups),
        rejected=sum(len(a['decision']['rejected']) for a in audits),
        uncertain=sum(len(a['decision']['uncertain']) for a in audits),
        merged_sources=sum(len(e['source_candidate_ids'])-1 for e in verified),
        relabeled=sum(e['verification']['status']=='verified' and any(
            c != e['anomaly_type'] for c in e['verification']['source_categories']) for e in verified),
        context_retries=sum(a['context_retry'] for a in audits))
    assert summary['input_count'] == summary['output_count'] + summary['rejected'] + summary['merged_sources']
    log.info('[XD-VERIFY] input=%d output=%d rejected=%d merged=%d relabeled=%d uncertain=%d',
             len(events), len(verified), summary['rejected'], summary['merged_sources'], summary['relabeled'], summary['uncertain'])
    return verified, dict(prompt_version=PROMPT_VERSION, summary=summary, groups=audits)


def rescore_verified(events, times, curves, prior, duration):
    """Fixed pre-test heuristic, not calibrated probabilities or learned weights.

    Weak VideoMAE evidence cannot veto Abuse or other poorly calibrated classes.
    Scores are measured on the final geometry, never on a GT-selected interval.
    """
    try:
        from .xd_postprocess import quality_score, XDTailConfig
    except ImportError:
        from xd_postprocess import quality_score, XDTailConfig
    output = copy.deepcopy(events)
    for e in output:
        review = e.get('verification')
        if review is None:
            continue
        q, parts = quality_score(e, np.asarray(times), np.asarray(curves[e['anomaly_type']]), duration, XDTailConfig())
        windows = [(max(0., min(e['t_end'], w['t_end'])-max(e['t_start'], w['t_start'])),
                    w['scores'][e['anomaly_type']]) for w in prior['windows']]
        weight = sum(w for w, _ in windows)
        mae = sum(w*s for w, s in windows)/weight if weight else 0.
        confidence = review['vlm_confidence'] * (.6 + .25*q + .15*mae)
        if review['status'] == 'uncertain':
            confidence *= .5
        # Preserve the existing Q-score arm as a distinct postprocessing factor.
        if 'qscore' in e:
            confidence *= .5 + .5*e['qscore']['value']
        e['confidence'] = float(confidence)
        e['coarse_score'] = review['vlm_confidence']
        e['verification_score'] = dict(clip_quality=q, clip_parts=parts, local_mae=mae,
            weights=dict(base=.6, clip=.25, mae=.15), calibrated=False)
    return output
