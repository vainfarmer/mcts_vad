"""Frozen, category-blind visual likelihood scorer for the AUC experiment."""
import math

PROMPT = """You are assessing a short surveillance video clip. The images are in chronological order.
Judge the central time interval specified below using the surrounding images as context.
An abnormal event means a visibly supported act of violence, assault, fighting, abuse, shooting, theft, shoplifting, robbery, forced entry, vandalism, deliberate fire setting, explosion, traffic collision, or police/security physically restraining someone.
Compare against ordinary activity: walking, talking, shopping, handling one's own belongings, normal entry or exit, routine work, and normal traffic.
Base the judgment on observable actions, interactions, object transfers and changes over time. Scene type, a hidden hand, looking around, crowding, or motion alone is insufficient evidence. Subtle actions may still be abnormal when supported by the sequence. Do not infer an event merely because it may have occurred before or after the target interval.
Does a visibly supported abnormal event occur within the target interval?
Reply with exactly Yes or No, matching this capitalization. Give no explanation or numeric confidence.
"""


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def extract_score(response, max_censor_width=0.001):
    """Compare canonical Yes/No at the same first-token position.

    DashScope returns at most five candidates. If an alternative is absent,
    its log probability is bounded by the fifth candidate. Use the midpoint
    of the resulting narrow score interval and explicitly record censoring.
    Never substitute a generated confidence, binary answer, or fabricated logit.
    """
    choice = response['choices'][0]
    positions = (choice.get('logprobs') or {}).get('content') or []
    if not positions:
        raise ValueError('Missing actual token log probabilities')
    first = positions[0]
    if first['token'] not in ('Yes', 'No'):
        raise ValueError('First response token is not canonical Yes/No')
    candidates = {v['token']: float(v['logprob']) for v in first['top_logprobs']}
    candidates[first['token']] = float(first['logprob'])
    if any(not math.isfinite(v) or v > 1e-6 for v in candidates.values()):
        raise ValueError('Invalid log probability')
    yes, no = candidates.get('Yes'), candidates.get('No')
    if yes is not None and no is not None:
        score = sigmoid(yes - no)
        lower = upper = score
        kind = 'exact_pair'
    else:
        if len(first['top_logprobs']) != 5:
            raise ValueError('Missing alternative without a complete top-five bound')
        cutoff = min(float(v['logprob']) for v in first['top_logprobs'])
        if yes is not None:
            lower, upper = sigmoid(yes - cutoff), 1.0
        elif no is not None:
            lower, upper = 0.0, sigmoid(cutoff - no)
        else:
            raise ValueError('Neither canonical label is present')
        if upper - lower > max_censor_width:
            raise ValueError('Top-five probability bound is too wide')
        score = (lower + upper) / 2
        kind = 'top5_bounded_midpoint'
    return dict(score=score, lower=lower, upper=upper, kind=kind,
                yes_logprob=yes, no_logprob=no,
                hard_score=float(first['token'] == 'Yes'),
                pair_probability_mass=(math.exp(yes) + math.exp(no)) if yes is not None and no is not None else None)


def windows(duration, stride=5., context=10.):
    """Cover the whole input, including the last partial target interval."""
    if duration <= 0:
        raise ValueError('Non-positive duration')
    result = []
    for i in range(math.ceil(duration / stride)):
        start, end = i * stride, min((i + 1) * stride, duration)
        center = (start + end) / 2
        result.append(dict(index=i, start=start, end=end, center=center,
                           context_start=max(0., center-context/2),
                           context_end=min(duration, center+context/2)))
    return result
