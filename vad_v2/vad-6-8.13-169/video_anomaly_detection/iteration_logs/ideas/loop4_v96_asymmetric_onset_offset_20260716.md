# Loop 4 (v96): Asymmetric Start/End Likelihood Curves (BSN-inspired)
## Date: 2026-07-16

## Research Confirmation (via direct WebSearch, Agent tool unavailable this session)
Confirmed BSN's exact mechanism: Temporal Evaluation Module (TEM) = 3-layer
1D conv producing THREE separate probability curves at each temporal
location: P_start(t), P_end(t), P_actionness(t). Proposal Generation Module
(PGM) combines high-confidence start/end points as candidate proposals
directly (combinatorial pairing), not threshold-crossing on ONE curve.

Also found LGN (Local-Global Network, related lineage): "locates temporal
boundaries with high starting and ending probabilities SEPARATELY, then
directly combines these boundaries as proposals. Globally, evaluates
actionness probability of multiple-duration temporal regions simultaneously"
- confirms the asymmetric-then-combine pattern is a real, validated
architecture choice (not just our training-free hypothesis).

## Training-Free Analog (validated conceptually, not yet empirically for VAD)
start_likelihood(t) = anomaly_score(t:t+w) - anomaly_score(t-w:t)  [forward
  contrast: is the FUTURE window more anomalous than the PAST window?]
end_likelihood(t) = anomaly_score(t-w:t) - anomaly_score(t:t+w)  [reverse]

This is directionally ASYMMETRIC, unlike our existing symmetric inner-outer
OIC-style contrast (which treats "outside" uniformly regardless of whether
it's before or after the segment).

## V96 Implementation Decision: Safe Confidence-Only Application
Following the SAME safe pattern validated in V95 (TSM sharpness) and V89
(diverse-perspective tie-break): compute onset/offset sharpness at an
ALREADY-DETECTED candidate's specific t_start/t_end (not used to search for
NEW boundaries), and use ONLY to modulate confidence, never boundary
position. This respects all 4 known failure modes:
1. Does not touch curve shape (computed from existing relative_scores,
   read-only, not fused into the primary curve used for half-max detection)
2. Does not change detection count (pure confidence modulation)
3. Not a VLM self-consistency technique
4. Does not change candidate/segment inclusion

Formula: onset_sharpness = relative_scores[t_start:t_start+w].mean() -
relative_scores[t_start-w:t_start].mean() (should be POSITIVE for a real
onset - future is more anomalous than past, at the START boundary)
offset_sharpness = relative_scores[t_end-w:t_end].mean() -
relative_scores[t_end:t_end+w].mean() (should be POSITIVE for a real
offset - past is more anomalous than future, at the END boundary)

Combined: boundary_directional_quality = (onset_sharpness + offset_sharpness)
normalized to [0,1], used as an ADDITIONAL confidence modulator alongside
V95's TSM sharpness (stack both signals, similar spirit to a small
ensemble of independent confidence-quality signals).
