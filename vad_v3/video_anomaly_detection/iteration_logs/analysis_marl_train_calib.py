"""v146 MARL-D train-ONLY threshold calibration.

FORBIDDEN to read the 65 test JSONs (C-mandate 1). Train has NO temporal
segment GT (annotation_* all -1 = weak supervision), so we cannot maximize
selected-run-vs-GT IoU. Instead we use a label-informed, segment-GT-free rule:

  The DECISIVENESS gate should fire only when a video's v1 MIL curve is more
  sharply peaked than the background (anomaly-free) distribution. NORMAL train
  videos contain no anomaly, so their (prominence, run_margin) distribution IS
  the background/flat level. We freeze:
      marl_min_prominence  = P90 of NORMAL-train prominence
      marl_min_run_margin  = P90 of NORMAL-train run_margin
  i.e. only relocate onto a run that is more decisive than 90% of anomaly-free
  videos. This is derived on features/train ONLY and never touches test.

  marl_halfmax_frac = 0.5  (standard half-max; NOT tuned)
  marl_min_gap_sec  = 6.0  (structural: one scan-segment = bue_lmin_sec; NOT tuned)

Run: python3.14 analysis_marl_train_calib.py
"""
import os, re, sys
import numpy as np
import torch

sys.path.insert(0, '/Users/zzl/PycharmProjects/vad/video_anomaly_detection')
from mil_model import MILAnomalyScorer

TRAIN = '/Users/zzl/PycharmProjects/vad/features/train'
MODEL = '/Users/zzl/PycharmProjects/vad/video_anomaly_detection/model_mil.pth'
HALFMAX_FRAC = 0.5


def load_model():
    m = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6)
    st = torch.load(MODEL, map_location='cpu', weights_only=True)
    m.load_state_dict(st)
    m.eval()
    return m


def mil_curve(model, feats):
    # feats: (32, 768) float32 -> pipeline builds (1,32,768) and mil_model(...).squeeze(0)
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        s = model(x).squeeze(0).cpu().numpy()
    return np.asarray(s, dtype=np.float64).reshape(-1)


def stats(s):
    """Same statistics as _marl_relocate: prominence and argmax-run margin."""
    p20 = float(np.percentile(s, 20))
    smax = float(s.max())
    prom = smax - p20
    hm = p20 + HALFMAX_FRAC * (smax - p20)
    n = len(s)
    ai = int(np.argmax(s))
    above = s >= hm
    lo = ai
    while lo - 1 >= 0 and above[lo - 1]:
        lo -= 1
    hi = ai
    while hi + 1 < n and above[hi + 1]:
        hi += 1
    run_mean = float(s[lo:hi + 1].mean())
    margin = run_mean - float(s.mean())
    return prom, margin


def main():
    model = load_model()
    files = sorted(os.listdir(TRAIN))
    normal_prom, normal_marg = [], []
    anom_prom, anom_marg = [], []
    n_done = 0
    for f in files:
        if not f.endswith('.npy'):
            continue
        cat = re.match(r'([A-Za-z]+)', f).group(1)
        feats = np.load(os.path.join(TRAIN, f))
        if feats.shape[0] < 32:
            continue
        s = mil_curve(model, feats)
        prom, marg = stats(s)
        if cat == 'Normal':
            normal_prom.append(prom); normal_marg.append(marg)
        else:
            anom_prom.append(prom); anom_marg.append(marg)
        n_done += 1
        if n_done % 200 == 0:
            print(f'... {n_done} done', flush=True)

    np_prom = np.asarray(normal_prom); np_marg = np.asarray(normal_marg)
    a_prom = np.asarray(anom_prom); a_marg = np.asarray(anom_marg)
    min_prom = float(np.percentile(np_prom, 90))
    min_marg = float(np.percentile(np_marg, 90))
    print('=== MARL-D TRAIN CALIBRATION (features/train only, no test) ===')
    print(f'normal N={len(np_prom)} anomaly N={len(a_prom)}')
    print(f'NORMAL   prominence  P50={np.percentile(np_prom,50):.4f} P90={np.percentile(np_prom,90):.4f} P95={np.percentile(np_prom,95):.4f}')
    print(f'ANOMALY  prominence  P50={np.percentile(a_prom,50):.4f} P90={np.percentile(a_prom,90):.4f}')
    print(f'NORMAL   run_margin  P50={np.percentile(np_marg,50):.4f} P90={np.percentile(np_marg,90):.4f} P95={np.percentile(np_marg,95):.4f}')
    print(f'ANOMALY  run_margin  P50={np.percentile(a_marg,50):.4f} P90={np.percentile(a_marg,90):.4f}')
    frac_anom_pass = float(np.mean((a_prom >= min_prom) & (a_marg >= min_marg)))
    frac_norm_pass = float(np.mean((np_prom >= min_prom) & (np_marg >= min_marg)))
    print(f'--> FROZEN marl_min_prominence = {min_prom:.4f}')
    print(f'--> FROZEN marl_min_run_margin = {min_marg:.4f}')
    print(f'--> FROZEN marl_halfmax_frac   = {HALFMAX_FRAC}')
    print(f'--> FROZEN marl_min_gap_sec    = 6.0 (structural, one scan-segment)')
    print(f'decisive-pass rate: anomaly={frac_anom_pass:.3f} normal={frac_norm_pass:.3f}')


if __name__ == '__main__':
    main()
