"""
v109 offline pre-check — MIL cross-video confidence prior (Agent A proposal).

Re-ranking provably does NOT change detections/boundaries/count, so re-scoring
v108's REAL saved detections with the MIL per-video prior is a faithful in-vivo
(not simulated) measurement of what v109 would produce. Near-zero cost (no VLM).

conf' = conf * (base + (1-base) * mil_prior),  mil_prior = clip(topk_mean(mil_scores),0,1)

Sanity anchor: baseline re-score with prior=1.0 must reproduce v108 mAP=0.15834.
"""
import sys, json, glob, os, shutil, copy
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/Users/zzl/PycharmProjects/vad')
sys.path.insert(0, str(ROOT / 'video_anomaly_detection'))
from evaluate import evaluate  # noqa
from mil_model import MILAnomalyScorer  # noqa

V108_DIR = ROOT / 'results/batch_eval_v108_full'
FEAT_DIR = ROOT / 'features/test'
LABEL = str(ROOT / 'Movies_ucf_crime_data') if False else '/Users/zzl/Movies/ucf-crime/data'
MIL_PATH = ROOT / 'video_anomaly_detection/model_mil.pth'
K = 4  # top-k over 32 snippets (~32//8)


def load_mil():
    m = MILAnomalyScorer(feature_dim=768, hidden_dim=512, k=3, dropout=0.6)
    st = torch.load(str(MIL_PATH), map_location='cpu')
    st = st.get('model_state_dict', st) if isinstance(st, dict) else st
    m.load_state_dict(st)
    m.eval()
    return m


def mil_prior_for(video_id, mil):
    fp = FEAT_DIR / f'{video_id}.npy'
    if not fp.exists():
        return None
    feats = np.load(fp)  # (32,768)
    with torch.no_grad():
        scores = mil(torch.tensor(feats, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    topk = np.sort(scores)[-K:].mean()
    return float(np.clip(topk, 0.0, 1.0))


def run_eval(result_dir):
    res = evaluate(LABEL, str(result_dir), fps=30, split='test')
    tm = res['temporal_map'] if isinstance(res, dict) and 'temporal_map' in res else res
    return tm


def build_reranked(base, out_dir, priors, formula):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for jf in glob.glob(str(base / '*.json')):
        name = Path(jf).stem
        d = json.load(open(jf))
        p = priors.get(name)
        mult = formula(p) if p is not None else 1.0
        for a in d.get('anomalies', []):
            a['confidence'] = min(1.0, a.get('confidence', 0.0) * mult)
        json.dump(d, open(out_dir / f'{name}.json', 'w'))


def main():
    mil = load_mil()
    vids = [Path(f).stem for f in glob.glob(str(V108_DIR / '*.json'))]
    priors = {v: mil_prior_for(v, mil) for v in vids}
    n_ok = sum(1 for v in priors.values() if v is not None)
    normals = [v for v in vids if v.startswith('Normal')]
    print(f'videos={len(vids)} priors_ok={n_ok}')
    print('normal video priors:', {v: round(priors[v], 3) for v in normals if priors[v] is not None})
    anom_priors = [priors[v] for v in vids if not v.startswith('Normal') and priors[v] is not None]
    print(f'anomaly prior: mean={np.mean(anom_priors):.3f} median={np.median(anom_priors):.3f} min={np.min(anom_priors):.3f}')
    norm_priors = [priors[v] for v in normals if priors[v] is not None]
    if norm_priors:
        print(f'normal  prior: mean={np.mean(norm_priors):.3f} median={np.median(norm_priors):.3f} max={np.max(norm_priors):.3f}')

    # sanity: baseline (multiplier=1.0) must reproduce v108
    base_tm = run_eval(V108_DIR)
    print(f"\n[SANITY] v108 baseline mAP={base_tm['mAP']:.5f} "
          f"AP@0.3={base_tm.get('AP@0.3', base_tm.get('ap_0.3')):}")

    formulas = {
        'base0.7': lambda p: 0.7 + 0.3 * p,
        'base0.5': lambda p: 0.5 + 0.5 * p,
        'base0.8': lambda p: 0.8 + 0.2 * p,
        'base0.6': lambda p: 0.6 + 0.4 * p,
    }
    print(f"\n{'formula':<10} {'mAP':>9} {'AP@0.3':>9} {'AP@0.5':>9} {'AP@0.7':>9}  delta")
    print(f"{'baseline':<10} {base_tm['mAP']:>9.5f} {base_tm['AP@0.3']:>9.5f} {base_tm['AP@0.5']:>9.5f} {base_tm['AP@0.7']:>9.5f}")
    results = {'baseline': base_tm['mAP']}
    for name, fn in formulas.items():
        outd = ROOT / f'results/batch_eval_v109_precheck_{name}'
        build_reranked(V108_DIR, outd, priors, fn)
        tm = run_eval(outd)
        results[name] = tm['mAP']
        d = tm['mAP'] - base_tm['mAP']
        print(f"{name:<10} {tm['mAP']:>9.5f} {tm['AP@0.3']:>9.5f} {tm['AP@0.5']:>9.5f} {tm['AP@0.7']:>9.5f}  {d:+.5f}")
    json.dump({'priors': priors, 'results': results},
              open(ROOT / 'video_anomaly_detection/iteration_logs/analysis_v109_milprior_precheck.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
