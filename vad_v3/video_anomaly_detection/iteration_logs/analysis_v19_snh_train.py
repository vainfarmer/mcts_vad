#!/usr/bin/env python3.14
"""v19 SNH TrainDemo GO-gate, executed inline by the main session 2026-08-16
(the v19 workflow died overnight after feature extraction completed).
TRAIN-split only. GO iff CV normal-flag-rate >= 0.40 at the largest tau whose
CV anomaly-flag-rate <= 0.05. Numpy-only logistic regression (no sklearn dep).
"""
import json
import numpy as np

FEAT = '/Users/zzl/PycharmProjects/vad/results/snh_train_features.jsonl'
GO_MIN_NORMAL_FLAG = 0.40
GO_MAX_ANOM_FLAG = 0.05

rows = [json.loads(l) for l in open(FEAT)]
feat_names = sorted(rows[0]['features'].keys())
X = np.array([[r['features'][f] for f in feat_names] for r in rows], dtype=np.float64)
y = np.array([r['label'] for r in rows], dtype=np.float64)  # 1=normal, 0=anomaly
nonfinite = (~np.isfinite(X)).sum()
if nonfinite:
    print(f'sanitizing {nonfinite} non-finite feature values (MAD=0 z-scores) -> clip to +-50')
X = np.clip(np.nan_to_num(X, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
print(f'n={len(rows)} (normal={int(y.sum())} anomaly={int((1-y).sum())}) dims={len(feat_names)}')

def fit_lr(Xtr, ytr, l2=1.0, lr=0.05, steps=6000):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        g = Z.T @ (p - ytr) / len(ytr) + l2 * w / len(ytr)
        w -= lr * g; b -= lr * (p - ytr).mean()
    return w, b, mu, sd

def predict(Xte, w, b, mu, sd):
    return 1.0 / (1.0 + np.exp(-(((Xte - mu) / sd) @ w + b)))

# stratified 5-fold CV, fixed seed
rng = np.random.RandomState(20260816)
folds = [[] for _ in range(5)]
for cls in (0, 1):
    idx = np.where(y == cls)[0]; rng.shuffle(idx)
    for i, ix in enumerate(idx):
        folds[i % 5].append(ix)
oof = np.zeros(len(y))
for f in folds:
    te = np.array(f); tr = np.setdiff1d(np.arange(len(y)), te)
    w, b, mu, sd = fit_lr(X[tr], y[tr])
    oof[te] = predict(X[te], w, b, mu, sd)

# AUC (Mann-Whitney)
pos, neg = oof[y == 1], oof[y == 0]
ranks = np.argsort(np.concatenate([pos, neg])).argsort().astype(float) + 1
auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

# tau: largest threshold with anomaly-flag-rate <= 0.05
cands = np.unique(np.round(oof, 6))
best = None
for t in cands:
    af = (oof[y == 0] >= t).mean()
    nf = (oof[y == 1] >= t).mean()
    if af <= GO_MAX_ANOM_FLAG and (best is None or nf > best[2]):
        best = (t, af, nf)
tau, anom_flag, norm_flag = best if best else (float('nan'), float('nan'), 0.0)
print(f'CV ROC-AUC = {auc:.3f}')
print(f'tau = {tau:.4f}  ->  CV anomaly-flag-rate = {anom_flag:.3f} ({int((oof[y==0]>=tau).sum())}/{len(neg)}), CV normal-flag-rate = {norm_flag:.3f} ({int((oof[y==1]>=tau).sum())}/{len(pos)})')
go = (norm_flag >= GO_MIN_NORMAL_FLAG) and (anom_flag <= GO_MAX_ANOM_FLAG)
print(f'GO gate (>= {GO_MIN_NORMAL_FLAG:.0%} normals AND <= {GO_MAX_ANOM_FLAG:.0%} anomaly): {"GO" if go else "NO-GO"}')

# full-fit top features
w, b, mu, sd = fit_lr(X, y)
order = np.argsort(-np.abs(w))[:5]
print('top-5 |weight| features (full fit):')
for i in order:
    print(f'  {feat_names[i]:28s} w={w[i]:+.3f}')
print('score distributions: normal P>0.5 frac=%.2f  anomaly P>0.5 frac=%.2f' % ((pos > 0.5).mean(), (neg > 0.5).mean()))
print('normal  oof deciles:', np.round(np.percentile(pos, [10, 25, 50, 75, 90]), 3))
print('anomaly oof deciles:', np.round(np.percentile(neg, [10, 25, 50, 75, 90]), 3))

if go:
    model = {
        'version': 'v176-snh', 'trained_on': 'train-split 102 videos (50 normal / 52 anomaly), 5-fold CV',
        'cv_auc': float(auc), 'tau': float(tau),
        'cv_anomaly_flag_rate': float(anom_flag), 'cv_normal_flag_rate': float(norm_flag),
        'feature_order': feat_names, 'clip': 50.0,
        'scaler_mean': mu.tolist(), 'scaler_scale': sd.tolist(),
        'weights': w.tolist(), 'bias': float(b),
        'note': 'P(normal)=sigmoid((clip(nan_to_num(x))-mean)/scale @ w + b); veto all detections of a video iff P(normal) >= tau. Feature sanitization MUST match training (nan->0, +-50 clip).',
    }
    out = '/Users/zzl/PycharmProjects/vad/video_anomaly_detection/model_snh.json'
    json.dump(model, open(out, 'w'), indent=1)
    print('saved', out)
