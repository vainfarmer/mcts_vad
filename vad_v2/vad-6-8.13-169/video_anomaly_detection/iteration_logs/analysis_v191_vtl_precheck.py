"""v191 VTL train-only precheck.

Frozen VideoMAE hidden temporal embeddings feed a lightweight causal temporal
MIL model. Evaluation is video-grouped, category-stratified five-fold OOF;
each outer fold obtains its decision threshold from three-fold OOF predictions
on that outer fold's training videos only. No test metadata, video, or GT is
read.

The extraction is resumable because VideoMAE-large inference over all 1,610
official train videos is intentionally expensive. One uniformly sampled
16-frame clip produces eight temporal patch groups on a fixed grid; spatial
tokens are averaged within each group, preserving an 8 x 1024 hidden sequence.

Usage:
  HF_HUB_OFFLINE=1 python3.14 -B \
    video_anomaly_detection/iteration_logs/analysis_v191_vtl_precheck.py
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path("/Users/zzl/PycharmProjects/vad")
DATASET = Path("/Users/zzl/Movies/ucf-crime")
TRAIN_PARQUET = DATASET / "data" / "train-00000-of-00001.parquet"
DEFAULT_OUT = ROOT / "results" / "v191_vtl_precheck"
MODEL_NAME = "OPear/videomae-large-finetuned-UCF-Crime"

N_FOLDS = 5
INNER_FOLDS = 3
N_FRAMES = 16
N_STEPS = 8
TOKENS_PER_STEP = 14 * 14
TOP_K = 2
SEED = 191
HARD_CATS = ("Shooting", "RoadAccidents", "Shoplifting", "Explosion")

GATE_AUC = 0.90
GATE_RAW_GAIN = 0.05
GATE_NORMAL_FPR = 0.20
GATE_HARD_RECALL = 0.70
GATE_EACH_HARD_RECALL = 0.50
GATE_CONTEXT_GAIN = 0.15
GATE_SOURCE2_CHANGE = 0.10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", choices=("mps", "cpu"), default="mps")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--extract-only", action="store_true")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def load_train():
    # This explicit singleton allowlist prevents accidental test-split globbing.
    assert TRAIN_PARQUET.name.startswith("train-")
    df = pd.read_parquet(TRAIN_PARQUET).copy()
    required = {"video_path", "category", "annotation_start1",
                "annotation_end1", "annotation_start2", "annotation_end2"}
    assert required.issubset(df.columns), sorted(df.columns)
    assert df["video_path"].is_unique
    assert len(df) == 1610, f"unexpected official train size: {len(df)}"
    # UCF-Crime train has video labels only; segment annotations must be absent.
    ann = df[["annotation_start1", "annotation_end1",
              "annotation_start2", "annotation_end2"]].to_numpy()
    assert np.all(ann == -1), "train split unexpectedly contains temporal GT"
    df["video_id"] = df["video_path"].map(lambda x: Path(str(x)).stem)
    df["is_anomaly"] = (df["category"] != "Normal").astype(np.float32)
    return df.sort_values("video_id").reset_index(drop=True)


def read_uniform_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        raise RuntimeError("video has no frames")
    frames = []
    for frame_i in np.linspace(0, n - 1, N_FRAMES, dtype=np.int64):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_i))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"failed to decode frame {frame_i}/{n}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames), n


def extract_features(df, out_dir, device):
    feature_dir = out_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    pending = [r for _, r in df.iterrows()
               if not (feature_dir / f"{r.video_id}.npz").exists()]
    print(f"[VTL-EXTRACT] total={len(df)} cached={len(df) - len(pending)} "
          f"pending={len(pending)}", flush=True)
    if not pending:
        return []

    sys.path.insert(0, str(ROOT / "video_anomaly_detection"))
    from videomae_classifier import VideoMAEClassifier

    clf = VideoMAEClassifier(model_name=MODEL_NAME, device=device,
                             num_frames=N_FRAMES)
    if clf.model is None:
        raise RuntimeError("frozen VideoMAE failed to load")
    clf.model.eval()
    if device == "mps":
        # Frozen fp16 inference is materially cheaper and the cache records it.
        clf.model.half()

    errors = []
    started = time.time()
    for done, row in enumerate(pending, 1):
        t0 = time.time()
        path = DATASET / row.video_path
        try:
            frames, source_frames = read_uniform_frames(path)
            x = clf._preprocess(frames)
            if device == "mps":
                x = x.half()
            with torch.inference_mode():
                hidden = clf.model.videomae(x).last_hidden_state
                if tuple(hidden.shape[1:]) != (N_STEPS * TOKENS_PER_STEP, 1024):
                    raise RuntimeError(f"unexpected hidden shape {tuple(hidden.shape)}")
                temporal = hidden[0].reshape(
                    N_STEPS, TOKENS_PER_STEP, 1024).mean(dim=1)
                logits = clf.model.classifier(temporal)
                raw = 1.0 - torch.softmax(logits.float(), dim=-1)[:, 7]
            np.savez_compressed(
                feature_dir / f"{row.video_id}.npz",
                embedding=temporal.float().cpu().numpy().astype(np.float16),
                raw_score=raw.cpu().numpy().astype(np.float32),
                source_frames=np.int64(source_frames),
            )
            del frames, x, hidden, temporal, logits, raw
            if device == "mps":
                torch.mps.empty_cache()
            elapsed = time.time() - started
            eta = elapsed / done * (len(pending) - done)
            print(f"[VTL-EXTRACT] {done}/{len(pending)} {row.video_id} "
                  f"{time.time() - t0:.1f}s eta={eta / 60:.1f}m", flush=True)
        except Exception as exc:
            err = {"video_id": row.video_id, "video_path": row.video_path,
                   "error": repr(exc)}
            errors.append(err)
            print(f"[VTL-EXTRACT-ERROR] {err}", flush=True)
    atomic_json(out_dir / "extraction_errors.json", errors)
    return errors


def load_features(df, out_dir):
    embeddings, raw_scores, missing = [], [], []
    for row in df.itertuples(index=False):
        path = out_dir / "features" / f"{row.video_id}.npz"
        if not path.exists():
            missing.append(row.video_id)
            continue
        with np.load(path) as z:
            emb = z["embedding"].astype(np.float32)
            raw = z["raw_score"].astype(np.float32)
        if emb.shape != (N_STEPS, 1024) or raw.shape != (N_STEPS,):
            missing.append(row.video_id)
            continue
        embeddings.append(emb)
        raw_scores.append(raw)
    if missing:
        raise RuntimeError(f"missing/invalid feature artifacts: {len(missing)}; "
                           f"first={missing[:5]}")
    return np.stack(embeddings), np.stack(raw_scores)


class CausalTemporalMIL(nn.Module):
    """Small learned head; only the causal convolution mixes time steps."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(1024)
        self.project = nn.Linear(1024, 64)
        self.context = nn.Conv1d(64, 64, kernel_size=3, groups=64)
        self.score = nn.Linear(64, 1)

    def forward(self, x, return_steps=False):
        z = F.gelu(self.project(self.norm(x)))
        mixed = self.context(F.pad(z.transpose(1, 2), (2, 0))).transpose(1, 2)
        step_logits = self.score(F.gelu(z + mixed)).squeeze(-1)
        video_logits = torch.topk(step_logits, TOP_K, dim=1).values.mean(dim=1)
        return (video_logits, step_logits) if return_steps else video_logits


def fit_model(x, y, epochs, seed):
    seed_all(seed)
    model = CausalTemporalMIL()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
    xb = torch.from_numpy(x)
    yb = torch.from_numpy(y.astype(np.float32))
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(neg / max(pos, 1.0)))
    order_rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for batch in np.array_split(order_rng.permutation(len(y)),
                                    max(1, int(np.ceil(len(y) / 64)))):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb[batch]), yb[batch])
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def predict(model, x):
    videos, steps = [], []
    xt = torch.from_numpy(x)
    for batch in np.array_split(np.arange(len(x)),
                                max(1, int(np.ceil(len(x) / 128)))):
        video_logits, step_logits = model(xt[batch], return_steps=True)
        videos.append(torch.sigmoid(video_logits).numpy())
        steps.append(torch.sigmoid(step_logits).numpy())
    return np.concatenate(videos), np.concatenate(steps)


def no_context_clone(model):
    """Causal intervention: zero only lag-2 and lag-1 convolution weights."""
    ablated = copy.deepcopy(model)
    with torch.no_grad():
        ablated.context.weight[:, :, :2] = 0.0
    return ablated.eval()


def fpr_threshold(scores, labels):
    normal = np.asarray(scores)[np.asarray(labels) == 0]
    if len(normal) == 0:
        raise RuntimeError("threshold fold has no Normal videos")
    # Lowest observed threshold satisfying FPR <= 0.20 maximizes recall under
    # the fixed constraint. Include +inf for pathological tied scores.
    for threshold in np.r_[np.unique(normal), np.inf]:
        if float(np.mean(normal >= threshold)) <= GATE_NORMAL_FPR:
            return float(threshold)
    raise AssertionError("unreachable threshold selection")


def topk_indices(scores):
    return tuple(int(i) for i in sorted(
        np.argsort(-np.asarray(scores), kind="stable")[:TOP_K]))


def run_oof(df, x, raw_steps, epochs):
    labels = df["is_anomaly"].to_numpy(dtype=np.float32)
    categories = df["category"].to_numpy()
    groups = df["video_id"].to_numpy()
    splitter = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)

    oof = np.full(len(df), np.nan, dtype=np.float32)
    oof_ablation = np.full(len(df), np.nan, dtype=np.float32)
    oof_steps = np.full((len(df), N_STEPS), np.nan, dtype=np.float32)
    fold_ids = np.full(len(df), -1, dtype=np.int64)
    thresholds = []
    fold_rows = []

    for fold, (train_idx, val_idx) in enumerate(
            splitter.split(x, categories, groups=groups)):
        inner = StratifiedGroupKFold(INNER_FOLDS, shuffle=True,
                                     random_state=SEED + 100 + fold)
        inner_scores = np.full(len(train_idx), np.nan, dtype=np.float32)
        for inner_fold, (it, iv) in enumerate(inner.split(
                x[train_idx], categories[train_idx], groups=groups[train_idx])):
            model = fit_model(x[train_idx[it]], labels[train_idx[it]], epochs,
                              SEED + fold * 10 + inner_fold)
            inner_scores[iv], _ = predict(model, x[train_idx[iv]])
        assert np.isfinite(inner_scores).all()
        threshold = fpr_threshold(inner_scores, labels[train_idx])

        model = fit_model(x[train_idx], labels[train_idx], epochs,
                          SEED + 1000 + fold)
        oof[val_idx], oof_steps[val_idx] = predict(model, x[val_idx])
        oof_ablation[val_idx], _ = predict(no_context_clone(model), x[val_idx])
        fold_ids[val_idx] = fold
        thresholds.append(threshold)
        fold_pred = oof[val_idx] >= threshold
        normal_mask = labels[val_idx] == 0
        fold_rows.append({
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "threshold_from_inner_oof": threshold,
            "validation_auc": float(roc_auc_score(labels[val_idx], oof[val_idx])),
            "validation_normal_fpr": float(np.mean(fold_pred[normal_mask])),
        })
        print(f"[VTL-FOLD] {fold} threshold={threshold:.6f} "
              f"auc={fold_rows[-1]['validation_auc']:.4f} "
              f"normal_fpr={fold_rows[-1]['validation_normal_fpr']:.4f}",
              flush=True)
    assert np.isfinite(oof).all() and np.isfinite(oof_ablation).all()
    assert np.isfinite(oof_steps).all() and np.all(fold_ids >= 0)
    return oof, oof_ablation, oof_steps, fold_ids, thresholds, fold_rows


def analyze(df, raw_steps, oof, oof_ablation, oof_steps, fold_ids,
            thresholds, fold_rows):
    y = df["is_anomaly"].to_numpy(dtype=np.int64)
    cats = df["category"].to_numpy()
    raw_video = np.sort(raw_steps, axis=1)[:, -TOP_K:].mean(axis=1)
    pred = np.array([oof[i] >= thresholds[fold_ids[i]]
                     for i in range(len(df))])
    normal = y == 0
    hard = np.isin(cats, HARD_CATS)

    auc = float(roc_auc_score(y, oof))
    raw_auc = float(roc_auc_score(y, raw_video))
    ablation_auc = float(roc_auc_score(y, oof_ablation))
    hard_recalls = {cat: float(np.mean(pred[cats == cat]))
                    for cat in HARD_CATS}
    source2_changed = np.array([
        topk_indices(raw_steps[i]) != topk_indices(oof_steps[i])
        for i in range(len(df))
    ])

    metrics = {
        "oof_auc": auc,
        "raw_frozen_head_auc": raw_auc,
        "auc_gain_over_raw": auc - raw_auc,
        "pooled_normal_fpr_fold_local_thresholds": float(np.mean(pred[normal])),
        "hard_category_recall_pooled": float(np.mean(pred[hard])),
        "hard_category_recall_each": hard_recalls,
        "no_temporal_context_ablation_auc": ablation_auc,
        "causal_temporal_context_auc_gain": auc - ablation_auc,
        "source2_top2_changed_rate": float(np.mean(source2_changed)),
        "source2_top2_changed_videos": int(np.sum(source2_changed)),
        "source2_total_videos": int(len(df)),
    }
    gates = {
        "oof_auc_at_least_0_90": auc >= GATE_AUC,
        "oof_auc_at_least_raw_plus_0_05": auc >= raw_auc + GATE_RAW_GAIN,
        "pooled_normal_fpr_at_most_0_20": metrics[
            "pooled_normal_fpr_fold_local_thresholds"] <= GATE_NORMAL_FPR,
        "hard_recall_at_least_0_70": metrics[
            "hard_category_recall_pooled"] >= GATE_HARD_RECALL,
        "each_hard_recall_at_least_0_50": all(
            v >= GATE_EACH_HARD_RECALL for v in hard_recalls.values()),
        "causal_context_auc_gain_at_least_0_15": (
            auc - ablation_auc >= GATE_CONTEXT_GAIN),
        "source2_topk_change_at_least_0_10": metrics[
            "source2_top2_changed_rate"] >= GATE_SOURCE2_CHANGE,
        "source2_count_grid_call_budget_invariant": True,
    }
    return {
        "go": bool(all(gates.values())),
        "data": {
            "split": "official train only",
            "parquet": str(TRAIN_PARQUET),
            "n_videos": int(len(df)),
            "n_normal": int(np.sum(normal)),
            "n_anomaly": int(np.sum(~normal)),
            "temporal_gt_used": False,
            "test_data_or_gt_used": False,
        },
        "recipe": {
            "frozen_encoder": MODEL_NAME,
            "precision": "fp16 extraction; fp32 temporal head",
            "clip_grid": "16 uniform source frames -> 8 temporal patch groups",
            "embedding": "spatial mean of each 14x14 final hidden-token group",
            "temporal_model": "LayerNorm, 1024->64, depthwise causal Conv1d(k=3), top-2 MIL",
            "outer_cv": "category-stratified video-grouped 5-fold",
            "thresholding": "outer-train-only 3-fold OOF; lowest threshold with Normal FPR<=0.20",
            "context_ablation": "same fitted model with lag-2/lag-1 convolution weights zeroed",
            "raw_comparator": "top-2 mean of per-temporal-group 1-P(Normal) frozen-head scores",
            "source2": "raw top-2 versus VTL top-2 on identical 8-cell grid",
        },
        "metrics": metrics,
        "gates": gates,
        "folds": fold_rows,
        "fold_thresholds": thresholds,
        "invariants": {
            "groups_are_unique_videos": bool(df["video_id"].is_unique),
            "grid_cells_per_video_raw": N_STEPS,
            "grid_cells_per_video_vtl": N_STEPS,
            "selected_source2_cells_raw": TOP_K,
            "selected_source2_cells_vtl": TOP_K,
            "implied_vlm_calls_raw": TOP_K,
            "implied_vlm_calls_vtl": TOP_K,
        },
    }


def main():
    args = parse_args()
    seed_all(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_train()
    manifest = {
        "data": str(TRAIN_PARQUET),
        "split": "train",
        "n_videos": int(len(df)),
        "category_counts": {str(k): int(v) for k, v in
                            df["category"].value_counts().sort_index().items()},
        "model": MODEL_NAME,
        "n_frames": N_FRAMES,
        "n_temporal_steps": N_STEPS,
        "seed": SEED,
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    errors = extract_features(df, args.output_dir, args.device)
    if errors:
        raise RuntimeError(f"feature extraction failed for {len(errors)} videos")
    if args.extract_only:
        print("[VTL] extraction complete (--extract-only)", flush=True)
        return

    x, raw_steps = load_features(df, args.output_dir)
    oof, ablation, learned_steps, folds, thresholds, fold_rows = run_oof(
        df, x, raw_steps, args.epochs)
    result = analyze(df, raw_steps, oof, ablation, learned_steps, folds,
                     thresholds, fold_rows)
    predictions = pd.DataFrame({
        "video_id": df["video_id"],
        "category": df["category"],
        "is_anomaly": df["is_anomaly"].astype(int),
        "fold": folds,
        "fold_local_threshold": [thresholds[f] for f in folds],
        "raw_video_score": np.sort(raw_steps, axis=1)[:, -TOP_K:].mean(axis=1),
        "vtl_oof_score": oof,
        "no_context_oof_score": ablation,
        "predicted_anomaly": [bool(oof[i] >= thresholds[folds[i]])
                              for i in range(len(df))],
        "raw_source2_top2": [json.dumps(topk_indices(v)) for v in raw_steps],
        "vtl_source2_top2": [json.dumps(topk_indices(v)) for v in learned_steps],
    })
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    atomic_json(args.output_dir / "precheck_result.json", result)
    print("[VTL-RESULT] " + json.dumps(result["metrics"], sort_keys=True),
          flush=True)
    print(f"[VTL-VERDICT] go={result['go']}", flush=True)


if __name__ == "__main__":
    main()
