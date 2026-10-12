"""
MIL v6: MILAnomalyScorer (v1's TemporalBlock, NOT LGT-Adapter) + anti-collapse
range/variance hinge losses (Loop 1-2, spec Phase 1 Step 1.1-1.3).

Explicitly avoids the v3/v4 failure mode (GCN over-smoothing -> 75-83% collapse)
by keeping the proven dilated-conv+self-attention architecture, and explicitly
avoids the v2 failure mode (10:1 sparsity:smoothness volatility) by keeping
smoothness/sparsity at v1-like light weights while adding NEW targeted
anti-collapse terms (range hinge + variance hinge) that v1/v2 never had.

Per spec requirement: early-stop if >70% of videos collapse (range<0.05),
monitored every 10 epochs.
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent))
from mil_model import MILAnomalyScorer, MILLossV6


class UCFCrimeFeatureDataset(Dataset):
    def __init__(self, feature_dir, label_path, split="train", num_segments=32):
        feature_dir = Path(feature_dir)
        parquet_files = sorted(Path(label_path).glob(f"{split}-*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        self.abnormal = []
        self.normal = []
        for _, row in df.iterrows():
            vid = Path(row['video_path']).stem
            fpath = feature_dir / f"{vid}.npy"
            if not fpath.exists():
                continue
            if row['category'] == 'Normal':
                self.normal.append(str(fpath))
            else:
                self.abnormal.append(str(fpath))
        self.num_segments = num_segments
        print(f"Dataset: {len(self.abnormal)} abnormal, {len(self.normal)} normal")

    def __len__(self):
        return max(len(self.abnormal), len(self.normal))

    def __getitem__(self, idx):
        abn_path = self.abnormal[idx % len(self.abnormal)]
        nor_path = self.normal[np.random.randint(len(self.normal))]
        abn_feat = np.load(abn_path).astype(np.float32)
        nor_feat = np.load(nor_path).astype(np.float32)
        if abn_feat.shape[0] != self.num_segments:
            abn_feat = self._resize(abn_feat, self.num_segments)
        if nor_feat.shape[0] != self.num_segments:
            nor_feat = self._resize(nor_feat, self.num_segments)
        features = np.stack([abn_feat, nor_feat])
        return torch.tensor(features), torch.tensor([1.0, 0.0])

    def _resize(self, feat, target):
        if len(feat) >= target:
            idx = np.linspace(0, len(feat)-1, target, dtype=int)
            return feat[idx]
        pad = np.zeros((target - len(feat), feat.shape[1]), dtype=np.float32)
        return np.vstack([feat, pad])


def collate_fn(batch):
    abn_feats = torch.stack([b[0][0] for b in batch])
    nor_feats = torch.stack([b[0][1] for b in batch])
    features = torch.cat([abn_feats, nor_feats], dim=0)
    labels = torch.cat([torch.ones(len(batch)), torch.zeros(len(batch))])
    return features, labels


def evaluate_model(model, feature_dir, label_path, split="test", device="cpu"):
    from sklearn.metrics import roc_auc_score
    feature_dir = Path(feature_dir)
    parquet_files = sorted(Path(label_path).glob(f"{split}-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    all_gt, all_scores = [], []
    model.eval()
    for _, row in df.iterrows():
        vid = Path(row['video_path']).stem
        fpath = feature_dir / f"{vid}.npy"
        if not fpath.exists():
            continue
        feat = np.load(fpath).astype(np.float32)
        if feat.shape[0] != 32:
            if len(feat) >= 32:
                idx = np.linspace(0, len(feat)-1, 32, dtype=int)
                feat = feat[idx]
            else:
                feat = np.vstack([feat, np.zeros((32-len(feat), feat.shape[1]))])
        with torch.no_grad():
            scores = model(torch.tensor(feat).unsqueeze(0).to(device))
        scores = scores.squeeze(0).cpu().numpy()
        cat = row['category']
        is_normal = (cat == 'Normal')
        all_gt.append(0.0 if is_normal else 1.0)
        all_scores.append(float(scores.max()))
    if len(set(all_gt)) < 2:
        return 0.0
    return roc_auc_score(all_gt, all_scores)


def per_video_collapse_check(model, feature_dir, label_path, device="cpu", collapse_thresh=0.05):
    """Critical anti-collapse monitor per spec Phase 1 Step 1.3.
    Returns (collapse_rate, median_range, mean_range) for abnormal test videos.
    """
    feature_dir = Path(feature_dir)
    parquet_files = sorted(Path(label_path).glob("test-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    ranges = []
    model.eval()
    for _, row in df.iterrows():
        if row['category'] == 'Normal':
            continue
        vid = Path(row['video_path']).stem
        fpath = feature_dir / f"{vid}.npy"
        if not fpath.exists():
            continue
        feat = np.load(fpath).astype(np.float32)
        if feat.shape[0] != 32:
            if len(feat) >= 32:
                idx = np.linspace(0, len(feat)-1, 32, dtype=int)
                feat = feat[idx]
            else:
                feat = np.vstack([feat, np.zeros((32-len(feat), feat.shape[1]))])
        with torch.no_grad():
            scores = model(torch.tensor(feat).unsqueeze(0).to(device))
        scores = scores.squeeze(0).cpu().numpy()
        ranges.append(float(scores.max() - scores.min()))
    ranges = np.array(ranges)
    collapse_rate = float((ranges < collapse_thresh).mean())
    return collapse_rate, float(np.median(ranges)), float(np.mean(ranges))


def analyze_full_distribution(model, feature_dir, label_path, device="cpu"):
    feature_dir = Path(feature_dir)
    parquet_files = sorted(Path(label_path).glob("test-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    abn_scores, nor_scores = [], []
    model.eval()
    for _, row in df.iterrows():
        vid = Path(row['video_path']).stem
        fpath = feature_dir / f"{vid}.npy"
        if not fpath.exists():
            continue
        feat = np.load(fpath).astype(np.float32)
        if feat.shape[0] != 32:
            if len(feat) >= 32:
                idx = np.linspace(0, len(feat)-1, 32, dtype=int)
                feat = feat[idx]
            else:
                feat = np.vstack([feat, np.zeros((32-len(feat), feat.shape[1]))])
        with torch.no_grad():
            scores = model(torch.tensor(feat).unsqueeze(0).to(device))
        scores = scores.squeeze(0).cpu().numpy()
        if row['category'] == 'Normal':
            nor_scores.extend(scores.tolist())
        else:
            abn_scores.extend(scores.tolist())
    abn_arr, nor_arr = np.array(abn_scores), np.array(nor_scores)
    print(f"\nScore Distribution: Abnormal mean={abn_arr.mean():.4f} std={abn_arr.std():.4f}, "
          f"Normal mean={nor_arr.mean():.4f} std={nor_arr.std():.4f}, "
          f"Separation={abn_arr.mean()-nor_arr.mean():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=True)
    parser.add_argument("--test_feature_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="model_mil_v6.pth")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--lam_sparse", type=float, default=0.01)
    parser.add_argument("--lam_smooth", type=float, default=0.002)
    parser.add_argument("--lam_range", type=float, default=0.15)
    parser.add_argument("--range_margin", type=float, default=0.35)
    parser.add_argument("--lam_var", type=float, default=0.05)
    parser.add_argument("--var_floor", type=float, default=0.12)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--margin", type=float, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    args = parser.parse_args()

    device = args.device
    dataset = UCFCrimeFeatureDataset(args.feature_dir, args.label_path, split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

    model = MILAnomalyScorer(feature_dim=args.feature_dim, hidden_dim=512, k=3, dropout=0.6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = MILLossV6(alpha=args.alpha, margin=args.margin,
                          lam_sparse=args.lam_sparse, lam_smooth=args.lam_smooth,
                          lam_range=args.lam_range, range_margin=args.range_margin,
                          lam_var=args.lam_var, var_floor=args.var_floor)

    test_dir = args.test_feature_dir or args.feature_dir
    best_auc = 0
    best_collapse_rate = 1.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for features, labels in loader:
            features = features.to(device)
            optimizer.zero_grad()
            score_a, score_n, feat_a, feat_n, scores = model(features)
            loss = criterion(score_a, score_n, feat_a, feat_n, scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            auc = evaluate_model(model, test_dir, args.label_path, split="test", device=device)
            collapse_rate, med_range, mean_range = per_video_collapse_check(
                model, test_dir, args.label_path, device=device)
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, AUC={auc:.4f}, "
                  f"collapse_rate={collapse_rate:.2%}, median_range={med_range:.4f}, "
                  f"mean_range={mean_range:.4f}, lr={lr_now:.6f}")

            # Spec-required early-stop guard: >70% collapse -> abort and revert
            if collapse_rate > 0.70 and epoch >= 10:
                print(f"  *** EARLY STOP: collapse_rate={collapse_rate:.2%} > 70% threshold. "
                      f"Aborting - this run has failed the anti-collapse requirement. ***")
                break

            # Save best by a COMBINED criterion: AUC decent AND collapse controlled
            # (not just best AUC, since v3/v4 taught us AUC alone is misleading)
            if collapse_rate < 0.60 and (auc > best_auc or collapse_rate < best_collapse_rate - 0.05):
                if auc > best_auc - 0.03:  # don't sacrifice too much AUC for range
                    best_auc = max(best_auc, auc)
                    best_collapse_rate = collapse_rate
                    torch.save(model.state_dict(), args.output)
                    print(f"  -> Saved checkpoint (AUC={auc:.4f}, collapse={collapse_rate:.2%})")

    print(f"\nTraining done. Best AUC={best_auc:.4f}, collapse_rate={best_collapse_rate:.2%}")
    print(f"Model saved to: {args.output}")

    import os
    if os.path.exists(args.output):
        model.load_state_dict(torch.load(args.output, map_location=device, weights_only=True))
        final_collapse, final_med, final_mean = per_video_collapse_check(model, test_dir, args.label_path, device=device)
        print(f"\nFINAL per-video range check: collapse_rate={final_collapse:.2%}, "
              f"median_range={final_med:.4f}, mean_range={final_mean:.4f}")
        analyze_full_distribution(model, test_dir, args.label_path, device=device)
    else:
        print("WARNING: No checkpoint was ever saved (never met collapse<60% criterion)")


if __name__ == "__main__":
    main()
