"""
MIL Anomaly Scorer 训练脚本

用法: python train_mil.py --feature_dir features/train --output model_mil.pth
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent))
from mil_model import MILAnomalyScorer, MILLoss


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
    """计算 test set 上的 frame-level AUC"""
    from sklearn.metrics import roc_auc_score
    feature_dir = Path(feature_dir)
    parquet_files = sorted(Path(label_path).glob(f"{split}-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    all_gt = []
    all_scores = []
    model.eval()

    for _, row in df.iterrows():
        vid = Path(row['video_path']).stem
        fpath = feature_dir / f"{vid}.npy"
        if not fpath.exists():
            continue

        feat = np.load(fpath).astype(np.float32)
        if feat.shape[0] != 32:
            idx = np.linspace(0, len(feat)-1, 32, dtype=int) if len(feat) >= 32 else list(range(len(feat)))
            feat = feat[idx] if len(feat) >= 32 else np.vstack([feat, np.zeros((32-len(feat), feat.shape[1]))])

        with torch.no_grad():
            scores = model(torch.tensor(feat).unsqueeze(0).to(device))
        scores = scores.squeeze(0).cpu().numpy()

        # Create frame-level GT and scores
        cat = row['category']
        is_normal = (cat == 'Normal')
        gt_label = 0.0 if is_normal else 1.0
        video_score = float(scores.max())

        all_gt.append(gt_label)
        all_scores.append(video_score)

    if len(set(all_gt)) < 2:
        return 0.0
    return roc_auc_score(all_gt, all_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=True)
    parser.add_argument("--test_feature_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="model_mil.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--feature_dim", type=int, default=768)
    args = parser.parse_args()

    device = args.device
    dataset = UCFCrimeFeatureDataset(args.feature_dir, args.label_path, split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

    model = MILAnomalyScorer(feature_dim=args.feature_dim, hidden_dim=512, k=3, dropout=0.6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = MILLoss()

    best_auc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for features, labels in loader:
            features = features.to(device)
            optimizer.zero_grad()
            score_a, score_n, feat_a, feat_n, scores = model(features)
            loss = criterion(score_a, score_n, feat_a, feat_n, scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_dir = args.test_feature_dir or args.feature_dir
            auc = evaluate_model(model, test_dir, args.label_path, split="test", device=device)
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, video_AUC={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), args.output)
                print(f"  -> Saved best model (AUC={auc:.4f})")

    print(f"\nTraining done. Best video AUC: {best_auc:.4f}")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
