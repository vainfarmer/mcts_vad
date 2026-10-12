"""
MIL Snippet-Level Anomaly Scorer

RTFM 风格：时序卷积 + 自注意力 + top-k 特征幅度选择 + MIL ranking loss
输入: (batch, 32, 768) CLIP 特征
输出: (batch, 32) 逐 snippet 异常分数
"""
import torch
import torch.nn as nn
import numpy as np


class TemporalBlock(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim // 4, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(in_dim, out_dim // 4, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv1d(in_dim, out_dim // 4, 3, padding=4, dilation=4)
        self.query = nn.Conv1d(in_dim, out_dim // 4, 1)
        self.key = nn.Conv1d(in_dim, out_dim // 4, 1)
        self.value = nn.Conv1d(in_dim, out_dim // 4, 1)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        d4 = self.conv4(x)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.bmm(q.transpose(1, 2), k)
        attn = torch.softmax(attn / (q.shape[1] ** 0.5), dim=-1)
        tsa = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2)
        out = torch.cat([d1, d2, d4, tsa], dim=1)
        return self.relu(self.bn(out))


class MILAnomalyScorer(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=512, k=3, dropout=0.6):
        super().__init__()
        self.k = k
        self.embedding = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, 1),
            nn.ReLU()
        )
        self.temporal = TemporalBlock(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        xt = x.permute(0, 2, 1)
        xt = self.embedding(xt)
        xt = self.temporal(xt)
        xout = xt.permute(0, 2, 1)
        scores = self.classifier(xout).squeeze(-1)
        magnitudes = torch.norm(xout, p=2, dim=-1)

        if self.training:
            bs = B // 2
            abn_scores, nor_scores = scores[:bs], scores[bs:]
            abn_feats, nor_feats = xout[:bs], xout[bs:]
            abn_mag = self.drop(magnitudes[:bs])
            nor_mag = self.drop(magnitudes[bs:])
            _, idx_a = torch.topk(abn_mag, self.k, dim=1)
            _, idx_n = torch.topk(nor_mag, self.k, dim=1)
            feat_a = torch.gather(abn_feats, 1, idx_a.unsqueeze(-1).expand(-1, -1, xout.shape[-1]))
            feat_n = torch.gather(nor_feats, 1, idx_n.unsqueeze(-1).expand(-1, -1, xout.shape[-1]))
            score_a = torch.mean(torch.gather(abn_scores, 1, idx_a), dim=1)
            score_n = torch.mean(torch.gather(nor_scores, 1, idx_n), dim=1)
            return score_a, score_n, feat_a, feat_n, scores
        return scores


class MILLoss(nn.Module):
    def __init__(self, alpha=0.0001, margin=100, lam_sparse=0.01, lam_smooth=0.001):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.lam_sparse = lam_sparse
        self.lam_smooth = lam_smooth
        self.bce = nn.BCELoss()

    def forward(self, score_a, score_n, feat_a, feat_n, scores):
        bs = scores.shape[0] // 2
        label = torch.cat([torch.ones(bs), torch.zeros(bs)]).to(scores.device)
        s = torch.cat([score_a, score_n])
        loss_cls = self.bce(s, label)

        abn_mag = torch.norm(feat_a, p=2, dim=-1).mean(dim=-1)
        nor_mag = torch.norm(feat_n, p=2, dim=-1).mean(dim=-1)
        loss_mag = torch.mean(torch.relu(self.margin - abn_mag + nor_mag))

        abn_scores = scores[:bs]
        loss_smooth = torch.mean((abn_scores[:, 1:] - abn_scores[:, :-1]) ** 2)
        loss_sparse = torch.mean(abn_scores)

        return loss_cls + self.alpha * loss_mag + self.lam_smooth * loss_smooth + self.lam_sparse * loss_sparse


class MILLossV6(nn.Module):
    """v80: MILLoss + anti-collapse regularization (Loop 1-2 research).

    Adds two terms targeting the exact collapse-to-constant failure mode
    documented across v1 (45% collapse), v2 (volatile), v3/v4 (75-83% collapse):
      - Range hinge (K-maxmin Inner Bag Loss): forces gap between top-k and
        bottom-k snippet scores within EACH abnormal video to exceed a margin.
      - Variance hinge (VICReg-inspired): forces per-video score variance
        above a floor. Complementary/simpler backstop to the range hinge.
    Both are zero for normal videos in the range term (should stay uniform-low)
    but the variance hinge is abnormal-only (normal videos SHOULD have low
    variance - that's not a failure mode for them).
    """
    def __init__(self, alpha=0.0001, margin=100, lam_sparse=0.01, lam_smooth=0.001,
                 lam_range=0.15, range_margin=0.35, lam_var=0.05, var_floor=0.12,
                 k_range=3):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.lam_sparse = lam_sparse
        self.lam_smooth = lam_smooth
        self.lam_range = lam_range
        self.range_margin = range_margin
        self.lam_var = lam_var
        self.var_floor = var_floor
        self.k_range = k_range
        self.bce = nn.BCELoss()

    def forward(self, score_a, score_n, feat_a, feat_n, scores):
        bs = scores.shape[0] // 2
        label = torch.cat([torch.ones(bs), torch.zeros(bs)]).to(scores.device)
        s = torch.cat([score_a, score_n])
        loss_cls = self.bce(s, label)

        abn_mag = torch.norm(feat_a, p=2, dim=-1).mean(dim=-1)
        nor_mag = torch.norm(feat_n, p=2, dim=-1).mean(dim=-1)
        loss_mag = torch.mean(torch.relu(self.margin - abn_mag + nor_mag))

        abn_scores = scores[:bs]
        nor_scores = scores[bs:]
        loss_smooth = torch.mean((abn_scores[:, 1:] - abn_scores[:, :-1]) ** 2)
        loss_sparse = torch.mean(abn_scores)

        # v80: K-maxmin range hinge (anti-collapse, primary term)
        k = min(self.k_range, abn_scores.shape[1] // 2)
        abn_topk = torch.topk(abn_scores, k, dim=1).values.mean(dim=1)
        abn_botk = torch.topk(abn_scores, k, dim=1, largest=False).values.mean(dim=1)
        loss_range_abn = torch.relu(self.range_margin - (abn_topk - abn_botk)).mean()

        nor_topk = torch.topk(nor_scores, k, dim=1).values.mean(dim=1)
        nor_botk = torch.topk(nor_scores, k, dim=1, largest=False).values.mean(dim=1)
        loss_range_nor = (nor_topk - nor_botk).mean()  # minimize gap for normals

        loss_range = loss_range_abn + loss_range_nor

        # v80: variance hinge (anti-collapse, backstop term, abnormal-only)
        abn_var = abn_scores.var(dim=1)
        loss_var = torch.relu(self.var_floor - torch.sqrt(abn_var + 1e-8)).mean()

        return (loss_cls + self.alpha * loss_mag + self.lam_smooth * loss_smooth
                + self.lam_sparse * loss_sparse + self.lam_range * loss_range
                + self.lam_var * loss_var)
