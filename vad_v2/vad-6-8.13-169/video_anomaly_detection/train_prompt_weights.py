"""
v93: Learned prompt-ensemble weights (CoOp-lite, Loop 12 research).

Full CoOp requires injecting learnable continuous embeddings into CLIP's
token-embedding layer - complex, high-risk to touch model internals.
Tractable simplification: keep the existing 6 hand-crafted prompts per
category (already used, currently mean-ensembled), but LEARN a softmax
weight vector over the 6 prompts per category via a MIL-style top-k video-
level loss on the training set's video-level labels (video-level labels are
all we have - no frame-level GT in the standard UCF-Crime train split).

This operates STRICTLY upstream of the fusion/boundary detection logic:
the ONLY change is which weighted-combination of prompt embeddings becomes
"the" text embedding for a category, computed once and cached exactly like
today's mean-ensembled embedding. Zero risk to curve shape or candidate
pooling - the boundary detection algorithm and MIL fusion downstream are
completely unaware whether the text embedding came from a mean or a learned
weighted combination.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))


def load_category_prompts():
    from anomaly_expert import VideoAnomalyExpert
    return VideoAnomalyExpert.CATEGORY_PROMPTS, VideoAnomalyExpert.NORMAL_PROMPTS


def compute_raw_prompt_embeddings(model, tokenizer, prompts, device):
    """Returns (n_prompts, dim) L2-normalized embeddings, NOT ensembled."""
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default="features/train")
    parser.add_argument("--label_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="prompt_weights.json")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--k_topk", type=int, default=3)
    args = parser.parse_args()

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    device = args.device
    model = model.to(device).eval()

    CATEGORY_PROMPTS, NORMAL_PROMPTS = load_category_prompts()

    # Precompute raw (non-ensembled) per-prompt text embeddings
    normal_embs = compute_raw_prompt_embeddings(model, tokenizer, NORMAL_PROMPTS, device)
    normal_mean = F.normalize(normal_embs.mean(dim=0, keepdim=True), dim=-1)  # (1, dim)

    parquet_files = sorted(Path(args.label_path).glob("train-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    feature_dir = Path(args.feature_dir)

    logit_scale = model.logit_scale.exp().detach().clamp(max=100.0).item()

    learned_weights = {}
    for cat, prompts in CATEGORY_PROMPTS.items():
        cat_embs = compute_raw_prompt_embeddings(model, tokenizer, prompts, device)  # (6, dim)
        n_prompts = cat_embs.shape[0]

        abn_feats, nor_feats = [], []
        for _, row in df.iterrows():
            if row['category'] not in (cat, 'Normal'):
                continue
            vid = Path(row['video_path']).stem
            fpath = feature_dir / f"{vid}.npy"
            if not fpath.exists():
                continue
            feat = np.load(fpath).astype(np.float32)  # (32, 768)
            if row['category'] == cat:
                abn_feats.append(feat)
            else:
                nor_feats.append(feat)

        if len(abn_feats) < 3 or len(nor_feats) < 3:
            print(f"  [{cat}] insufficient data ({len(abn_feats)} abn, {len(nor_feats)} nor), skip -> uniform weights")
            learned_weights[cat] = [1.0 / n_prompts] * n_prompts
            continue

        abn_t = torch.tensor(np.stack(abn_feats)).to(device)  # (Na, 32, 768)
        nor_t = torch.tensor(np.stack(nor_feats)).to(device)  # (Nn, 32, 768)

        logits = torch.zeros(n_prompts, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=args.lr)

        for epoch in range(args.epochs):
            optimizer.zero_grad()
            w = torch.softmax(logits, dim=0)  # (n_prompts,)
            weighted_cat_emb = F.normalize((w.unsqueeze(-1) * cat_embs).sum(dim=0, keepdim=True), dim=-1)  # (1, dim)

            def video_level_score(feats):
                # feats: (N, 32, 768)
                sim_cat = (feats @ weighted_cat_emb.T).squeeze(-1) * logit_scale  # (N, 32)
                sim_nor = (feats @ normal_mean.T).squeeze(-1) * logit_scale  # (N, 32)
                probs = torch.softmax(torch.stack([sim_cat, sim_nor], dim=-1), dim=-1)[..., 0]  # (N, 32)
                topk = torch.topk(probs, k=min(args.k_topk, probs.shape[1]), dim=1).values
                return topk.mean(dim=1)  # (N,)

            score_abn = video_level_score(abn_t)
            score_nor = video_level_score(nor_t)
            labels = torch.cat([torch.ones_like(score_abn), torch.zeros_like(score_nor)])
            scores = torch.cat([score_abn, score_nor])
            loss = F.binary_cross_entropy(scores.clamp(1e-6, 1 - 1e-6), labels)
            loss.backward()
            optimizer.step()

        final_w = torch.softmax(logits, dim=0).detach().cpu().numpy().tolist()
        with torch.no_grad():
            auc_score_abn = video_level_score(abn_t).mean().item()
            auc_score_nor = video_level_score(nor_t).mean().item()
        print(f"  [{cat}] n_abn={len(abn_feats)}, n_nor={len(nor_feats)}, "
              f"weights={[f'{x:.3f}' for x in final_w]}, "
              f"abn_score={auc_score_abn:.3f}, nor_score={auc_score_nor:.3f}, "
              f"separation={auc_score_abn - auc_score_nor:.3f}")
        learned_weights[cat] = final_w

    with open(args.output, 'w') as f:
        json.dump(learned_weights, f, indent=2)
    print(f"\nSaved learned prompt weights to {args.output}")


if __name__ == "__main__":
    main()
