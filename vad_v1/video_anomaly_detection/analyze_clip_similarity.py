"""
Analyze CLIP text embedding similarity between anomaly categories.

Investigates how well CLIP ViT-B-32 distinguishes between similar anomaly types
using the project's current simple prompts, and whether enhanced prompts help.
"""

import torch
import open_clip
import numpy as np

# ── 1. Load CLIP ViT-B-32 ──────────────────────────────────────────────────

print("Loading CLIP ViT-B-32 (openai)...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai", device="cpu"
)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

logit_scale = model.logit_scale.exp().detach().clamp(max=100.0).item()
print(f"Learned logit_scale (temperature): {logit_scale:.2f}")

# ── 2. Current project prompts ─────────────────────────────────────────────

CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]

prompts = [f"a surveillance camera video of {cat.lower()}" for cat in CATEGORIES]
prompts.append("normal daily activity in a surveillance camera")
labels = CATEGORIES + ["Normal"]

tokens = tokenizer(prompts).to("cpu")
with torch.no_grad():
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ── 3. Full cosine similarity matrix ───────────────────────────────────────

cos_sim = (text_features @ text_features.T).cpu().numpy()

print("\n" + "=" * 100)
print("FULL COSINE SIMILARITY MATRIX (raw, without logit_scale)")
print("=" * 100)

# Print header
header = f"{'':>16s}" + "".join(f"{l[:8]:>9s}" for l in labels)
print(header)
print("-" * len(header))

high_pairs = []
for i, li in enumerate(labels):
    row = f"{li:>16s}"
    for j, lj in enumerate(labels):
        val = cos_sim[i, j]
        marker = " *" if (i != j and val > 0.85) else "  "
        row += f"{val:7.4f}{marker}"
    print(row)
    for j in range(i + 1, len(labels)):
        if cos_sim[i, j] > 0.85:
            high_pairs.append((labels[i], labels[j], cos_sim[i, j]))

print("\n(* marks off-diagonal pairs with similarity > 0.85)")

if high_pairs:
    print(f"\nPairs with cosine similarity > 0.85 ({len(high_pairs)} found):")
    high_pairs.sort(key=lambda x: -x[2])
    for a, b, s in high_pairs:
        print(f"  {a:>16s} <-> {b:<16s}  sim = {s:.4f}")
else:
    print("\nNo off-diagonal pairs exceed 0.85 cosine similarity.")

# ── 4. Scaled similarity (with logit_scale) then softmax ────────────────────

print("\n" + "=" * 100)
print(f"SOFTMAX PROBABILITIES (logit_scale = {logit_scale:.2f})")
print("Each row: if one category's text embedding were treated as the 'query',")
print("what probability would softmax assign to each category?")
print("=" * 100)

scaled = cos_sim * logit_scale
# Numerically stable softmax
scaled_shifted = scaled - scaled.max(axis=1, keepdims=True)
softmax_matrix = np.exp(scaled_shifted) / np.exp(scaled_shifted).sum(axis=1, keepdims=True)

header = f"{'':>16s}" + "".join(f"{l[:8]:>9s}" for l in labels)
print(header)
print("-" * len(header))
for i, li in enumerate(labels):
    row = f"{li:>16s}"
    for j in range(len(labels)):
        val = softmax_matrix[i, j]
        row += f"{val:9.4f}"
    print(row)

# ── 5. Confusable groups ───────────────────────────────────────────────────

print("\n" + "=" * 100)
print("CONFUSABLE GROUP ANALYSIS")
print("=" * 100)

groups = {
    "Theft-related (Shoplifting, Stealing, Burglary, Robbery)":
        ["Shoplifting", "Stealing", "Burglary", "Robbery"],
    "Violence-related (Abuse, Assault, Fighting)":
        ["Abuse", "Assault", "Fighting"],
}

for group_name, members in groups.items():
    print(f"\n--- {group_name} ---")
    idxs = [labels.index(m) for m in members]

    header = f"{'':>16s}" + "".join(f"{m:>14s}" for m in members)
    print(header)
    for i, mi in zip(idxs, members):
        row = f"{mi:>16s}"
        for j in idxs:
            row += f"{cos_sim[i, j]:14.4f}"
        print(row)

    # Off-diagonal stats
    off_diag = []
    for ii, i in enumerate(idxs):
        for jj, j in enumerate(idxs):
            if ii < jj:
                off_diag.append((members[ii], members[jj], cos_sim[i, j]))
    off_diag.sort(key=lambda x: -x[2])

    print(f"\n  Pairwise similarities (sorted):")
    for a, b, s in off_diag:
        print(f"    {a:>16s} <-> {b:<16s}  {s:.4f}")

    # Show softmax confusion within this group
    print(f"\n  Softmax confusion within group (P(col | row's embedding)):")
    header2 = f"{'':>16s}" + "".join(f"{m:>14s}" for m in members)
    print(f"  {header2}")
    for i, mi in zip(idxs, members):
        sub_logits = np.array([cos_sim[i, j] for j in idxs]) * logit_scale
        sub_logits_shifted = sub_logits - sub_logits.max()
        sub_probs = np.exp(sub_logits_shifted) / np.exp(sub_logits_shifted).sum()
        row = f"  {mi:>16s}"
        for p in sub_probs:
            row += f"{p:14.4f}"
        print(row)

# ── 6. Enhanced prompts ────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("ENHANCED PROMPTS vs ORIGINAL (Theft-related group)")
print("=" * 100)

theft_cats = ["Robbery", "Burglary", "Shoplifting", "Stealing"]

original_prompts = {
    cat: f"a surveillance camera video of {cat.lower()}" for cat in theft_cats
}

enhanced_prompts = {
    "Robbery": "a surveillance camera video of robbery, armed person threatening and taking items by force",
    "Burglary": "a surveillance camera video of burglary, person breaking into a building or locked area",
    "Shoplifting": "a surveillance camera video of shoplifting, person concealing store merchandise and leaving without paying",
    "Stealing": "a surveillance camera video of stealing, person secretly taking someone else's belongings",
}

def compute_sim_matrix(prompt_dict, category_order):
    prompt_list = [prompt_dict[c] for c in category_order]
    toks = tokenizer(prompt_list).to("cpu")
    with torch.no_grad():
        feats = model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return (feats @ feats.T).cpu().numpy()

orig_sim = compute_sim_matrix(original_prompts, theft_cats)
enhanced_sim = compute_sim_matrix(enhanced_prompts, theft_cats)

print("\nOriginal simple prompts:")
for cat, prompt in original_prompts.items():
    print(f"  {cat:>16s}: \"{prompt}\"")

print("\nEnhanced prompts:")
for cat, prompt in enhanced_prompts.items():
    print(f"  {cat:>16s}: \"{prompt}\"")

print("\n--- Cosine Similarity: ORIGINAL ---")
header = f"{'':>16s}" + "".join(f"{c:>14s}" for c in theft_cats)
print(header)
for i, ci in enumerate(theft_cats):
    row = f"{ci:>16s}"
    for j in range(len(theft_cats)):
        row += f"{orig_sim[i, j]:14.4f}"
    print(row)

print("\n--- Cosine Similarity: ENHANCED ---")
print(header)
for i, ci in enumerate(theft_cats):
    row = f"{ci:>16s}"
    for j in range(len(theft_cats)):
        row += f"{enhanced_sim[i, j]:14.4f}"
    print(row)

print("\n--- DIFFERENCE (Enhanced - Original) ---")
print(header)
for i, ci in enumerate(theft_cats):
    row = f"{ci:>16s}"
    for j in range(len(theft_cats)):
        diff = enhanced_sim[i, j] - orig_sim[i, j]
        row += f"{diff:+14.4f}"
    print(row)

# Off-diagonal comparison
print("\n--- Pairwise off-diagonal comparison ---")
print(f"{'Pair':>35s}   {'Original':>10s}  {'Enhanced':>10s}  {'Change':>10s}")
print("-" * 70)
for i in range(len(theft_cats)):
    for j in range(i + 1, len(theft_cats)):
        o = orig_sim[i, j]
        e = enhanced_sim[i, j]
        d = e - o
        pair_name = f"{theft_cats[i]} <-> {theft_cats[j]}"
        direction = "BETTER" if d < 0 else "WORSE"
        print(f"{pair_name:>35s}   {o:10.4f}  {e:10.4f}  {d:+10.4f}  ({direction})")

# Softmax within group comparison
print("\n--- Softmax self-probability (P(correct | embedding), within 4 theft cats) ---")
print(f"{'Category':>16s}   {'Original':>10s}  {'Enhanced':>10s}  {'Change':>10s}")
print("-" * 55)
for i, cat in enumerate(theft_cats):
    orig_logits = orig_sim[i] * logit_scale
    orig_logits_s = orig_logits - orig_logits.max()
    orig_probs = np.exp(orig_logits_s) / np.exp(orig_logits_s).sum()

    enh_logits = enhanced_sim[i] * logit_scale
    enh_logits_s = enh_logits - enh_logits.max()
    enh_probs = np.exp(enh_logits_s) / np.exp(enh_logits_s).sum()

    print(f"{cat:>16s}   {orig_probs[i]:10.4f}  {enh_probs[i]:10.4f}  {enh_probs[i] - orig_probs[i]:+10.4f}")

def stable_softmax_self_prob(sim_matrix, idx, scale):
    logits = sim_matrix[idx] * scale
    logits_s = logits - logits.max()
    probs = np.exp(logits_s) / np.exp(logits_s).sum()
    return probs[idx]

avg_orig = np.mean([stable_softmax_self_prob(orig_sim, i, logit_scale)
                     for i in range(len(theft_cats))])
avg_enh = np.mean([stable_softmax_self_prob(enhanced_sim, i, logit_scale)
                    for i in range(len(theft_cats))])
print(f"{'AVERAGE':>16s}   {avg_orig:10.4f}  {avg_enh:10.4f}  {avg_enh - avg_orig:+10.4f}")

# ── 7. Summary ─────────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

# Most confusable pairs overall
all_pairs = []
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        all_pairs.append((labels[i], labels[j], cos_sim[i, j]))
all_pairs.sort(key=lambda x: -x[2])

print("\nTop-10 most similar category pairs (highest confusion risk):")
for rank, (a, b, s) in enumerate(all_pairs[:10], 1):
    print(f"  {rank:2d}. {a:>16s} <-> {b:<16s}  cosine = {s:.4f}")

print(f"\nMost distinct category pair:")
a, b, s = all_pairs[-1]
print(f"      {a:>16s} <-> {b:<16s}  cosine = {s:.4f}")

# Average within-group vs between-group
theft_idxs = [labels.index(c) for c in ["Shoplifting", "Stealing", "Burglary", "Robbery"]]
violence_idxs = [labels.index(c) for c in ["Abuse", "Assault", "Fighting"]]

def group_avg_off_diag(idxs):
    vals = []
    for ii in range(len(idxs)):
        for jj in range(ii + 1, len(idxs)):
            vals.append(cos_sim[idxs[ii], idxs[jj]])
    return np.mean(vals)

print(f"\nAverage within-group cosine similarity:")
print(f"  Theft group:    {group_avg_off_diag(theft_idxs):.4f}")
print(f"  Violence group: {group_avg_off_diag(violence_idxs):.4f}")

# Average between all pairs
all_off_diag = [cos_sim[i, j] for i in range(len(labels))
                for j in range(i + 1, len(labels))]
print(f"  All pairs avg:  {np.mean(all_off_diag):.4f}")

print(f"\nEnhanced prompts effect on theft group:")
orig_off = [orig_sim[i, j] for i in range(4) for j in range(i + 1, 4)]
enh_off = [enhanced_sim[i, j] for i in range(4) for j in range(i + 1, 4)]
print(f"  Avg off-diagonal cosine: {np.mean(orig_off):.4f} -> {np.mean(enh_off):.4f} "
      f"(change: {np.mean(enh_off) - np.mean(orig_off):+.4f})")
print(f"  Avg self-probability:    {avg_orig:.4f} -> {avg_enh:.4f} "
      f"(change: {avg_enh - avg_orig:+.4f})")

print("\nDone.")
