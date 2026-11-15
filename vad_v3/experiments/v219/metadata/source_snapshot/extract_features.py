"""
CLIP ViT-L/14 特征提取 — 为 MIL 训练准备 snippet-level 特征

每个视频 → 均匀采样 32 snippets → 每 snippet 取中间帧 → CLIP 编码 → (32, 768) npy
"""
import argparse
import cv2
import numpy as np
import sys
import time
from pathlib import Path

import pandas as pd
import torch

try:
    import open_clip
    from PIL import Image
except ImportError:
    print("需要 open_clip 和 PIL")
    sys.exit(1)


def extract_one_video(video_path, model, preprocess, device, num_segments=32):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_segments:
        cap.release()
        return None

    segment_len = total_frames // num_segments
    frames = []
    for seg_idx in range(num_segments):
        mid = seg_idx * segment_len + segment_len // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        if not ret:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    images = [preprocess(Image.fromarray(f)) for f in frames]
    batch = torch.stack(images).to(device)

    with torch.no_grad():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--clip_model", type=str, default="ViT-L-14")
    parser.add_argument("--num_segments", type=int, default=32)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load video list
    parquet_files = sorted((dataset_dir / "data").glob(f"{args.split}-*.parquet"))
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Found {len(df)} videos for split={args.split}")

    # Load CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained="openai", device=args.device)
    model.eval()
    print(f"CLIP {args.clip_model} loaded on {args.device}")

    # Extract
    t0 = time.time()
    done = 0
    skipped = 0
    for idx, row in df.iterrows():
        video_id = Path(row['video_path']).stem
        out_path = output_dir / f"{video_id}.npy"
        if out_path.exists():
            done += 1
            continue

        video_path = dataset_dir / row['video_path']
        if not video_path.exists():
            # Try alternate paths
            video_path = dataset_dir / "videos" / Path(row['video_path']).name
            if not video_path.exists():
                skipped += 1
                continue

        features = extract_one_video(video_path, model, preprocess, args.device, args.num_segments)
        if features is not None:
            np.save(out_path, features)
            done += 1
        else:
            skipped += 1

        if done % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(done, 1) * (len(df) - done)
            print(f"  [{done}/{len(df)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"Done: {done} extracted, {skipped} skipped in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
