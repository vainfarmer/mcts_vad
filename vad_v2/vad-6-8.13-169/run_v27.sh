#!/bin/bash
export DASHSCOPE_API_KEY=sk-6b3cb9eef1fa43fa8d897a09a0f1dd0b
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
cd /Users/zzl/PycharmProjects/vad/video_anomaly_detection
python3 -u batch_evaluate.py \
  --dataset_dir /Users/zzl/Movies/ucf-crime \
  --output_dir ../results/batch_eval_v27 \
  --max_per_category 5 \
  --concurrency 3 \
  --detect_model qwen-vl-plus \
  --classify_model qwen3-vl-plus \
  --fps 2 \
  --clip_model ViT-L-14 \
  --clip_pretrained openai \
  --device mps \
  --mcts_simulations 12 \
  --mcts_max_depth 4
