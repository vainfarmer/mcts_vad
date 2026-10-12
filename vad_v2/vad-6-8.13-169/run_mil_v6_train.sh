#!/bin/bash
cd /Users/zzl/PycharmProjects/vad
python3 -m video_anomaly_detection.train_mil_v6 \
  --feature_dir features/train \
  --test_feature_dir features/test \
  --label_path /Users/zzl/Movies/ucf-crime/data \
  --output video_anomaly_detection/model_mil_v6.pth \
  --epochs 80 \
  --batch_size 16 \
  --lr 0.0003 \
  --lam_sparse 0.01 \
  --lam_smooth 0.002 \
  --lam_range 0.15 \
  --range_margin 0.35 \
  --lam_var 0.05 \
  --var_floor 0.12 \
  --device mps 2>&1 | tee results/mil_v6_train.log
