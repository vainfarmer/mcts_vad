# v212 配对实验

V3 CLIP 移除 force_quick_gelu，与 V2 使用完全相同的 create_model_and_transforms 参数。ViT-L-14/openai，CUDA，并发视频1。检测/分类均用已验证可调用的 qwen3-vl-plus。

直接读取 F:/dataset/ucf_crime/data_parquet；每类1视频，random_state=42，启动时断言与 v210 的14个视频完全一致。不重新生成索引。

MCTS 使用既定默认参数，refine + paired_eval + trace；所有分支共享上游，完全相同的请求按视频复用。比较 original_off、skip_only、on 完整尾流程。主指标 mAP；补充帧 AUC/AUPR、视频指标、搜索接受率、时间、token。视频级 bootstrap 1000次。

原实验 v212_model_probe 是独立模型可用性测试，保留不覆盖。当前实验目录 v212 专用于本次性能评测。预训练权重的 QuickGELU mismatch 警告预期会与 V2 一样出现，这正是本次要求保持一致的配置。
