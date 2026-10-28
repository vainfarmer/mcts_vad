# v210 预设实验配置

- 数据：F:/dataset/ucf_crime，官方 Anomaly_Test 290 条；按官方清单顺序分组，pandas sample(n=1, random_state=42)，14 类各一个视频，包含 Normal。
- CLIP：ViT-L-14 / openai；CUDA；并发视频数 1。
- VLM：检测和分类均为 qwen3-vl-plus。v209 的 qwen-vl-plus 返回 404，失败日志另行保留。
- MCTS：默认配置，refine + paired_eval + trace；不根据本次测试标注调参；dense_fps=0。
- 配对分支：original_off（相同上游的 V2 尾流程）、skip_only、on。完全相同的 VLM 请求按视频缓存复用。
- 主指标：temporal mAP（IoU .3/.5/.7 平均），辅以帧 AUC、AUPR、视频检测指标、类别准确率、实际 token、时间、MCTS 触发及接受率。
- 只评估选出的 14 个视频；视频级配对 bootstrap 1000 次。样本很少，不据此声称全测试集改善。
- on/off 的 shared-upstream 对照旨在隔离 MCTS 效果；不是重跑旧环境的历史 V2 数字。VideoMAE 加载与 CLIP 激活兼容性修复对所有分支相同。
- 时间和 token 是全部配对分支合计，不作为单独 on/off 部署成本。
