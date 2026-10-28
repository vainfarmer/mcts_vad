# v213 新提示词验证

相同14视频、现有data_parquet、随机种子42、ViT-L-14/openai（与V2相同激活）、CUDA、qwen3-vl-plus。视频并发度2。

MCTS refine + paired_eval + trace；默认shrink动作不变，citation_repair_attempts=1，提示词boundary-v3-2。只测试提示词/引用协议变化，不同时启用新外扩动作。

报告off/on/skip_only指标、搜索和接受数、引用纠正及拒绝原因、失败请求、时间与token。跨次与v212比较还受独立VLM回复和并发差异影响，不作单因果归因。时间与token为完整配对运行成本。
