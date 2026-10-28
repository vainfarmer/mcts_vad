# v214 宽松配置验证

现有data_parquet、相同14视频、ViT-L-14/V2激活、qwen3-vl-plus、CUDA、并发2。refine + paired_eval，提示词boundary-v3-3，all_valid，relative_nonloss，shrink_expand_shift。每请求32张图片，边界间隔1.5秒、覆盖间隔3秒，每视频4候选/40请求，每候选10请求；引用纠正最多1次且占原预算。

与本次original_off对照主要看完整尾流程mAP、接受数、各候选拒绝原因、时间及token。跨次与v213的差异包含多个配置变化及API随机性，不做单因素因果归因。不根据本次GT调参。
