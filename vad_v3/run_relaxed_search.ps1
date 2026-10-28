# 宽松搜索入口。采用相对不退步接受标准，保留同一事件、内容保留与有效引用检查。
# 执行前在环境中设置 DASHSCOPE_API_KEY；输出应使用新的实验目录。
param(
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$DatasetDir = 'F:\dataset\ucf_crime',
    [int]$Concurrency = 2
)

if (-not $env:DASHSCOPE_API_KEY) {
    throw '请先设置环境变量 DASHSCOPE_API_KEY。'
}
if ($Concurrency -lt 1) {
    throw 'Concurrency 必须大于0。'
}
$batchScript = Join-Path $PSScriptRoot 'video_anomaly_detection\batch_evaluate.py'
& conda run --no-capture-output -n common python $batchScript `
    --dataset_dir $DatasetDir --output_dir $OutputDir `
    --max_per_category 1 --concurrency $Concurrency `
    --clip_model ViT-L-14 --device cuda `
    --detect_model qwen3-vl-plus --classify_model qwen3-vl-plus `
    --mcts_mode refine --mcts_paired_eval --mcts_trace `
    --mcts_final_nms_enabled `
    --mcts_trigger_policy all_valid `
    --mcts_acceptance_policy relative_nonloss `
    --mcts_action_set shrink_expand_shift `
    --mcts_max_images_per_evaluation 32 `
    --mcts_max_boundary_gap_seconds 1.5 `
    --mcts_max_evidence_gap_seconds 3.0 `
    --mcts_max_candidates_per_video 4 `
    --mcts_max_vlm_evaluations_per_candidate 10 `
    --mcts_max_vlm_evaluations_per_video 40
if ($LASTEXITCODE -ne 0) {
    throw "评测失败，退出码 $LASTEXITCODE；请检查输出日志。"
}
