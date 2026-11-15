param(
    [string]$DatasetDir = 'F:/dataset/xd-violence',
    [string]$OutputDir = (Join-Path $PSScriptRoot 'experiments/v218'),
    [int]$PerCategory = 5
)
$ErrorActionPreference = 'Stop'
if (-not $env:DASHSCOPE_API_KEY) {
    throw 'Set DASHSCOPE_API_KEY in this process before running the evaluation.'
}
$entry = Join-Path $PSScriptRoot 'video_anomaly_detection/batch_evaluate.py'
conda run --no-capture-output -n common python $entry `
    --dataset xd_violence --dataset_dir $DatasetDir --output_dir $OutputDir `
    --max_per_category $PerCategory --seed 42 --device cuda --clip_model ViT-L-14 `
    --concurrency 2 --api_concurrency 4 `
    --detect_model qwen2.5-vl-72b-instruct --classify_model qwen3-vl-plus `
    --max_tokens 2048 --max_frames 32 --api_timeout_seconds 120 --mcts_mode off
exit $LASTEXITCODE
