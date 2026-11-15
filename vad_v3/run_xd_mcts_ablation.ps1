param(
    [string]$Source = "$PSScriptRoot/experiments/v218",
    [string]$Output = "$PSScriptRoot/experiments/v219",
    [string]$Dataset = 'F:/dataset/xd-violence',
    [string]$Python = 'C:/Users/Gigabyte/.conda/envs/common/python.exe',
    [int]$Concurrency = 2,
    [int]$ApiConcurrency = 4,
    [int]$MaxCandidates = 4
)
$ErrorActionPreference = 'Stop'
if (-not $env:DASHSCOPE_API_KEY) { throw 'Set DASHSCOPE_API_KEY in this process before running.' }
& $Python -u "$PSScriptRoot/video_anomaly_detection/xd_mcts_ablation.py" `
    --source_dir $Source --output_dir $Output --dataset_dir $Dataset `
    --model qwen3-vl-plus --concurrency $Concurrency --api_concurrency $ApiConcurrency `
    --max_candidates $MaxCandidates
if ($LASTEXITCODE -ne 0) { throw "XD ablation failed: exit code $LASTEXITCODE" }
