param(
    [string]$Output = "$PSScriptRoot/experiments/v220",
    [string]$Dataset = 'F:/dataset/xd-violence',
    [string]$Config = "$PSScriptRoot/experiments/v219/metadata/config.json",
    [string]$Python = 'C:/Users/Gigabyte/.conda/envs/common/python.exe',
    [int]$Concurrency = 4,
    [int]$ApiConcurrency = 8
)
$ErrorActionPreference = 'Stop'
if (-not $env:DASHSCOPE_API_KEY) { throw 'Set DASHSCOPE_API_KEY in this process before running.' }
& $Python -u "$PSScriptRoot/video_anomaly_detection/xd_full_ablation.py" `
    --dataset_dir $Dataset --output_dir $Output --config $Config --expected_videos 800 `
    --concurrency $Concurrency --api_concurrency $ApiConcurrency `
    --detect_model qwen3-vl-plus --classify_model qwen3-vl-plus
if ($LASTEXITCODE -ne 0) { throw "XD full evaluation failed: exit code $LASTEXITCODE" }
