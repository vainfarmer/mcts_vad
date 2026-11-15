param(
    [string]$Output = "$PSScriptRoot/experiments/v221",
    [string]$Source = "$PSScriptRoot/experiments/v220",
    [string]$Dataset = 'F:/dataset/xd-violence',
    [string]$Python = 'C:/Users/Gigabyte/.conda/envs/common/python.exe',
    [int]$PerClass = 10,
    [int]$Concurrency = 4,
    [int]$ApiConcurrency = 8,
    [switch]$AllAbnormal,
    [string]$ReuseValidated = '',
    [ValidateSet('', 'shrink', 'shrink_expand_shift')][string]$ActionSet = '',
    [string]$FrozenReviews = ''
)
$ErrorActionPreference = 'Stop'
if (-not $env:DASHSCOPE_API_KEY) { throw 'Set DASHSCOPE_API_KEY in this process before running.' }
$extraArgs = @()
if ($AllAbnormal) { $extraArgs += '--all_abnormal' }
if ($ReuseValidated) { $extraArgs += @('--reuse_validated', $ReuseValidated) }
if ($ActionSet) { $extraArgs += @('--mcts_action_set', $ActionSet) }
if ($FrozenReviews) { $extraArgs += @('--frozen_reviews', $FrozenReviews) }
& $Python -u "$PSScriptRoot/video_anomaly_detection/xd_verification_eval.py" `
    --source $Source --output_dir $Output --dataset_dir $Dataset --per_class $PerClass `
    --concurrency $Concurrency --api_concurrency $ApiConcurrency --seed 42 @extraArgs
if ($LASTEXITCODE -ne 0) { throw "XD verification evaluation failed: exit code $LASTEXITCODE" }
