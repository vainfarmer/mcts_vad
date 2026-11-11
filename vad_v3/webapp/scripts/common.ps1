$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
function Resolve-VadPython([string]$Requested) {
    if ($Requested) { return (Resolve-Path -LiteralPath $Requested).Path }
    $CommonPython = Join-Path $env:USERPROFILE '.conda\envs\common\python.exe'
    if (Test-Path -LiteralPath $CommonPython) { return $CommonPython }
    return (Get-Command python -ErrorAction Stop).Source
}
