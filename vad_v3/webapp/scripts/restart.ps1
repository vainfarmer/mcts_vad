param([string]$PythonExe, [string]$DataDir, [int]$Port = 7860)
. (Join-Path $PSScriptRoot 'common.ps1')
$VadPython = Resolve-VadPython $PythonExe
if (-not $DataDir) { $DataDir = Join-Path $ProjectRoot 'vad_v3\web_data' }
Push-Location -LiteralPath $ProjectRoot
try {
    & $VadPython -m vad_v3.webapp.service stop --data-dir $DataDir
    if ($LASTEXITCODE -ne 0) { throw 'Stop failed.' }
    & $VadPython -m vad_v3.webapp.service start --data-dir $DataDir --port $Port
} finally { Pop-Location }
exit $LASTEXITCODE
