param([string]$PythonExe, [switch]$Full)
. (Join-Path $PSScriptRoot 'common.ps1')
$VadPython = Resolve-VadPython $PythonExe
Push-Location -LiteralPath $ProjectRoot
try {
    if ($Full) {
        & $VadPython -m pip install -r (Join-Path $ProjectRoot 'vad_v3\requirement.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Detector dependency installation failed.' }
    }
    & $VadPython -m pip install -r (Join-Path $ProjectRoot 'vad_v3\requirements-web.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Web dependency installation failed.' }
    & $VadPython -m vad_v3.webapp.service check
} finally { Pop-Location }
exit $LASTEXITCODE
