<#
.SYNOPSIS
  Windows equivalent of the Makefile (same target names).

.EXAMPLE
  .\scripts\tasks.ps1 setup
  .\scripts\tasks.ps1 train -RunProfile local_cpu -Set "train.batch_size=16"
#>
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet("setup", "fixtures", "download", "data", "train", "train-baseline", "serve", "ui", "server", "test", "clean")]
    [string]$Target,
    [string]$RunProfile = "",
    [string[]]$Set = @()
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = "src"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

$CfgArgs = @()
if ($RunProfile) { $CfgArgs += @("--profile", $RunProfile) }
foreach ($s in $Set) { $CfgArgs += @("--set", $s) }

function Invoke-Py { & $Py @args; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }

switch ($Target) {
    "setup" {
        # TensorFlow needs Python 3.11/3.12; uv fetches 3.11 without touching the system Python.
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            uv venv --python 3.11 .venv
            uv pip install --python .venv\Scripts\python.exe -r requirements.txt
        } else {
            python -m venv .venv
            .venv\Scripts\python.exe -m pip install -r requirements.txt
        }
    }
    "fixtures"       { Invoke-Py scripts/make_synthetic_fixtures.py --out tests/fixtures/synthetic }
    "download"       { Invoke-Py -m dr.download @CfgArgs }
    "data"           { Invoke-Py -m dr.data @CfgArgs }
    "train"          { Invoke-Py -m dr.train @CfgArgs }
    "train-baseline" { Invoke-Py -m dr.train @CfgArgs --set model.family=baseline_cnn --set model.pretrained=false }
    "serve"          { Invoke-Py -m dr.api @CfgArgs }
    "ui"             { Invoke-Py -m mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000 }
    "server"         { Invoke-Py -m mlflow server --backend-store-uri sqlite:///mlruns.db --artifacts-destination ./mlartifacts --serve-artifacts --host 127.0.0.1 --port 5000 }
    "test"           { Invoke-Py -m pytest -q }
    "clean" {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .pytest_cache, tests\fixtures\synthetic
        Get-ChildItem -Recurse -Directory -Filter __pycache__ | Where-Object { $_.FullName -notlike "*\.venv\*" } | Remove-Item -Recurse -Force
    }
}
