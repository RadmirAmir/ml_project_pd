<#
.SYNOPSIS
    Build the Docker image of the scoring API, run it locally and smoke-test it.

.DESCRIPTION
    1. verifies that the trained model artifact exists (models/*.joblib);
    2. builds the image (requirements.txt = inference dependencies);
    3. (re)starts the container on the requested host port;
    4. waits for /health and runs scripts/smoke_test_api.py.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/docker_build_run.ps1
    powershell -ExecutionPolicy Bypass -File scripts/docker_build_run.ps1 -Port 8080 -Tag credit-default-api:dev
#>
param(
    [string]$Tag = "credit-default-api:latest",
    [int]$Port = 8000,
    [string]$Name = "credit-default-api",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $model = Join-Path $repoRoot "models\credit_default_model.joblib"
    if (-not (Test-Path $model)) {
        throw "Model artifact not found: $model. Run `python -m src.models.train` (or `dvc repro`) first."
    }

    if (-not $SkipBuild) {
        Write-Host "==> Building image $Tag" -ForegroundColor Cyan
        docker build -t $Tag .
        if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
    }

    Write-Host "==> Removing previous container (if any)" -ForegroundColor Cyan
    # `cmd /c` keeps PowerShell 5.1 from turning stderr into a terminating error
    # when the container does not exist yet.
    cmd /c "docker rm -f $Name >nul 2>&1"

    Write-Host "==> Starting container on port $Port" -ForegroundColor Cyan
    docker run -d --name $Name -p "${Port}:8000" $Tag | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "docker run failed" }

    Write-Host "==> Waiting for the API" -ForegroundColor Cyan
    $pythonCandidates = @(
        (Join-Path $repoRoot ".venv\Scripts\python.exe"),
        (Join-Path (Split-Path -Parent $repoRoot) ".venv\Scripts\python.exe")
    )
    $python = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($python) {
        & $python scripts\smoke_test_api.py --url "http://127.0.0.1:$Port"
    } else {
        Start-Sleep -Seconds 5
        Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" | ConvertTo-Json
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "==> Container logs" -ForegroundColor Yellow
        docker logs $Name
        throw "API smoke test failed"
    }

    Write-Host "==> Service is running: http://127.0.0.1:$Port/docs" -ForegroundColor Green
    Write-Host "    Stop it with: docker rm -f $Name" -ForegroundColor DarkGray
}
finally {
    Pop-Location
}
