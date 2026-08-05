$ErrorActionPreference = "Stop"
$PROXY_NAME = "deepseek-proxy"
$PROXY_URL = "http://localhost:4000/health"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR
$VENV_PYTHON = Join-Path $PROJECT_ROOT ".venv-gpu312\Scripts\python.exe"
$DAEMON_SCRIPT = Join-Path $SCRIPT_DIR "proxy_daemon.py"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Claude Code + DeepSeek Proxy Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[0/5] Checking Claude Code credentials..." -ForegroundColor Yellow
$claudeDir = Join-Path $env:USERPROFILE ".claude"
$credFile = Join-Path $claudeDir "credentials.json"
if (-not (Test-Path $claudeDir)) { New-Item -ItemType Directory -Path $claudeDir -Force | Out-Null }
if (-not (Test-Path $credFile)) {
    '{"access_token":"proxy","expires_at":9999999999}' | Out-File -FilePath $credFile -Encoding ascii
    Write-Host "[0/5] Credentials created" -ForegroundColor Green
} else {
    Write-Host "[0/5] Credentials exist" -ForegroundColor Green
}
Write-Host ""

Write-Host "[1/5] Checking for existing proxy..." -ForegroundColor Yellow
& python "$PROJECT_ROOT\bg.py" stop $PROXY_NAME 2>$null
Start-Sleep -Seconds 1

Write-Host "[2/5] Starting DeepSeek Proxy Daemon (waitress + auto-restart)..." -ForegroundColor Yellow
$result = & python "$PROJECT_ROOT\bg.py" start $PROXY_NAME $VENV_PYTHON $DAEMON_SCRIPT "--pid-file" "$PROJECT_ROOT\.bg\$PROXY_NAME.pid" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Proxy daemon failed to start!`n$result" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host $result

Write-Host "[3/5] Waiting for proxy to be ready..." -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 20; $i++) {
    try {
        $response = Invoke-WebRequest -Uri $PROXY_URL -TimeoutSec 1 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "[3/5] Proxy ready" -ForegroundColor Green
            $ready = $true
            break
        }
    } catch {}
    Start-Sleep -Seconds 1
}
if (-not $ready) {
    Write-Host "[ERROR] Proxy startup timed out!" -ForegroundColor Red
    & python "$PROJECT_ROOT\bg.py" stop $PROXY_NAME 2>$null
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[4/5] Setting environment variables..." -ForegroundColor Yellow
$env:ANTHROPIC_BASE_URL = "http://localhost:4000"
$env:ANTHROPIC_API_KEY = "proxy-mode"
Write-Host "[4/5] Done" -ForegroundColor Green
Write-Host ""

Write-Host "[5/5] Launching Claude Code..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    claude
} finally {
    Write-Host ""
    Write-Host "Claude Code exited, stopping proxy..." -ForegroundColor Yellow
    & python "$PROJECT_ROOT\bg.py" stop $PROXY_NAME 2>$null
    Write-Host "Done." -ForegroundColor Green
}
