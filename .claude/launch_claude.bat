@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

set "PROXY_NAME=deepseek-proxy"
set "PROXY_URL=http://localhost:4000/health"
set "VENV_PYTHON=%~dp0..\.venv-gpu312\Scripts\python.exe"

echo ========================================
echo   Claude Code + DeepSeek Proxy Launcher
echo ========================================
echo.

REM 0. Ensure credentials
echo [0/5] Checking Claude Code credentials...
if not exist "%USERPROFILE%\.claude" mkdir "%USERPROFILE%\.claude"
if not exist "%USERPROFILE%\.claude\credentials.json" (
    echo {"access_token":"proxy","expires_at":9999999999} > "%USERPROFILE%\.claude\credentials.json"
    echo [0/5] Credentials created
) else (
    echo [0/5] Credentials exist
)
echo.

REM 1. Kill old proxy if running
echo [1/5] Checking for existing proxy...
python bg.py stop %PROXY_NAME% >nul 2>&1
timeout /t 1 /nobreak >nul

REM 2. Start proxy daemon via bg.py
echo [2/5] Starting DeepSeek Proxy Daemon (waitress + auto-restart)...
python bg.py start %PROXY_NAME% "%VENV_PYTHON%" "%~dp0proxy_daemon.py" --pid-file "%~dp0..\.bg\%PROXY_NAME%.pid"
if errorlevel 1 (
    echo [ERROR] Proxy daemon failed to start!
    pause
    exit /b 1
)

REM 3. Wait for health check
echo [3/5] Waiting for proxy to be ready...
for /l %%i in (1,1,20) do (
    curl -s -o nul -w "%%{http_code}" %PROXY_URL% 2>nul | findstr "200" >nul
    if not errorlevel 1 (
        echo [3/5] Proxy ready
        goto :proxy_ready
    )
    timeout /t 1 /nobreak >nul
)
echo [ERROR] Proxy startup timed out!
python bg.py stop %PROXY_NAME% >nul 2>&1
pause
exit /b 1

:proxy_ready

REM 4. Set env vars
echo [4/5] Setting environment variables...
set ANTHROPIC_BASE_URL=http://localhost:4000
set ANTHROPIC_API_KEY=proxy-mode
echo [4/5] Done
echo.

REM 5. Launch Claude Code
echo [5/5] Launching Claude Code...
echo ========================================
echo.
claude

REM 6. Always cleanup
echo.
echo Claude Code exited, stopping proxy...
python bg.py stop %PROXY_NAME%
echo Done.
