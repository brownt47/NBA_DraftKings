@echo off
setlocal

if "%~1"=="" (
    echo Usage: show filename
    exit /b 1
)

if not exist "%~1" (
    echo File not found: %~1
    exit /b 1
)

set "SHOW_FILE=%~f1"

powershell -NoProfile -Command "$content = Get-Content -LiteralPath $env:SHOW_FILE -Raw; [Console]::Write($content); Set-Clipboard -Value $content"

endlocal