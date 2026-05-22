@echo off
REM Kraken Art launcher — double-click to start the app.
REM
REM What this does:
REM   1. Frees TCP 7780 (where the Python sidecar binds) in case a previous
REM      run left a stale process behind.
REM   2. cd's to the project root regardless of where the shortcut was invoked.
REM   3. Runs `npm run tauri dev`, which:
REM        - starts vite on :1420
REM        - builds & runs the Tauri shell (Rust)
REM        - the Tauri shell spawns the Python sidecar on :7780
REM
REM Close the app to stop everything. The Tauri shell tears down both the
REM Python sidecar and the vite dev server on exit.

setlocal
cd /d "%~dp0"

echo [Kraken Art] checking for stale sidecar on port 7780...
for /f "tokens=5" %%P in ('netstat -ano -p TCP ^| findstr ":7780 " ^| findstr "LISTENING"') do (
    echo [Kraken Art] killing stale PID %%P
    taskkill /PID %%P /F >nul 2>&1
)

echo [Kraken Art] launching...
call npm run tauri dev

REM If npm exits non-zero, keep the window open so the error is readable.
if errorlevel 1 (
    echo.
    echo [Kraken Art] exited with error %errorlevel%. Press any key to close.
    pause >nul
)

endlocal
