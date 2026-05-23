@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ============================================================
echo [Kraken Art - TESTER LAUNCHER]
echo This is a disposable tester tool only. Not for production.
echo ============================================================
echo.

:: ============================================
:: 1. AGGRESSIVELY FREE THE PORT
:: ============================================
echo [1/5] Killing anything currently on port 7780...
for /f "tokens=5" %%P in ('netstat -ano -p TCP ^| findstr ":7780 " ^| findstr "LISTENING" 2^>nul') do (
    echo     Found listener on PID %%P - killing...
    taskkill /PID %%P /F >nul 2>&1
)
timeout /t 2 >nul

:: Double-check
netstat -ano | findstr ":7780 " | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo [WARNING] Something is still on 7780 after kill attempt.
) else (
    echo     Port 7780 is now free.
)

:: ============================================
:: 2. START THE CORRECT VENV SIDECAR
:: ============================================
echo.
echo [2/5] Starting the CORRECT venv sidecar (hidden window)...
start "KrakenArt-Sidecar" /min "python\venv\Scripts\python.exe" "python\main.py"

:: ============================================
:: 3. WAIT FOR THE SIDECAR TO ACTUALLY LISTEN
:: ============================================
echo.
echo [3/5] Waiting for sidecar to bind port 7780 (max 90 seconds)...
set /a waited=0
set /a maxwait=90

:wait_for_port
netstat -ano | findstr ":7780 " | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo     Sidecar is now listening on 7780.
    goto :port_ready
)

timeout /t 2 >nul
set /a waited+=2
echo     ... waiting (%waited%s / %maxwait%s)
if %waited% LSS %maxwait% goto :wait_for_port

echo [ERROR] Sidecar never bound to 7780 after %maxwait% seconds.
echo         Check the "KrakenArt-Sidecar" window for errors.
pause
exit /b 1

:port_ready

:: ============================================
:: 4. LAUNCH THE TAURI DEV APP
:: ============================================
echo.
echo [4/5] Sidecar is healthy. Launching Tauri dev...
echo.
call npm run tauri dev

:: ============================================
:: 5. CLEANUP ON EXIT
:: ============================================
echo.
echo ============================================================
echo [5/5] Tauri dev session has ended.
echo.
echo Do you want to kill the sidecar now? (recommended for clean testing)
echo Press Y to kill it, or N to leave it running for faster restarts.
echo ============================================================
choice /c YN /n /m "Kill sidecar now? [Y/N]"

if errorlevel 2 (
    echo Sidecar left running (you can kill it manually later).
) else (
    echo Killing sidecar...
    for /f "tokens=5" %%P in ('netstat -ano -p TCP ^| findstr ":7780 " ^| findstr "LISTENING" 2^>nul') do (
        taskkill /PID %%P /F >nul 2>&1
    )
    echo Sidecar terminated.
)

echo.
echo [Kraken Art - TESTER LAUNCHER] Finished.
endlocal
