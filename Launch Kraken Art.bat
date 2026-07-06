@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

:: ============================================================================
::  KRAKEN ART - LAUNCHER
::  Starts ONE server (the Kraken Art sidecar) and the Tauri UI.
::
::  Audio is no longer launched here. The sidecar (port 7780) now owns the
::  lifecycle of the audio engines (ACE-Step / Stable Audio 3): it starts the
::  picked engine on demand when you press Generate and kills it when you
::  switch engines or do image/video work. The GUI only ever talks to 7780 —
::  no second window, no second port to think about. The engine-port wipes
::  below are just a safety net for orphaned subprocesses.
:: ============================================================================

set "PORT_SIDECAR=7780"
set "PORT_ACE_API=8001"
set "PORT_SONG_STUDIO=8010"
set "PORT_SA3=8021"

:: Tell Song Studio to send cover-art jobs to Kraken Art's FLUX1 pipeline
:: (Phase C of the audio integration). Unsetting these reverts it to the
:: legacy ComfyUI path -- here we always want the new path during testing.
set "KRAKEN_COVER_URL=http://127.0.0.1:%PORT_SIDECAR%"
set "KRAKEN_COVER_ARCH=flux1"

echo ============================================================
echo [Kraken Art - LAUNCHER]
echo  Will start:
echo    - Kraken Art sidecar             (port %PORT_SIDECAR%)  ^<- the only server
echo    - Tauri dev app
echo.
echo  Audio engines (ACE-Step / Stable Audio 3) start on demand,
echo  managed by the sidecar. No separate audio window or port.
echo ============================================================
echo.

:: ============================================
:: STEP 1: PRE-FLIGHT - WIPE PORTS (sidecar + any orphaned audio engines)
:: ============================================
echo [1/4] Pre-flight: freeing ports ...
call :kill_port %PORT_SIDECAR%      "Kraken Art sidecar"
call :kill_port %PORT_ACE_API%      "ACE-Step engine orphan"
call :kill_port %PORT_SONG_STUDIO%  "Song Studio orphan"
call :kill_port %PORT_SA3%          "Stable Audio 3 engine orphan"
echo.

:: ============================================
:: STEP 2: (sidecar is owned by Tauri, not started here)
:: ============================================
:: IMPORTANT: the Tauri host (src-tauri/src/lib.rs) spawns AND supervises the
:: Python sidecar itself. Starting a second one here used to create a duplicate
:: sidecar -> two CUDA contexts fighting for VRAM, which can tip Ideogram's
:: large NF4 load into a hard crash on a shared card. So we do NOT pre-start it;
:: `npm run tauri dev` brings up the single, supervised sidecar.
echo [2/4] Sidecar is launched + supervised by the Tauri host (single instance).
echo.

:: ============================================
:: STEP 3: LAUNCH TAURI DEV (blocks until closed; it spawns the sidecar)
:: ============================================
echo [3/4] Launching Tauri dev (this brings up the one sidecar on port %PORT_SIDECAR%).
echo       Close the Kraken Art window to begin cleanup.
echo.
call npm run tauri dev

:: ============================================
:: STEP 5: CLEANUP (always runs, no prompt)
:: ============================================
:cleanup
echo.
echo ============================================================
echo [4/4] Tauri dev session ended. Wiping all servers from memory ...
echo ============================================================

call :kill_port %PORT_SIDECAR%      "Kraken Art sidecar"
:: The sidecar spawns audio engines as child processes; wipe their ports too in
:: case any orphaned when the sidecar exited.
call :kill_port %PORT_ACE_API%      "ACE-Step engine"
call :kill_port %PORT_SONG_STUDIO%  "Song Studio"
call :kill_port %PORT_SA3%          "Stable Audio 3 engine"
taskkill /FI "WINDOWTITLE eq KrakenArt-Sidecar*" /F >nul 2>&1

echo.
echo     All ports verified free:
call :verify_port_free %PORT_SIDECAR%      "Kraken Art sidecar"
call :verify_port_free %PORT_ACE_API%      "ACE-Step engine"
call :verify_port_free %PORT_SA3%          "Stable Audio 3 engine"

echo.
echo [Kraken Art - TESTER LAUNCHER] Finished. All servers wiped.
endlocal
exit /b 0


:: ============================================================================
::  HELPERS
:: ============================================================================

:kill_port
:: %1 = port number, %2 = friendly label (quoted)
set "_KP=%~2"
set "_KP_PORT=%~1"
set "_KP_KILLED=0"
for /f "tokens=5" %%P in ('netstat -ano -p TCP ^| findstr ":%_KP_PORT% " ^| findstr "LISTENING" 2^>nul') do (
    echo     Killing PID %%P on %_KP_PORT% [%_KP%] ...
    taskkill /PID %%P /F /T >nul 2>&1
    set "_KP_KILLED=1"
)
if "%_KP_KILLED%"=="0" (
    echo     Port %_KP_PORT% [%_KP%] was already free.
)
:: Brief settle so the next bind doesn't race a half-released socket.
timeout /t 1 >nul
exit /b 0

:wait_port
:: %1 = port, %2 = max seconds. Returns 0 when port is LISTENING, 1 on timeout.
set "_WP_PORT=%~1"
set "_WP_MAX=%~2"
set /a _WP_WAITED=0
:wait_port_loop
netstat -ano | findstr ":%_WP_PORT% " | findstr "LISTENING" >nul
if !errorlevel!==0 (
    echo     Port %_WP_PORT% is now LISTENING.
    exit /b 0
)
timeout /t 2 >nul
set /a _WP_WAITED+=2
echo     ... waiting on %_WP_PORT% (%_WP_WAITED%s / %_WP_MAX%s)
if !_WP_WAITED! LSS !_WP_MAX! goto :wait_port_loop
exit /b 1

:verify_port_free
:: %1 = port, %2 = label. Print one line either way.
set "_VP=%~2"
netstat -ano | findstr ":%~1 " | findstr "LISTENING" >nul
if !errorlevel!==0 (
    echo       [%_VP%] still LISTENING on %~1 -- you may need to kill it manually.
) else (
    echo       [%_VP%] port %~1 free.
)
exit /b 0
