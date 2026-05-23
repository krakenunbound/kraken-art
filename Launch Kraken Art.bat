@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

:: ============================================================================
::  KRAKEN ART - TESTER LAUNCHER (audio integration build)
::  Orchestrates ALL servers needed for a clean end-to-end test:
::    1. Kraken_Audio  ACE-Step API       127.0.0.1:8001  (started by #2)
::    2. Kraken_Audio  Codex Song Studio  127.0.0.1:8010
::    3. Kraken Art    Python sidecar     127.0.0.1:7780
::    4. Tauri dev app (the user-facing UI)
::
::  Pre-flight:  every port + every prior process is forcibly wiped first.
::  Post-flight: same wipe on exit so nothing lingers in memory.
::
::  This is a tester tool, not the production launcher. Logs print verbosely
::  so you can see exactly what's happening.
:: ============================================================================

set "KRAKEN_AUDIO_ROOT=F:\Kraken_Audio"
set "AUDIO_LAUNCHER=%KRAKEN_AUDIO_ROOT%\start_codex_song_studio.bat"
set "AUDIO_STOP_PS1=%KRAKEN_AUDIO_ROOT%\stop_kraken_services.ps1"

set "PORT_SIDECAR=7780"
set "PORT_ACE_API=8001"
set "PORT_SONG_STUDIO=8010"

:: Tell Song Studio to send cover-art jobs to Kraken Art's FLUX1 pipeline
:: (Phase C of the audio integration). Unsetting these reverts it to the
:: legacy ComfyUI path -- here we always want the new path during testing.
set "KRAKEN_COVER_URL=http://127.0.0.1:%PORT_SIDECAR%"
set "KRAKEN_COVER_ARCH=flux1"

echo ============================================================
echo [Kraken Art - TESTER LAUNCHER]
echo  Will orchestrate:
echo    - Kraken Art sidecar             (port %PORT_SIDECAR%)
echo    - Kraken_Audio Song Studio       (port %PORT_SONG_STUDIO%)
echo    - Kraken_Audio ACE-Step API      (port %PORT_ACE_API%)
echo    - Tauri dev app
echo.
echo  Cover-art backend env for Song Studio:
echo    KRAKEN_COVER_URL  = %KRAKEN_COVER_URL%
echo    KRAKEN_COVER_ARCH = %KRAKEN_COVER_ARCH%
echo ============================================================
echo.

:: ============================================
:: STEP 1: PRE-FLIGHT - WIPE EVERY PORT
:: ============================================
echo [1/6] Pre-flight: freeing ports %PORT_SIDECAR% / %PORT_SONG_STUDIO% / %PORT_ACE_API% ...
call :kill_port %PORT_SIDECAR%      "Kraken Art sidecar"
call :kill_port %PORT_SONG_STUDIO%  "Codex Song Studio"
call :kill_port %PORT_ACE_API%      "ACE-Step API"

:: Also run the Kraken_Audio stop script for thoroughness (kills anything
:: whose CommandLine/ExecutablePath is rooted at F:\Kraken_Audio, catching
:: orphans the port-based kill might miss).
if exist "%AUDIO_STOP_PS1%" (
    echo     Running Kraken_Audio orphan-killer script...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%AUDIO_STOP_PS1%"
) else (
    echo     [warn] %AUDIO_STOP_PS1% not found - skipping orphan sweep.
)
echo     Pre-flight complete.
echo.

:: ============================================
:: STEP 2: START KRAKEN_AUDIO STACK
::         (Song Studio launcher handles ACE-Step API itself.)
:: ============================================
echo [2/6] Starting Kraken_Audio stack via %AUDIO_LAUNCHER% ...
if not exist "%AUDIO_LAUNCHER%" (
    echo [ERROR] Audio launcher not found:
    echo         %AUDIO_LAUNCHER%
    echo         Cover-art will fail; Music tab will still load existing library.
    echo         Continuing without Song Studio.
    set "SKIP_AUDIO=1"
) else (
    set "SKIP_AUDIO="
    :: Launch in a separate minimized window so its logs are inspectable
    :: but don't clutter the launcher view. The `start "title"` form
    :: opens a fresh cmd that inherits KRAKEN_COVER_URL/_ARCH from this
    :: environment, which is what Song Studio's run_cover_art_job_worker
    :: reads to know whether to use Kraken Art.
    start "KrakenAudio-Stack" /min cmd /k "call ""%AUDIO_LAUNCHER%"""
)

if not defined SKIP_AUDIO (
    echo     Waiting for Song Studio to bind port %PORT_SONG_STUDIO% (max 180s) ...
    call :wait_port %PORT_SONG_STUDIO% 180 || (
        echo [ERROR] Song Studio never bound %PORT_SONG_STUDIO%.
        echo         Check the "KrakenAudio-Stack" window for the real error.
        echo         You can still test Phase B (library view); cover-art and
        echo         MP3 export will fail until Song Studio is up.
    )
)
echo.

:: ============================================
:: STEP 3: START KRAKEN ART SIDECAR
:: ============================================
echo [3/6] Starting Kraken Art sidecar (port %PORT_SIDECAR%) ...
start "KrakenArt-Sidecar" /min "python\venv\Scripts\python.exe" "python\main.py"
echo     Waiting for sidecar to bind port %PORT_SIDECAR% (max 90s) ...
call :wait_port %PORT_SIDECAR% 90 || (
    echo [ERROR] Sidecar never bound %PORT_SIDECAR%. Check "KrakenArt-Sidecar" window.
    pause
    goto :cleanup
)
echo.

:: ============================================
:: STEP 4: SUMMARY BEFORE TAURI
:: ============================================
echo [4/6] All servers up. Summary:
echo     - Kraken Art sidecar : http://127.0.0.1:%PORT_SIDECAR%/health
if not defined SKIP_AUDIO (
    echo     - Codex Song Studio  : http://127.0.0.1:%PORT_SONG_STUDIO%/api/config
    echo     - ACE-Step API       : http://127.0.0.1:%PORT_ACE_API%/health
)
echo.

:: ============================================
:: STEP 5: LAUNCH TAURI DEV (blocks until closed)
:: ============================================
echo [5/6] Launching Tauri dev. Close the Kraken Art window to begin cleanup.
echo.
call npm run tauri dev

:: ============================================
:: STEP 6: CLEANUP (always runs, no prompt)
:: ============================================
:cleanup
echo.
echo ============================================================
echo [6/6] Tauri dev session ended. Wiping all servers from memory ...
echo ============================================================

call :kill_port %PORT_SIDECAR%      "Kraken Art sidecar"
call :kill_port %PORT_SONG_STUDIO%  "Codex Song Studio"
call :kill_port %PORT_ACE_API%      "ACE-Step API"

:: Belt + suspenders: run the Kraken_Audio orphan sweeper too, which
:: catches anything whose ports we missed but whose process tree is
:: rooted at F:\Kraken_Audio (uv subprocesses, model-loader threads, ...).
if exist "%AUDIO_STOP_PS1%" (
    echo     Running Kraken_Audio orphan-killer script...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%AUDIO_STOP_PS1%"
)

:: Also close the minimized helper windows by title so they don't sit
:: there showing dead Python prompts.
taskkill /FI "WINDOWTITLE eq KrakenArt-Sidecar*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq KrakenAudio-Stack*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq ACE-Step API*"      /F >nul 2>&1

echo.
echo     All ports verified free:
call :verify_port_free %PORT_SIDECAR%      "Kraken Art sidecar"
call :verify_port_free %PORT_SONG_STUDIO%  "Codex Song Studio"
call :verify_port_free %PORT_ACE_API%      "ACE-Step API"

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
    echo     Killing PID %%P on %_KP_PORT% (%_KP%) ...
    taskkill /PID %%P /F /T >nul 2>&1
    set "_KP_KILLED=1"
)
if "%_KP_KILLED%"=="0" (
    echo     Port %_KP_PORT% (%_KP%) was already free.
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
