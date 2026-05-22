# Dev workflow

## Daily

```powershell
cd 'F:\Kraken Art'
npm run tauri dev
```

That's the only command you need most days. It does:
1. `cargo build` on the Rust host (incremental — sub-second after the first build),
2. `vite` on `localhost:1420` for the frontend with HMR,
3. spawns the Python sidecar on `localhost:7780`,
4. opens the Kraken Art window pointing at the Vite dev URL.

Frontend changes hot-reload. Backend (`python/**`) changes require a sidecar restart.

## Restarting just the sidecar

The Tauri Rust host owns the sidecar lifecycle and only spawns it at app start. If you change Python code (or the sidecar crashes during a heavy load), the in-window sidecar is gone or stale. Options:

**Option A** — close and reopen the Tauri window. Cleanest, but loses your current UI state.

**Option B** — kill the orphaned sidecar and start a new one manually (it will bind to the same port; the React app's `/health` polling reconnects on its own):

```powershell
# kill the existing sidecar
Get-Process python -EA SilentlyContinue |
  Where-Object { (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -EA SilentlyContinue).CommandLine -like '*F:\Kraken Art\python\main.py*' } |
  Stop-Process -Force

# launch a fresh one
& 'F:\Kraken Art\python\venv\Scripts\python.exe' 'F:\Kraken Art\python\main.py'
```

Auto-restart watchdog from the Rust host is a tracked task (#25).

## Running the sidecar alone (no Tauri)

Useful for backend smoke-tests:

```powershell
& 'F:\Kraken Art\python\venv\Scripts\python.exe' 'F:\Kraken Art\python\main.py'
```

Then poke endpoints:

```powershell
Invoke-RestMethod 'http://127.0.0.1:7780/health'
Invoke-RestMethod 'http://127.0.0.1:7780/api/gpu'
Invoke-RestMethod 'http://127.0.0.1:7780/api/models' | ConvertTo-Json -Depth 4
```

Or via `curl` (Git Bash):

```bash
curl http://127.0.0.1:7780/health
curl -X POST http://127.0.0.1:7780/api/clear_memory
curl http://127.0.0.1:7780/api/logs?limit=50
```

## Python environment

```powershell
# from-scratch install (re-runnable)
& 'F:\Kraken Art\python\install.ps1'
```

Or manually:

```powershell
py -3.11 -m venv 'F:\Kraken Art\python\venv'
& 'F:\Kraken Art\python\venv\Scripts\python.exe' -m pip install --upgrade pip
& 'F:\Kraken Art\python\venv\Scripts\python.exe' -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
& 'F:\Kraken Art\python\venv\Scripts\python.exe' -m pip install -r 'F:\Kraken Art\python\requirements.txt'
```

**transformers pinning:** we pin `transformers<5` because diffusers 0.38's single-file loaders still reference the 4.x CLIPTextModel layout. Bumping past 5.0 breaks SDXL loading.

## Logs

Three places:
- The **Logs drawer** in the UI — drag the top edge to resize, double-click the handle to collapse/restore, Copy-all dumps the buffer to clipboard for pasting back.
- `F:\Kraken Art\logs\sidecar-YYYY-MM-DD.log` — daily file, persists across restarts.
- Sidecar stdout (Tauri-spawned: appears prefixed `[sidecar]` in the Rust host's stderr; manual launches: directly in the terminal).

The drawer auto-pops on any ERROR and badges the topbar button with a red count when closed.

## Debugging a crash

If a generate hangs or crashes:

1. Check the **Logs drawer** first — any traceback shows up tagged with the job_id and pipeline name.
2. If the drawer doesn't reach a conclusion (status stuck on "running" with no progress for a long time), the sidecar was likely **SIGKILL'd by the OS** — usually CPU-RAM-OOM during model load. The drawer can't capture that. Look in Task Manager for memory pressure; close other apps; retry with the FP8 variant.
3. The React app detects WebSocket close on a non-terminal job status and snaps back to `idle` with a red message — you can immediately retry without restarting the window.
4. Use **Clear VRAM** before retrying after any failed load to make sure nothing zombie is holding memory.

## Rebuilding Tauri icons

If the logo changes:

```powershell
cd 'F:\Kraken Art'
npm run tauri -- icon 'F:\Kraken Art\Logo\Kraken Logo.png'
```

Generates Windows `.ico`, macOS `.icns`, iOS, and Android variants under `src-tauri/icons/`. Then restart the dev server (the window icon caches between sessions).
