# Test Log — Audio Engine Unification (sidecar-managed lazy engines)

**Date:** 2026-05-28
**Hardware:** RTX 3090 24 GB, Windows 11
**Backup:** `backup/2026-05-28-1916-pre-audio-engine-unification` (git tag + filesystem snapshot)

Goal: prove the rewired audio architecture behaves as specified —

1. Nothing loads at app launch (engines detected, not running).
2. The picked engine loads on Generate; the other engine is killed first.
3. Switching ACE-Step ↔ Stable Audio 3 leaves only one audio model resident.
4. An image/video job frees the audio engine's VRAM.
5. The GUI/automation talks to one server only (port 7780).

## Pre-flight

- Ports 7780/8001/8010/8021: none listening.
- GPU idle: 1005 MiB / 24576 MiB used.
- `uv` on PATH: yes.

## Results

### Step 1 — nothing loads at launch ✅
`GET /api/audio/engines` right after sidecar start: both engines `available:true, running:false`, `current:null`. No engine ports (8001/8010/8021) listening. GPU 1016 MiB. **Nothing loaded.**

### Lazy load on Generate ✅
Submitting an ACE-Step job spawned the engine: port 8001 bound at **t=46s**, VRAM rose **1.0 → 10.8 GB**, `current=ace_step, running=true`. The engine loads only when Generate is pressed.

### Kill = unload ✅
`POST /api/audio/engines/stop`: ACE killed, VRAM **10.8 GB → 986 MiB**, port 8001 freed. Stopping an engine fully releases its VRAM (no engine cooperation needed).

### Bugs found + fixed during the live run

1. **ACE orchestrator called async client methods un-awaited** — `'coroutine' object has no attribute 'get'`. Pre-existing defect (the sync JobManager worker called `AceClient`'s `async` methods directly). Fixed: wrap in `asyncio.run(...)` (`submit_release_task`, `get_job_status`, `free_memory`), matching the bulk-export worker pattern.
2. **ACE response envelope not unwrapped** — ACE returns `{"data": {...}, "code": 200}`; the orchestrator read `task_id`/progress at the top level. Fixed: unwrap `data` for both submit and poll.
3. **Launcher closed instantly on double-click** — in `:kill_port`, the "already free" message is echoed inside an `if (...)` block; my new STEP 1 labels contained literal parentheses (`"ACE-Step engine (orphan)"`), so the `)` closed the `if` block early → `] was unexpected at this time` on the second port. Surfaced only via double-click (my programmatic sidecar starts bypassed the launcher). Fixed: removed parens from the labels. Verified: STEP 1 now enumerates all four ports cleanly and the script runs to exit 0.

4. **Clear VRAM didn't free Z-Image (or WAN)** — `POST /api/clear_memory` only unloaded `sdxl`/`flux`/`upscale`; it never learned about `z_image` or `wan_video`. After a Z-Image gen, Clear VRAM freed only the ~4.5 GB cache (`freed=4544 MB` in the log) while the ~13 GB transformer stayed resident. Fixed: added `z_image` + `wan_video` to the unload list, plus `audio_engines.stop_all()` so Clear VRAM also releases a running audio engine. **Requires a sidecar restart to take effect** (the running instance has the old code).

5. **404 log spam** — not a product bug. Leftover background poll loops from this verification session (job IDs `3e9628…` and `5bdde376…`) kept polling the restarted sidecar every 5 s; their `until` loops never saw a terminal status on 404 so they span forever. Killed the loops (`TaskStop` + process kill). The app's own pollers already stop on error/use WebSockets, so no frontend change needed.

6. **Music tab layout broken (4 panes in a 3-column grid)** — `.main` is `grid-template-columns: 280px 1fr 340px`. The Music tab rendered App's own `left-system` (GPU/deps/models) **plus** its own three panes (sidebar+center+right) = 4 visible panes → the 4th overflowed ("library under the center panel"). Fixed: App's `left-system` is now `tab-hidden` on the Music tab, so Music's three panes fill the three columns.

7. **Music tab nagged "start Song Studio via the Kraken_Audio launcher"** — directly contradicted the user's "no separate launcher" requirement. Song Studio (8010) backs the library/playlists/catalog/MP3 features and was no longer auto-launched after the rewire. Fixed: new `pipelines/audio/song_studio_service.py` lets the **sidecar lazily auto-start Song Studio** (on loopback, in the ACE venv via `uv run acestep-codex-studio`), wired into the `/audio/library`, `/audio/playlists`, `/audio/song-studio/health` proxy endpoints; stopped on sidecar shutdown. Verified the spawn works (8010 binds, `/api/config` → 200). **Cold boot measured ~2.5 min** (Song Studio loads its workstation models at startup), so `ensure()` timeout is 300 s and the UI messaging now says it auto-starts (no launcher). Removed all "port 8010 / use launcher" nags from the Music tab.

### Pending (verification interrupted; needs an app restart to load backend changes)
- Re-verify **Clear VRAM** now fully unloads Z-Image/WAN (fix in `api/system.py`).
- Confirm the Music tab layout + Song Studio auto-start end-to-end after restart.
- Step 2: ACE-Step full generation producing a playable file (async + envelope fixes in place; not yet re-run to completion).
- Step 3: switch to Stable Audio 3 → confirm ACE killed + SA3 server loads + instrumental WAV produced (SA3 server never run live yet).
- Step 4: image gen after audio frees the audio engine's VRAM (arbiter direction 2).
