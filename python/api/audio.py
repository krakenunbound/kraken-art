"""Audio (music/voice) API surface for Kraken Art.

These endpoints let the React UI drive the external ACE-Step engine
without the React frontend ever needing to know the ACE port or deal
with CORS / auth differences.

All heavy lifting stays in the already-finished Kraken_Audio / ACE-Step
installation. We only orchestrate and (later) inject album art generated
by Kraken Art's own image pipelines.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

import asyncio
import httpx
import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from jobs import manager
from pipelines.audio import get_default_ace_client
from pipelines.audio.orchestrator import run as run_audio_job
from pipelines.audio.song_studio_client import get_default_song_studio_client
from pipelines.audio import engine_manager
from pipelines.audio.engine_manager import manager as audio_engines
from pipelines.audio import local_providers
from pipelines.audio import song_studio_service
from pipelines.audio.tts_client import get_default_tts_client
from config import OUTPUTS_ROOT, AUDIO_MODELS_ROOT, AUDIO_PROVIDERS_ROOT

router = APIRouter(prefix="/audio", tags=["audio"])

log = logging.getLogger("kraken.audio.api")


# ---- VRAM arbiter (direction 1): free image/video pipelines before an audio
# engine loads. Registered as the EngineManager's pre-start hook so the picked
# audio model never has to fight a resident FLUX/SDXL/WAN for VRAM. The reverse
# direction (image/video jobs stop the audio engine) lives in api/generate.py.
def _unload_image_video_for_audio(engine_id: str) -> None:
    try:
        from pipelines import flux, ideogram4, sdxl, z_image, wan_video
        for mod in (flux, sdxl, z_image, ideogram4, wan_video):
            fn = getattr(mod, "unload", None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:
                    log.debug("unload(%s) raised: %s", getattr(mod, "__name__", mod), e)
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        log.info("VRAM arbiter: image/video pipelines unloaded before audio engine %s", engine_id)
    except Exception as e:
        log.warning("VRAM arbiter pre-start hook failed: %s", e)


engine_manager.on_before_engine_start = _unload_image_video_for_audio


async def _ensure_song_studio() -> None:
    """Auto-start the Song Studio workstation (library/catalog/MP3) on demand.

    Replaces the old 'start it via the Kraken_Audio launcher' requirement — the
    sidecar owns its lifecycle now, so the Music tab's library/catalog just work
    without a separate window. Runs the blocking spawn+health-wait off the loop."""
    await asyncio.to_thread(song_studio_service.ensure)


class AudioHealthResponse(BaseModel):
    ok: bool
    detail: dict | str | None = None


@router.get("/health", response_model=AudioHealthResponse)
async def audio_health():
    """Proxy health check to the running ACE-Step API (usually port 8001)."""
    client = get_default_ace_client()
    try:
        data = await client.health()
        return {"ok": True, "detail": data}
    except Exception as e:
        return {"ok": False, "detail": str(e)}


@router.get("/engines")
def audio_engines_list() -> dict:
    """Inventory of audio generation engines for the Music-tab selector.

    `available` = the engine's files are on disk (NOT loaded). `running` = its
    process is alive. Nothing here loads a model — that happens on Generate."""
    return {"engines": audio_engines.list(), "current": audio_engines.current()}


@router.post("/engines/stop")
def audio_engines_stop() -> dict:
    """Stop any running audio engine (releases its VRAM). Used when leaving the
    Music tab or before heavy image/video work."""
    audio_engines.stop_all()
    return {"ok": True, "current": audio_engines.current()}


@router.get("/models")
async def audio_models():
    """Return the model inventory reported by the live ACE service."""
    client = get_default_ace_client()
    try:
        data = await client.list_models()
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ACE service unreachable: {e}")


# ---------- Real generation (the thing the user clicks "Generate" for) ----------

class AudioGenerateRequest(BaseModel):
    """Payload from the Music tab form."""
    engine_id: str = "ace_step"           # "ace_step" (vocals) | "stable_audio_3" (instrumental)
    prompt: str = ""
    lyrics: str = ""
    ace_model: str | None = None          # which DiT the user picked from the live list
    bpm: int = 120
    key_scale: str = "C"
    duration: int = 60                    # seconds
    steps: int = 8                        # used by Stable Audio 3
    temperature: float = 0.85
    generate_cover: bool = True
    cover_prompt: str | None = None       # optional override for the cover art prompt
    thinking: bool = False
    sample_mode: bool = False


class AudioProviderStatus(BaseModel):
    id: str
    capability: str
    label: str
    available: bool
    installed: bool
    recommended: bool = False
    notes: str = ""
    install_hint: str | None = None


@router.get("/providers")
def audio_providers() -> dict:
    engines = {e["id"]: e for e in audio_engines.list()}
    dia_installed = (
        (AUDIO_PROVIDERS_ROOT / "dia" / ".venv" / "Scripts" / "python.exe").exists()
        and (AUDIO_MODELS_ROOT / "dia" / "Dia-1.6B-0626" / "dia-v1.pth").exists()
    )
    providers = [
        AudioProviderStatus(
            id="moss_tts",
            capability="speech",
            label="MOSS-TTS Local v1.5",
            available=bool(engines.get("moss_tts", {}).get("available")),
            installed=bool(engines.get("moss_tts", {}).get("available")),
            recommended=True,
            notes="Installed 48 kHz stereo speech path. Best main open speech model; LuxTTS remains the custom voice-library fallback.",
        ),
        AudioProviderStatus(
            id="tts_luxtts",
            capability="speech_dialogue",
            label="LuxTTS",
            available=bool(engines.get("tts_luxtts", {}).get("available")),
            installed=bool(engines.get("tts_luxtts", {}).get("available")),
            recommended=True,
            notes="Installed local voice cloning path when LuxTTS, LinaCodec, and the ACE-Step venv are present.",
            install_hint="Already expected under F:\\Kraken_Audio\\LuxTTS and F:\\Kraken_Audio\\LinaCodec.",
        ),
        AudioProviderStatus(
            id="moss_sfx",
            capability="sfx",
            label="MOSS-SoundEffect v2.0",
            available=bool(engines.get("moss_sfx", {}).get("available")),
            installed=bool(engines.get("moss_sfx", {}).get("available")),
            recommended=True,
            notes="Installed specialist provider for nature, ambience, action, fantasy, and Foley-style sound effects.",
        ),
        AudioProviderStatus(
            id="stable_audio_3",
            capability="sfx_ambience_instrumental",
            label="Stable Audio 3 Medium",
            available=bool(engines.get("stable_audio_3", {}).get("available")),
            installed=bool(engines.get("stable_audio_3", {}).get("available")),
            recommended=True,
            notes="Best installed local path for ambience, SFX, and instrumental music.",
        ),
        AudioProviderStatus(
            id="ace_step",
            capability="song_vocals",
            label="ACE-Step 1.5",
            available=bool(engines.get("ace_step", {}).get("available")),
            installed=bool(engines.get("ace_step", {}).get("available")),
            recommended=True,
            notes="Best installed local path for songs with lyrics and vocals.",
        ),
        AudioProviderStatus(
            id="heartmula",
            capability="song_vocals",
            label="HeartMuLa 3B",
            available=bool(engines.get("heartmula", {}).get("available")),
            installed=bool(engines.get("heartmula", {}).get("available")),
            recommended=True,
            notes="Installed secondary song-with-vocals engine using lazy-load for the RTX 3090.",
        ),
        AudioProviderStatus(
            id="dia",
            capability="dialogue",
            label="Dia",
            available=dia_installed,
            installed=dia_installed,
            recommended=False,
            notes="Optional experiment. Installed, but not a primary route for this app unless you decide you want one-pass tagged dialogue.",
        ),
    ]
    return {"providers": [p.model_dump() for p in providers]}


async def _ensure_tts() -> None:
    await asyncio.to_thread(audio_engines.ensure, "tts_luxtts")


@router.get("/tts/health")
async def tts_health() -> dict:
    try:
        await _ensure_tts()
        cfg = await get_default_tts_client().config()
        return {"ok": True, "config": cfg}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/tts/voices")
async def tts_voices() -> dict:
    try:
        await _ensure_tts()
        return await get_default_tts_client().voices()
    except Exception as e:
        raise HTTPException(502, f"TTS service unavailable: {e}")


@router.post("/tts/voices")
async def upload_tts_voice(
    name: str = Form(...),
    ref_text: str = Form(""),
    audio: UploadFile = File(...),
) -> dict:
    suffix = Path(audio.filename or "voice.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = await audio.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
    try:
        await _ensure_tts()
        return await get_default_tts_client().upload_voice(name, tmp_path, ref_text)
    except Exception as e:
        raise HTTPException(502, f"Voice upload failed: {e}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


class SpeechGenerateRequest(BaseModel):
    text: str
    voice_id: str
    provider: str = "tts_luxtts"
    speed: float = 1.0
    num_steps: int = 4
    t_shift: float = 0.5
    ref_duration: float = 5.0


@router.post("/speech/generate")
async def generate_speech(req: SpeechGenerateRequest) -> dict:
    if not req.text.strip():
        raise HTTPException(400, "text is required")
    if req.provider == "moss_tts":
        try:
            data = await asyncio.to_thread(local_providers.generate_moss_tts, req.text)
            data["audio_url"] = data.get("path")
            return {"ok": True, "kind": "speech", "provider": req.provider, "result": data}
        except Exception as e:
            raise HTTPException(502, f"MOSS-TTS generation failed: {e}")
    if req.provider != "tts_luxtts":
        raise HTTPException(400, f"provider {req.provider!r} is not installed yet")
    try:
        await _ensure_tts()
        data = await get_default_tts_client().generate(
            text=req.text,
            voice=req.voice_id,
            speed=req.speed,
            num_steps=req.num_steps,
            t_shift=req.t_shift,
            ref_duration=req.ref_duration,
        )
        return {"ok": True, "kind": "speech", "provider": req.provider, "result": data}
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text[:1000])
    except Exception as e:
        raise HTTPException(502, f"Speech generation failed: {e}")


class DialogueGenerateRequest(BaseModel):
    script: str
    speakers: dict[str, str]
    provider: str = "tts_luxtts"
    speed: float = 1.0
    num_steps: int = 4
    t_shift: float = 0.5
    ref_duration: float = 5.0
    gap_seconds: float = 0.35


_ROLE_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.+?)\s*$")


def _parse_dialogue_script(script: str) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for raw in script.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = _ROLE_RE.match(raw)
        if not m:
            raise HTTPException(400, f"Dialogue line must start with [Role]: {raw[:80]}")
        lines.append((m.group(1).strip(), m.group(2).strip()))
    if not lines:
        raise HTTPException(400, "script contains no dialogue lines")
    return lines


async def _download_tts_output(url: str, dest: Path) -> None:
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        dest.write_bytes(r.content)


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg was not found on PATH")
    return exe


def _write_silence(path: Path, seconds: float) -> None:
    subprocess.run(
        [_ffmpeg(), "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", str(max(0.05, seconds)), str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def _concat_wavs(inputs: list[Path], output: Path) -> None:
    list_file = output.with_suffix(".concat.txt")
    list_file.write_text(
        "".join(f"file '{str(p).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for p in inputs),
        encoding="utf-8",
    )
    subprocess.run(
        [_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(output)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


@router.post("/dialogue/generate")
async def generate_dialogue(req: DialogueGenerateRequest) -> dict:
    if req.provider != "tts_luxtts":
        raise HTTPException(400, f"provider {req.provider!r} is not installed yet")
    lines = _parse_dialogue_script(req.script)
    missing = sorted({role for role, _ in lines if role not in req.speakers})
    if missing:
        raise HTTPException(400, f"Missing voice mapping for: {', '.join(missing)}")

    await _ensure_tts()
    client = get_default_tts_client()
    out_dir = OUTPUTS_ROOT / "audio" / "dialogue" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    manifest: list[dict] = []
    try:
        silence = out_dir / "gap.wav"
        _write_silence(silence, req.gap_seconds)
        for idx, (role, text) in enumerate(lines, 1):
            data = await client.generate(
                text=text,
                voice=req.speakers[role],
                speed=req.speed,
                num_steps=req.num_steps,
                t_shift=req.t_shift,
                ref_duration=req.ref_duration,
            )
            url = data.get("audio_url")
            if not url:
                raise RuntimeError(f"TTS response for line {idx} did not include an audio URL")
            clip = out_dir / f"{idx:03d}-{re.sub(r'[^A-Za-z0-9_-]+', '_', role).strip('_') or 'role'}.wav"
            await _download_tts_output(url, clip)
            generated.append(clip)
            if idx != len(lines):
                generated.append(silence)
            manifest.append({"index": idx, "role": role, "text": text, "voice_id": req.speakers[role], "clip": str(clip)})
        final = out_dir / "dialogue_mix.wav"
        _concat_wavs(generated, final)
        (out_dir / "manifest.json").write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
        return {
            "ok": True,
            "kind": "dialogue",
            "provider": req.provider,
            "path": str(final),
            "clips": manifest,
            "line_count": len(lines),
        }
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text[:1000])
    except Exception as e:
        raise HTTPException(500, f"Dialogue generation failed: {e}")


@router.post("/generate")
def generate_audio(req: AudioGenerateRequest) -> dict:
    """Create an audio job.

    The job will (optionally) first generate a beautiful album cover using
    Kraken Art's own FLUX/SDXL pipelines, then submit the request to the
    user's already-running ACE-Step engine, poll for completion, and release
    VRAM on the ACE side when done.
    """
    payload = req.model_dump()
    job = manager.submit("audio", payload, run_audio_job)
    return {"job_id": job.id, "status": job.status, "kind": "audio"}


@router.get("/jobs/{job_id}")
def get_audio_job(job_id: str) -> dict:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.snapshot()


@router.post("/jobs/{job_id}/cancel")
def cancel_audio_job(job_id: str) -> dict:
    ok = manager.cancel(job_id)
    if not ok:
        raise HTTPException(409, "job not cancellable")
    return {"job_id": job_id, "status": "cancel_requested"}


@router.post("/free")
async def free_ace_vram():
    """Explicitly ask the external ACE service to unload its models (symmetric release)."""
    client = get_default_ace_client()
    try:
        data = await client.free_memory()
        return {"ok": True, "detail": data}
    except Exception as e:
        return {"ok": False, "detail": str(e)}


# ---------- Audio file proxy (so the UI can play files the ACE engine produced) ----------


@router.get("/file")
async def proxy_audio_file(path: str):
    """Stream an audio file from the ACE engine's output area.

    The ACE service already has a secure /v1/audio?path=... route with directory checks.
    We just forward the bytes so the Kraken Art webview can <audio src="..."> without CORS headaches.
    """
    client = get_default_ace_client()
    # We reuse the same base URL the client has
    target = f"{client.base_url}/v1/audio?path={path}"
    # For simplicity we fetch and re-stream (the files are not huge for preview).
    # A true streaming proxy would use httpx streaming, but this is good enough for v1.
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.get(target, headers=client._headers())
        if r.status_code != 200:
            raise HTTPException(r.status_code, "ACE refused to serve the file")
        media = r.headers.get("content-type", "audio/mpeg")
        return StreamingResponse(r.iter_bytes(), media_type=media)


@router.get("/local-file")
async def stream_local_audio_file(path: str):
    """Stream a local audio artifact created by Kraken Art audio workflows.

    This intentionally only serves files under OUTPUTS_ROOT so arbitrary local
    paths cannot be exposed through the sidecar.
    """
    try:
        p = Path(path).resolve()
        root = OUTPUTS_ROOT.resolve()
        if root not in p.parents and p != root:
            raise HTTPException(403, "path is outside Kraken Art outputs")
        if not p.exists() or not p.is_file():
            raise HTTPException(404, "audio file not found")
        suffix = p.suffix.lower()
        media = "audio/wav" if suffix == ".wav" else "audio/mpeg" if suffix == ".mp3" else "application/octet-stream"
        return FileResponse(p, media_type=media, filename=p.name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


# =============================================================================
# Song Studio proxy (port 8010) — the user-facing library/playlist/workspace API
# =============================================================================
#
# The bare ACE-Step API (port 8001, proxied above) only handles raw audio jobs.
# Kraken_Audio's Codex Song Studio (port 8010, codex_song_studio.py) is the
# real workstation: persistent library, workspaces, playlists, song metadata,
# MP3 download with embedded cover, lyric sync, etc.
#
# These routes thinly forward to it so the React UI only talks to Kraken Art's
# port 7780. The user never sees the 8010 URL.


@router.get("/song-studio/health")
async def song_studio_health():
    """Combined Song Studio probe + model catalog. Returns the full /api/config
    payload so the UI can render the dropdown (`generationModels`) AND show
    health indicators (assistantReady, voiceCloneReady, coverArtStatus)."""
    client = get_default_song_studio_client()
    try:
        await _ensure_song_studio()
        cfg = await client.config()
        return {"ok": True, "base_url": client.base_url, "config": cfg}
    except Exception as e:
        return {"ok": False, "base_url": client.base_url, "error": str(e)}


@router.get("/library")
async def song_library():
    """Whole library: {songs: [...]}. Returned as-is so React can render the
    Suno-style grid. Heavy call — Song Studio scans the filesystem."""
    client = get_default_song_studio_client()
    try:
        await _ensure_song_studio()
        return await client.library()
    except httpx.RequestError as e:
        raise HTTPException(502, f"Song Studio unreachable: {e}")
    except RuntimeError as e:
        raise HTTPException(502, f"Song Studio could not start: {e}")


# NOTE: there's intentionally no GET /songs/{id} here. Song Studio only ships
# the full /api/library payload (which already includes every song's prompt,
# lyrics, bpm, key, cover path, etc.) plus bulk endpoints. The React UI loads
# the library once and looks up by id from its in-memory copy.


@router.delete("/songs/{song_id}")
async def delete_song(song_id: str):
    client = get_default_song_studio_client()
    try:
        return await client.delete_song(song_id)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text[:500])
    except httpx.RequestError as e:
        raise HTTPException(502, f"Song Studio unreachable: {e}")


class BulkDeleteRequest(BaseModel):
    songIds: list[str]


@router.post("/songs/bulk-delete")
async def bulk_delete_songs(req: BulkDeleteRequest):
    client = get_default_song_studio_client()
    try:
        return await client.bulk_delete(req.songIds)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text[:500])
    except httpx.RequestError as e:
        raise HTTPException(502, f"Song Studio unreachable: {e}")


@router.get("/playlists")
async def list_playlists():
    client = get_default_song_studio_client()
    try:
        await _ensure_song_studio()
        return await client.playlists()
    except httpx.RequestError as e:
        raise HTTPException(502, f"Song Studio unreachable: {e}")
    except RuntimeError as e:
        raise HTTPException(502, f"Song Studio could not start: {e}")


class CreatePlaylistRequest(BaseModel):
    title: str


@router.post("/playlists")
async def create_playlist(req: CreatePlaylistRequest):
    client = get_default_song_studio_client()
    try:
        return await client.create_playlist(req.title)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text[:500])
    except httpx.RequestError as e:
        raise HTTPException(502, f"Song Studio unreachable: {e}")


class AddSongsToPlaylistRequest(BaseModel):
    songIds: list[str]


@router.post("/playlists/{playlist_id}/songs")
async def add_to_playlist(playlist_id: str, req: AddSongsToPlaylistRequest):
    client = get_default_song_studio_client()
    try:
        return await client.add_songs_to_playlist(playlist_id, req.songIds)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text[:500])
    except httpx.RequestError as e:
        raise HTTPException(502, f"Song Studio unreachable: {e}")


class CreateWorkspaceRequest(BaseModel):
    title: str


@router.post("/workspaces")
async def create_workspace(req: CreateWorkspaceRequest):
    client = get_default_song_studio_client()
    try:
        return await client.create_workspace(req.title)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text[:500])
    except httpx.RequestError as e:
        raise HTTPException(502, f"Song Studio unreachable: {e}")


class RenameWorkspaceRequest(BaseModel):
    title: str


@router.patch("/workspaces/{workspace_id}")
async def rename_workspace(workspace_id: str, req: RenameWorkspaceRequest):
    client = get_default_song_studio_client()
    try:
        return await client.rename_workspace(workspace_id, req.title)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text[:500])
    except httpx.RequestError as e:
        raise HTTPException(502, f"Song Studio unreachable: {e}")


# ---- Audio stream + MP3 download (kept separate from /file which serves the
# bare ACE folder; these go through Song Studio's library-aware endpoints) ----

@router.get("/stream")
async def stream_song_audio(path: str):
    """Stream a Song Studio audio file. Used by <audio src=...> in the React
    player. The Song Studio's own /api/audio has directory-traversal guards."""
    client = get_default_song_studio_client()
    target = client.audio_url(path)
    async with httpx.AsyncClient(timeout=120.0) as c:
        try:
            r = await c.get(target)
        except httpx.RequestError as e:
            raise HTTPException(502, f"Song Studio unreachable: {e}")
        if r.status_code != 200:
            raise HTTPException(r.status_code, "Song Studio refused the file")
        media = r.headers.get("content-type", "audio/mpeg")
        return StreamingResponse(r.iter_bytes(), media_type=media)


@router.get("/songs/{song_id}/download")
async def download_song_mp3(song_id: str):
    """Triggers a browser download for the song's exported MP3 (Song Studio
    bakes in embedded cover art). Returns the bytes with a
    Content-Disposition: attachment so the browser saves rather than plays."""
    client = get_default_song_studio_client()
    target = client.song_download_url(song_id)
    async with httpx.AsyncClient(timeout=300.0) as c:
        try:
            r = await c.get(target)
        except httpx.RequestError as e:
            raise HTTPException(502, f"Song Studio unreachable: {e}")
        if r.status_code != 200:
            raise HTTPException(r.status_code, "Song Studio refused the download")
        headers = {}
        cd = r.headers.get("content-disposition")
        if cd:
            headers["Content-Disposition"] = cd
        else:
            headers["Content-Disposition"] = f'attachment; filename="{song_id}.mp3"'
        media = r.headers.get("content-type", "audio/mpeg")
        return StreamingResponse(r.iter_bytes(), media_type=media, headers=headers)


# ============================================================================
# Phase D: MP3 export with embedded cover art (2026-05-23)
# ============================================================================
# WAV -> MP3 LAME VBR V0 + ID3v2.4 tags + APIC cover. The endpoint here
# fetches the song full payload from Song Studio /api/library (so titles,
# lyrics, cover paths are always fresh), then hands off to
# pipelines.audio.mp3_export.export_song which does the actual ffmpeg call
# + mutagen tag write. Returns a Kraken-Art-served URL the UI can hit to
# trigger a browser download.
#
# Two endpoints:
#   POST /api/audio/songs/{song_id}/export-mp3  -- synchronous, single song
#   POST /api/audio/songs/export-mp3            -- async bulk via JobManager
#                                                  (WS progress on /ws/jobs/{id})

from fastapi.responses import FileResponse  # noqa: E402  -- intentional grouping


def _find_song_in_library(library: dict, song_id: str) -> dict | None:
    for s in (library.get("songs") or []):
        if str(s.get("id")) == str(song_id):
            return s
    return None


class ExportMp3Request(BaseModel):
    overwrite: bool = False
    # Optional metadata overrides. If a caller wants to tag the export with
    # something other than what Song Studio knows about (e.g. a real artist
    # name instead of the workspace title), pass them here.
    title:      str | None = None
    artist:     str | None = None
    album:      str | None = None
    genre:      str | None = None
    comment:    str | None = None


class ExportMp3Response(BaseModel):
    ok: bool
    song_id: str
    mp3_path: str
    mp3_url: str
    source_wav: str
    bitrate_avg_kbps: int
    size_bytes: int
    duration_seconds: float
    cover_embedded: bool
    elapsed_s: float


@router.post("/songs/{song_id}/export-mp3", response_model=ExportMp3Response)
async def export_song_mp3(song_id: str, req: ExportMp3Request = ExportMp3Request()) -> ExportMp3Response:
    """Synchronous LAME VBR V0 export + cover embed for one song."""
    client = get_default_song_studio_client()
    try:
        library = await client.library()
    except Exception as e:
        raise HTTPException(502, f"Song Studio library unreachable: {e}")

    payload = _find_song_in_library(library, song_id)
    if not payload:
        raise HTTPException(404, f"Song {song_id!r} not found in Song Studio library.")

    # Allow per-request metadata overrides without mutating Song Studio state.
    if req.title:   payload["title"]           = req.title
    if req.artist:  payload["workspaceTitle"]  = req.artist  # artist + album both pull from workspaceTitle
    if req.album:   payload["workspaceTitle"]  = req.album
    if req.genre:   payload["styleTags"]       = req.genre
    if req.comment: payload["prompt"]          = req.comment

    # Heavy ffmpeg work runs in a thread so we do not block the event loop.
    import asyncio as _asyncio
    from pipelines.audio.mp3_export import export_song

    try:
        result = await _asyncio.to_thread(export_song, song_payload=payload, overwrite=req.overwrite)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        # ffmpeg / mutagen friendly errors land here
        raise HTTPException(500, str(e))

    return ExportMp3Response(
        ok=result.ok,
        song_id=song_id,
        mp3_path=result.mp3_path,
        mp3_url=result.mp3_url,
        source_wav=result.source_wav,
        bitrate_avg_kbps=result.bitrate_avg_kbps,
        size_bytes=result.size_bytes,
        duration_seconds=result.duration_seconds,
        cover_embedded=result.cover_embedded,
        elapsed_s=result.elapsed_s,
    )


@router.get("/songs/{song_id}/export-mp3/download")
async def download_exported_mp3(song_id: str):
    """Stream the most-recent exported MP3 for this song with a
    Content-Disposition attachment so the browser saves rather than plays.

    The Music tab uses this two-step flow:
      POST /api/audio/songs/{id}/export-mp3   -> {mp3_path, mp3_url, ...}
      GET  the returned mp3_url               -> bytes for an <a download> link
    """
    client = get_default_song_studio_client()
    try:
        library = await client.library()
    except Exception as e:
        raise HTTPException(502, f"Song Studio library unreachable: {e}")
    payload = _find_song_in_library(library, song_id)
    if not payload:
        raise HTTPException(404, f"Song {song_id!r} not found.")

    from pipelines.audio.mp3_export import export_dir_for, _safe_filename
    workspace = str(payload.get("workspaceTitle") or "").strip()
    title = _safe_filename(str(payload.get("title") or "Untitled").strip())
    out_dir = export_dir_for(workspace or None)
    matches = sorted(out_dir.glob(f"{title}*.mp3"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise HTTPException(
            404,
            "No export found for this song. POST /api/audio/songs/{id}/export-mp3 first.",
        )
    return FileResponse(
        matches[0],
        media_type="audio/mpeg",
        filename=matches[0].name,
    )


class BulkExportMp3Request(BaseModel):
    song_ids: list[str]
    overwrite: bool = False


class BulkExportMp3Response(BaseModel):
    job_id: str
    queued: int


def _bulk_export_worker(job) -> dict:
    """JobManager fn: export each song_id in sequence, emit progress, return summary."""
    import asyncio as _asyncio
    from pipelines.audio.mp3_export import export_song

    p = job.params or {}
    ids: list[str] = list(p.get("song_ids") or [])
    overwrite: bool = bool(p.get("overwrite"))

    job.progress.total_steps = len(ids)
    job.progress.step = 0
    job.emit({"type": "status", "message": f"Starting bulk export of {len(ids)} songs..."})

    # Fetch library once up front so the bulk job is one network round trip
    # rather than N. asyncio.run is fine here because we are inside a
    # JobManager worker thread that has no running loop.
    client = get_default_song_studio_client()
    try:
        library = _asyncio.run(client.library())
    except Exception as e:
        raise RuntimeError(f"Song Studio library unreachable: {e}")
    by_id = {str(s.get("id")): s for s in (library.get("songs") or [])}

    successes: list[dict] = []
    failures: list[dict] = []

    for i, sid in enumerate(ids):
        if job.cancel.is_set():
            break
        job.progress.step = i + 1
        payload = by_id.get(str(sid))
        title = (payload or {}).get("title", sid)
        job.emit({"type": "bulk_progress", "step": i + 1, "total": len(ids),
                  "message": f"Exporting {i + 1}/{len(ids)}: {title}"})
        if not payload:
            failures.append({"song_id": sid, "error": "not in Song Studio library"})
            continue
        try:
            r = export_song(song_payload=payload, overwrite=overwrite)
            successes.append({
                "song_id": sid,
                "title": title,
                "mp3_path": r.mp3_path,
                "mp3_url": r.mp3_url,
                "size_bytes": r.size_bytes,
                "bitrate_avg_kbps": r.bitrate_avg_kbps,
                "cover_embedded": r.cover_embedded,
            })
            job.emit({"type": "bulk_song_done", "song_id": sid, "title": title,
                      "mp3_url": r.mp3_url, "size_bytes": r.size_bytes})
        except Exception as e:
            failures.append({"song_id": sid, "title": title, "error": str(e)})
            job.emit({"type": "bulk_song_failed", "song_id": sid, "title": title,
                      "error": str(e)})

    return {
        "kind": "mp3_export",
        "requested": len(ids),
        "succeeded": len(successes),
        "failed": len(failures),
        "outputs": successes,
        "failures": failures,
    }


@router.post("/songs/export-mp3", response_model=BulkExportMp3Response)
def export_songs_bulk(req: BulkExportMp3Request) -> BulkExportMp3Response:
    """Bulk-export the listed song IDs. Returns a job_id immediately; the
    UI subscribes to /ws/jobs/{id} for per-song progress and a final summary.
    """
    if not req.song_ids:
        raise HTTPException(400, "song_ids must contain at least one song id.")
    job = manager.submit(
        kind="mp3_export",
        params={"song_ids": req.song_ids, "overwrite": req.overwrite},
        fn=_bulk_export_worker,
    )
    return BulkExportMp3Response(job_id=job.id, queued=len(req.song_ids))
