"""Audio (music/voice) API surface for Kraken Art.

These endpoints let the React UI drive the external ACE-Step engine
without the React frontend ever needing to know the ACE port or deal
with CORS / auth differences.

All heavy lifting stays in the already-finished Kraken_Audio / ACE-Step
installation. We only orchestrate and (later) inject album art generated
by Kraken Art's own image pipelines.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jobs import manager
from pipelines.audio import get_default_ace_client
from pipelines.audio.orchestrator import run as run_audio_job
from pipelines.audio.song_studio_client import get_default_song_studio_client

router = APIRouter(prefix="/audio", tags=["audio"])


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
    prompt: str = ""
    lyrics: str = ""
    ace_model: str | None = None          # which DiT the user picked from the live list
    bpm: int = 120
    key_scale: str = "C"
    duration: int = 60                    # seconds
    temperature: float = 0.85
    generate_cover: bool = True
    cover_prompt: str | None = None       # optional override for the cover art prompt
    thinking: bool = False
    sample_mode: bool = False


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

from fastapi.responses import StreamingResponse


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
    import httpx
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.get(target, headers=client._headers())
        if r.status_code != 200:
            raise HTTPException(r.status_code, "ACE refused to serve the file")
        media = r.headers.get("content-type", "audio/mpeg")
        return StreamingResponse(r.iter_bytes(), media_type=media)


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

import httpx


@router.get("/song-studio/health")
async def song_studio_health():
    """Combined Song Studio probe + model catalog. Returns the full /api/config
    payload so the UI can render the dropdown (`generationModels`) AND show
    health indicators (assistantReady, voiceCloneReady, coverArtStatus)."""
    client = get_default_song_studio_client()
    try:
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
        return await client.library()
    except httpx.RequestError as e:
        raise HTTPException(502, f"Song Studio unreachable: {e}")


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
        return await client.playlists()
    except httpx.RequestError as e:
        raise HTTPException(502, f"Song Studio unreachable: {e}")


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
