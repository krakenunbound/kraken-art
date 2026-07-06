"""Thin httpx wrapper around Kraken_Audio's Codex Song Studio API.

The Song Studio (codex_song_studio.py in F:\\Kraken_Audio\\ACE-Step-1.5\\acestep\\)
runs by default on http://127.0.0.1:8010 and is the user-facing music
workstation. It owns:

  * The model catalog (ACE-Step + Stable Audio 3 variants) — /api/config
  * The library of generated songs — /api/library (heavy: scans filesystem)
  * Workspaces (POST/PATCH only; enumeration is implicit via songs' workspaceId)
  * Playlists CRUD — /api/playlists
  * Song detail / delete / bulk-delete — /api/library/songs/{id} ...
  * MP3 download — /api/library/songs/{id}/download
  * Audio file stream — /api/audio?path=...
  * Cover-art job submission — /api/library/songs/{id}/cover-art (currently
    routed to ComfyUI; Phase C will replace with Kraken Art's FLUX/SDXL)

This client is intentionally tiny — no caching, no retries, no opinions.
Kraken Art's /api/audio router (in python/api/audio.py) thinly forwards.
"""
from __future__ import annotations

import os
from typing import Any

import httpx


DEFAULT_SONG_STUDIO_BASE = os.environ.get(
    "KRAKEN_SONG_STUDIO_URL", "http://127.0.0.1:8010"
)


class SongStudioClient:
    """Minimal async client. Built fresh per request — Song Studio is a local
    process; connection reuse isn't worth the lifecycle management."""

    def __init__(self, base_url: str = DEFAULT_SONG_STUDIO_BASE) -> None:
        self.base_url = base_url.rstrip("/")

    def _client(self, timeout: float = 30.0) -> httpx.AsyncClient:
        # Song Studio scans the filesystem on /api/library — needs a generous
        # timeout. 60 s covers libraries up to a few hundred songs comfortably.
        return httpx.AsyncClient(timeout=timeout)

    # ---- Catalog / health ---------------------------------------------------

    async def config(self) -> dict[str, Any]:
        """Whole /api/config payload. Includes assistantReady, voiceCloneReady,
        coverArtStatus, generationModels (the model dropdown source), and the
        Song Studio's view of its own LAN/desktop URLs."""
        async with self._client(timeout=10.0) as c:
            r = await c.get(f"{self.base_url}/api/config")
            r.raise_for_status()
            return r.json()

    # ---- Library + workspaces ----------------------------------------------

    async def library(self) -> dict[str, Any]:
        """All songs in the library. Returns {"songs": [...]}. Heavy call."""
        async with self._client(timeout=60.0) as c:
            r = await c.get(f"{self.base_url}/api/library")
            r.raise_for_status()
            return r.json()

    async def song_detail(self, song_id: str) -> dict[str, Any]:
        async with self._client() as c:
            r = await c.get(f"{self.base_url}/api/library/songs/{song_id}")
            r.raise_for_status()
            return r.json()

    async def delete_song(self, song_id: str) -> dict[str, Any]:
        async with self._client() as c:
            r = await c.delete(f"{self.base_url}/api/library/songs/{song_id}")
            r.raise_for_status()
            return r.json() if r.content else {"ok": True}

    async def bulk_delete(self, song_ids: list[str]) -> dict[str, Any]:
        async with self._client() as c:
            r = await c.post(
                f"{self.base_url}/api/library/songs/bulk-delete",
                json={"songIds": song_ids},
            )
            r.raise_for_status()
            return r.json() if r.content else {"ok": True}

    async def create_workspace(self, title: str) -> dict[str, Any]:
        async with self._client() as c:
            r = await c.post(
                f"{self.base_url}/api/workspaces", json={"title": title}
            )
            r.raise_for_status()
            return r.json()

    async def rename_workspace(self, workspace_id: str, title: str) -> dict[str, Any]:
        async with self._client() as c:
            r = await c.patch(
                f"{self.base_url}/api/workspaces/{workspace_id}",
                json={"title": title},
            )
            r.raise_for_status()
            return r.json()

    # ---- Playlists ----------------------------------------------------------

    async def playlists(self) -> dict[str, Any]:
        async with self._client() as c:
            r = await c.get(f"{self.base_url}/api/playlists")
            r.raise_for_status()
            return r.json()

    async def create_playlist(self, title: str) -> dict[str, Any]:
        async with self._client() as c:
            r = await c.post(
                f"{self.base_url}/api/playlists", json={"title": title}
            )
            r.raise_for_status()
            return r.json()

    async def add_songs_to_playlist(
        self, playlist_id: str, song_ids: list[str]
    ) -> dict[str, Any]:
        async with self._client() as c:
            r = await c.post(
                f"{self.base_url}/api/playlists/{playlist_id}/songs",
                json={"songIds": song_ids},
            )
            r.raise_for_status()
            return r.json() if r.content else {"ok": True}

    # ---- Audio / download streams (handled by the proxy router for headers) -

    def audio_url(self, path: str) -> str:
        # The actual byte-streaming lives in api/audio.py's StreamingResponse
        # because we need to set Content-Disposition + media type from the
        # upstream response. This helper just centralizes URL building.
        return f"{self.base_url}/api/audio?path={path}"

    def song_download_url(self, song_id: str) -> str:
        return f"{self.base_url}/api/library/songs/{song_id}/download"


_default_song_studio_client: SongStudioClient | None = None


def get_default_song_studio_client() -> SongStudioClient:
    """Lazy singleton. Rebuilt only if KRAKEN_SONG_STUDIO_URL changes between
    process restarts (which means: don't change it at runtime)."""
    global _default_song_studio_client
    if _default_song_studio_client is None:
        _default_song_studio_client = SongStudioClient()
    return _default_song_studio_client
