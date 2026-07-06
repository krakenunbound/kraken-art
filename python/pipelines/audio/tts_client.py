"""Async client for the local Kraken TTS Studio service.

The TTS service lives outside the main sidecar so LuxTTS/LinaCodec dependencies
stay isolated from image/video generation. This client keeps the sidecar API
small and model-agnostic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


DEFAULT_TTS_BASE = os.environ.get("KRAKEN_TTS_BASE", "http://127.0.0.1:8020")


@dataclass
class TtsClient:
    base_url: str = DEFAULT_TTS_BASE
    timeout: float = 300.0

    async def config(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(f"{self.base_url}/api/config")
            r.raise_for_status()
            return r.json()

    async def voices(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{self.base_url}/api/voices")
            r.raise_for_status()
            return r.json()

    async def upload_voice(self, name: str, audio_path: Path, ref_text: str = "") -> dict[str, Any]:
        with audio_path.open("rb") as f:
            files = {"audio": (audio_path.name, f, "application/octet-stream")}
            data = {"name": name, "ref_text": ref_text}
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(f"{self.base_url}/api/voices", data=data, files=files)
                r.raise_for_status()
                return r.json()

    async def generate(
        self,
        *,
        text: str,
        voice: str,
        model: str = "luxtts",
        speed: float = 1.0,
        num_steps: int = 4,
        t_shift: float = 0.5,
        ref_duration: float = 5.0,
    ) -> dict[str, Any]:
        data = {
            "text": text,
            "model": model,
            "voice": voice,
            "response_format": "wav",
            "speed": str(speed),
            "num_steps": str(num_steps),
            "t_shift": str(t_shift),
            "ref_duration": str(ref_duration),
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/api/generate", data=data)
            r.raise_for_status()
            out = r.json()
            if out.get("url", "").startswith("/"):
                out["audio_url"] = self.base_url.rstrip("/") + out["url"]
            return out


_default_client: TtsClient | None = None


def get_default_tts_client() -> TtsClient:
    global _default_client
    if _default_client is None:
        _default_client = TtsClient()
    return _default_client
