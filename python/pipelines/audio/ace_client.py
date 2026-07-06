"""Minimal async client for a running ACE-Step API (the finished Kraken_Audio engine).

This talks to the existing /release_task, /v1/* and /query_result endpoints
exposed by the ACE-Step FastAPI server (default port 8001 when using the
Kraken_Audio launcher).

We deliberately keep this client small and side-effect free so it can be
imported safely from the main Kraken Art sidecar without pulling in the
conflicting audio training dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import httpx

# Default port used by the Kraken_Audio launcher for the ACE API service
DEFAULT_ACE_API_BASE = os.environ.get("KRAKEN_ACE_API_BASE", "http://127.0.0.1:8001")


@dataclass
class AceClient:
    base_url: str = DEFAULT_ACE_API_BASE
    timeout: float = 30.0
    api_key: Optional[str] = None  # if the ACE instance requires the legacy ai_token

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def health(self) -> dict[str, Any]:
        """Call the ACE health endpoint (usually /health or /v1/stats)."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # The launcher exposes /health on the ace_api service
            r = await client.get(f"{self.base_url}/health", headers=self._headers())
            if r.status_code == 200:
                return r.json()
            # Some versions only have /v1/stats
            r2 = await client.get(f"{self.base_url}/v1/stats", headers=self._headers())
            r2.raise_for_status()
            return r2.json()

    async def list_models(self) -> dict[str, Any]:
        """Return the model inventory from the running ACE service."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(f"{self.base_url}/v1/models", headers=self._headers())
            if r.status_code == 200:
                return r.json()
            # Fallback to the richer inventory endpoint
            r2 = await client.get(f"{self.base_url}/v1/model_inventory", headers=self._headers())
            r2.raise_for_status()
            return r2.json()

    async def submit_release_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a generation job using the legacy-compatible /release_task endpoint.

        The payload can be form-encoded or JSON; we send as JSON for simplicity.
        The real ACE parser accepts both.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/release_task",
                json=payload,
                headers={**self._headers(), "Content-Type": "application/json"},
            )
            r.raise_for_status()
            return r.json()

    async def get_job_status(self, task_id: str) -> dict[str, Any]:
        """Poll job progress using the query_result endpoint."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(
                f"{self.base_url}/query_result",
                params={"task_id": task_id},
                headers=self._headers(),
            )
            r.raise_for_status()
            return r.json()

    async def free_memory(self) -> dict[str, Any]:
        """Ask the ACE service to unload its models (symmetric VRAM release)."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/v1/free", headers=self._headers())
            if r.status_code == 200:
                return r.json()
            return {"ok": False, "status_code": r.status_code}


# Convenience singleton for the sidecar
_default_client: Optional[AceClient] = None


def get_default_ace_client() -> AceClient:
    global _default_client
    if _default_client is None:
        _default_client = AceClient()
    return _default_client
