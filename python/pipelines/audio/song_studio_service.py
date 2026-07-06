"""Lazy lifecycle for the Codex Song Studio workstation (port 8010).

Song Studio provides the Music tab's library, playlists, workspaces, model
catalog and MP3 export. It used to require a separate launcher window — exactly
what the user does NOT want. So the sidecar now starts it on demand (the first
time the Music tab asks for the library/catalog) and stops it on shutdown. It
runs on loopback in the ACE-Step venv via `uv`; the GUI only ever talks to 7780.

Unlike the generation engines this is NOT VRAM-exclusive — Song Studio is a
lightweight workstation (filesystem library + metadata), so it just stays up
once started rather than being killed on engine switches.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

from config import AUDIO_ACE_ROOT

log = logging.getLogger("kraken.audio.song_studio")

_HOST = os.environ.get("KRAKEN_SONG_STUDIO_HOST", "127.0.0.1")
_PORT = os.environ.get("KRAKEN_SONG_STUDIO_PORT", "8010")
_BASE = f"http://{_HOST}:{_PORT}"
_ACE_BASE = os.environ.get("KRAKEN_ACE_API_BASE", "http://127.0.0.1:8001")

_proc: subprocess.Popen | None = None
_lock = threading.RLock()


def _uv() -> str | None:
    return shutil.which("uv")


def available() -> bool:
    """Song Studio can be auto-started (ACE-Step repo + uv present). No load."""
    return (AUDIO_ACE_ROOT / "pyproject.toml").exists() and _uv() is not None


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(_BASE + "/api/config", timeout=2) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _running() -> bool:
    return _proc is not None and _proc.poll() is None


def ensure(timeout_s: int = 300) -> str:
    """Start Song Studio if it isn't already healthy; return its base URL.

    Idempotent and fast when already up. First cold boot is slow (Song Studio
    loads its workstation models at startup — measured ~2.5 min on the 3090), so
    the timeout is generous; it stays resident afterward. Raises on failure."""
    global _proc
    with _lock:
        if _healthy():
            return _BASE
        if not available():
            raise RuntimeError(
                "Song Studio is unavailable — the ACE-Step repo or `uv` was not found. "
                f"(looked under {AUDIO_ACE_ROOT})"
            )
        if not _running():
            uv = _uv()
            argv = [
                uv, "run", "acestep-codex-studio",
                "--host", _HOST, "--port", _PORT,
                "--ace-api-base-url", _ACE_BASE,
            ]
            log.info("starting Song Studio: %s (cwd=%s)", " ".join(argv), AUDIO_ACE_ROOT)
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0  # type: ignore[attr-defined]
            _proc = subprocess.Popen(
                argv, cwd=str(AUDIO_ACE_ROOT),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=flags,
            )
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if _healthy():
                log.info("Song Studio healthy at %s", _BASE)
                return _BASE
            if _proc is not None and _proc.poll() is not None:
                raise RuntimeError(f"Song Studio exited during startup (code {_proc.returncode}).")
            time.sleep(2)
        raise RuntimeError(f"Song Studio did not become healthy within {timeout_s}s.")


def stop() -> None:
    global _proc
    with _lock:
        if _proc is None:
            return
        if _proc.poll() is None:
            log.info("stopping Song Studio (pid %s)", _proc.pid)
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(_proc.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                else:
                    _proc.terminate()
                _proc.wait(timeout=15)
            except Exception:
                try:
                    _proc.kill()
                except Exception:
                    pass
        _proc = None


def status() -> dict:
    return {"available": available(), "running": _running(), "base_url": _BASE}
