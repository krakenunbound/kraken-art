"""Sidecar-owned lifecycle for audio generation engines.

The 7780 Kraken Art sidecar is the *only* server the UI talks to. Audio
generation, however, runs in heavyweight engines (ACE-Step for vocals, Stable
Audio 3 for instrumental/SFX) whose Python dependencies conflict with the
image/video venv — so each engine runs as an isolated subprocess in its own
environment.

This manager enforces three rules the user asked for:

  1. Nothing loads at app launch. Engines are *detected* from disk (files
     present) but never started until a job needs one.
  2. Only one audio engine runs at a time. `ensure(id)` stops any other engine
     first, so the previous engine's VRAM is released before the next loads.
  3. The picked engine loads on Generate, exactly like switching an image model.

It is deliberately torch-free at import time; it only spawns/kills processes and
polls HTTP health, so importing it never pulls heavy libs into the sidecar.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config import AUDIO_ACE_ROOT, AUDIO_TTS_ROOT, AUDIO_PROVIDERS_ROOT, AUDIO_MODELS_ROOT

log = logging.getLogger("kraken.audio.engines")

# Optional hook the sidecar wires up (VRAM arbiter, task #17): called right
# before an audio engine is started so resident image/video pipelines can be
# unloaded to free VRAM. Set by api.audio at import time. Kept as a module
# global to avoid a circular import (pipelines <-> api).
on_before_engine_start: Callable[[str], None] | None = None


@dataclass
class EngineSpec:
    id: str
    label: str
    description: str
    base_url: str
    health_path: str
    # Returns True when the engine's files exist on disk (NO model load).
    detect: Callable[[], bool]
    # Returns (argv, cwd, extra_env) used to spawn the engine server.
    build_command: Callable[[], tuple[list[str], Path, dict[str, str]]]
    vram_hint_gb: float
    capability: str = "music"
    instrumental_only: bool = False
    start_timeout_s: int = 240
    direct_runner: bool = False


# ---------------------------------------------------------------------------
# Engine detection / spawn definitions
# ---------------------------------------------------------------------------

def _uv() -> str | None:
    return shutil.which("uv")


def _ace_detect() -> bool:
    """ACE-Step is available if its repo is present and `uv` can run it."""
    return (AUDIO_ACE_ROOT / "pyproject.toml").exists() and _uv() is not None


def _ace_command() -> tuple[list[str], Path, dict[str, str]]:
    uv = _uv()
    if not uv:
        raise RuntimeError("uv.exe not found on PATH — cannot start the ACE-Step engine.")
    host = os.environ.get("KRAKEN_ACE_API_HOST", "127.0.0.1")
    port = os.environ.get("KRAKEN_ACE_API_PORT", "8001")
    argv = [uv, "run", "acestep-api", "--host", host, "--port", port]
    env = {
        "ACESTEP_INIT_LLM": "true",
        "ACESTEP_LM_MODEL_PATH": os.environ.get("KRAKEN_ACE_LM_MODEL", "acestep-5Hz-lm-1.7B"),
        "ACESTEP_LM_BACKEND": "pt",
        "ACESTEP_OFFLOAD_TO_CPU": "true",
        "ACESTEP_API_BASE_URL": f"http://{host}:{port}",
    }
    return argv, AUDIO_ACE_ROOT, env


def _sa3_model_dir() -> Path:
    """Where the Stable Audio 3 Medium weights live (configurable).

    Mirrors the runner's resolution: env override, else the tidy top-level
    folder beside the ACE repo (F:\\Kraken_Audio\\Stable_Audio_3\\...)."""
    explicit = os.environ.get("STABLE_AUDIO3_MODEL_DIR", "").strip()
    if explicit:
        return Path(explicit)
    return AUDIO_ACE_ROOT.parent / "Stable_Audio_3" / "stable-audio-3-medium"


def _sa3_server_script() -> Path:
    """The persistent SA3 server we run inside the ACE-Step venv."""
    return AUDIO_ACE_ROOT / "scripts" / "sa3_server.py"


def _sa3_detect() -> bool:
    """SA3 is available when its weights, server script, and `uv` are all present
    (NO model load — pure file checks)."""
    return (
        _uv() is not None
        and _sa3_server_script().exists()
        and (_sa3_model_dir() / "model.safetensors").exists()
    )


def _sa3_command() -> tuple[list[str], Path, dict[str, str]]:
    uv = _uv()
    if not uv:
        raise RuntimeError("uv.exe not found on PATH — cannot start the Stable Audio 3 engine.")
    host = os.environ.get("KRAKEN_SA3_HOST", "127.0.0.1")
    port = os.environ.get("KRAKEN_SA3_PORT", "8021")
    # Run the server (in scripts/) inside the ACE-Step venv, which is where the
    # stable_audio_3 package + its deps are installed.
    argv = [uv, "run", "python", "scripts/sa3_server.py", "--host", host, "--port", port]
    env = {"STABLE_AUDIO3_MODEL_DIR": str(_sa3_model_dir())}
    return argv, AUDIO_ACE_ROOT, env


def _ace_python() -> Path:
    return AUDIO_ACE_ROOT / ".venv" / "Scripts" / "python.exe"


def _tts_detect() -> bool:
    """Local LuxTTS voice service is available when the lightweight server,
    LuxTTS repo, LinaCodec repo, and ACE-Step venv Python are present."""
    root = AUDIO_ACE_ROOT.parent
    return (
        _ace_python().exists()
        and (AUDIO_TTS_ROOT / "server.py").exists()
        and (root / "LuxTTS" / "zipvoice").exists()
        and (root / "LinaCodec" / "src").exists()
    )


def _tts_command() -> tuple[list[str], Path, dict[str, str]]:
    py = _ace_python()
    if not py.exists():
        raise RuntimeError(f"ACE-Step venv Python not found at {py}")
    host = os.environ.get("KRAKEN_TTS_HOST", "127.0.0.1")
    port = os.environ.get("KRAKEN_TTS_PORT", "8020")
    argv = [str(py), "server.py", "--host", host, "--port", port]
    env = {
        "LUXTTS_MODEL_ID": os.environ.get("LUXTTS_MODEL_ID", "YatharthS/LuxTTS"),
        "LUXTTS_DEVICE": os.environ.get("LUXTTS_DEVICE", "cuda"),
    }
    return argv, AUDIO_TTS_ROOT, env


def _provider_python(provider: str) -> Path:
    return AUDIO_PROVIDERS_ROOT / provider / ".venv" / "Scripts" / "python.exe"


def _moss_sfx_python() -> Path:
    return AUDIO_PROVIDERS_ROOT / "MOSS-TTS" / "moss_soundeffect_v2" / ".venv" / "Scripts" / "python.exe"


def _dummy_direct_command() -> tuple[list[str], Path, dict[str, str]]:
    raise RuntimeError("This audio provider is executed per job and has no persistent server.")


def _moss_sfx_detect() -> bool:
    return (
        _moss_sfx_python().exists()
        and (AUDIO_PROVIDERS_ROOT / "MOSS-TTS" / "moss_soundeffect_v2").exists()
        and (AUDIO_MODELS_ROOT / "moss" / "MOSS-SoundEffect-v2.0" / "transformer" / "diffusion_pytorch_model.safetensors").exists()
    )


def _moss_tts_detect() -> bool:
    return (
        _provider_python("MOSS-TTS").exists()
        and (AUDIO_MODELS_ROOT / "moss" / "MOSS-TTS-Local-Transformer-v1.5" / "model.safetensors").exists()
        and (AUDIO_MODELS_ROOT / "moss" / "MOSS-Audio-Tokenizer-v2" / "model.safetensors.index.json").exists()
    )


def _heartmula_detect() -> bool:
    ckpt = AUDIO_MODELS_ROOT / "heartmula" / "ckpt"
    return (
        _provider_python("heartlib").exists()
        and (ckpt / "HeartMuLa-oss-3B" / "model.safetensors.index.json").exists()
        and (ckpt / "HeartCodec-oss" / "model.safetensors.index.json").exists()
        and (ckpt / "tokenizer.json").exists()
    )


ENGINES: dict[str, EngineSpec] = {
    "ace_step": EngineSpec(
        id="ace_step",
        label="ACE-Step 1.5 (vocals + lyrics)",
        description="Full songs with sung, multilingual lyrics. Self-hosted, fast on a 3090.",
        base_url=os.environ.get("KRAKEN_ACE_API_BASE", "http://127.0.0.1:8001"),
        health_path="/health",
        detect=_ace_detect,
        build_command=_ace_command,
        vram_hint_gb=9.0,
        capability="song",
        instrumental_only=False,
    ),
    "stable_audio_3": EngineSpec(
        id="stable_audio_3",
        label="Stable Audio 3.0 (instrumental / SFX)",
        description="High-fidelity instrumental music and sound effects. No vocals/lyrics.",
        base_url=os.environ.get("KRAKEN_SA3_BASE", "http://127.0.0.1:8021"),
        health_path="/health",
        detect=_sa3_detect,
        build_command=_sa3_command,
        vram_hint_gb=6.0,
        capability="audio",
        instrumental_only=True,
    ),
    "tts_luxtts": EngineSpec(
        id="tts_luxtts",
        label="LuxTTS (speech + voice cloning)",
        description="Fast local 48 kHz voice cloning. Used for narration and multi-voice dialogue.",
        base_url=os.environ.get("KRAKEN_TTS_BASE", "http://127.0.0.1:8020"),
        health_path="/api/config",
        detect=_tts_detect,
        build_command=_tts_command,
        vram_hint_gb=1.0,
        capability="speech",
        instrumental_only=False,
        start_timeout_s=120,
    ),
    "moss_tts": EngineSpec(
        id="moss_tts",
        label="MOSS-TTS Local v1.5 (speech)",
        description="High-quality 48 kHz stereo local TTS. Best for direct narration; LuxTTS remains better for custom multi-voice stitching today.",
        base_url="direct://moss_tts",
        health_path="",
        detect=_moss_tts_detect,
        build_command=_dummy_direct_command,
        vram_hint_gb=16.0,
        capability="speech",
        direct_runner=True,
    ),
    "moss_sfx": EngineSpec(
        id="moss_sfx",
        label="MOSS-SoundEffect v2.0",
        description="Specialist text-to-sound model for precise SFX, ambience, nature, action, and fantasy effects.",
        base_url="direct://moss_sfx",
        health_path="",
        detect=_moss_sfx_detect,
        build_command=_dummy_direct_command,
        vram_hint_gb=12.0,
        capability="sfx",
        instrumental_only=True,
        direct_runner=True,
    ),
    "heartmula": EngineSpec(
        id="heartmula",
        label="HeartMuLa 3B (vocals + lyrics)",
        description="Secondary song-with-vocals engine with strong lyrics control. Runs lazy-load on a single 3090.",
        base_url="direct://heartmula",
        health_path="",
        detect=_heartmula_detect,
        build_command=_dummy_direct_command,
        vram_hint_gb=20.0,
        capability="song",
        direct_runner=True,
    ),
}


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class EngineManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._procs: dict[str, subprocess.Popen] = {}
        self._current: str | None = None

    # ---- introspection ----

    def _running(self, engine_id: str) -> bool:
        proc = self._procs.get(engine_id)
        return proc is not None and proc.poll() is None

    def list(self) -> list[dict]:
        """Engine inventory for the UI. `available` = files on disk (not loaded);
        `running` = subprocess alive."""
        with self._lock:
            out = []
            for spec in ENGINES.values():
                try:
                    available = bool(spec.detect())
                except Exception as e:  # detection must never crash the listing
                    log.debug("detect(%s) raised: %s", spec.id, e)
                    available = False
                out.append({
                    "id": spec.id,
                    "label": spec.label,
                    "description": spec.description,
                    "available": available,
                    "running": self._running(spec.id),
                    "instrumental_only": spec.instrumental_only,
                    "vram_hint_gb": spec.vram_hint_gb,
                    "capability": spec.capability,
                    "direct_runner": spec.direct_runner,
                })
            return out

    def current(self) -> str | None:
        with self._lock:
            return self._current if (self._current and self._running(self._current)) else None

    # ---- health ----

    def _health_ok(self, spec: EngineSpec) -> bool:
        url = spec.base_url.rstrip("/") + spec.health_path
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return 200 <= r.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _wait_health(self, spec: EngineSpec, timeout_s: int) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._health_ok(spec):
                return True
            proc = self._procs.get(spec.id)
            if proc is not None and proc.poll() is not None:
                # Process died during startup — stop waiting.
                log.error("engine %s exited during startup (code %s)", spec.id, proc.returncode)
                return False
            time.sleep(2)
        return False

    # ---- lifecycle ----

    def ensure(self, engine_id: str) -> EngineSpec:
        """Make `engine_id` the sole running audio engine and return its spec.

        Stops any other running engine first (VRAM release), unloads resident
        image/video pipelines via the arbiter hook, then starts the engine if
        it isn't already healthy."""
        spec = ENGINES.get(engine_id)
        if spec is None:
            raise KeyError(f"unknown audio engine: {engine_id!r}")
        with self._lock:
            if not spec.detect():
                raise RuntimeError(
                    f"audio engine {spec.label!r} is not installed/available on this machine."
                )
            if spec.direct_runner:
                raise RuntimeError(
                    f"audio engine {spec.label!r} is a direct-run provider and is not started as a persistent server."
                )

            # Already the active engine and healthy? Nothing to do.
            if self._current == engine_id and self._running(engine_id) and self._health_ok(spec):
                return spec

            # Externally-started instance already healthy (e.g. legacy launcher)?
            if self._running(engine_id) and self._health_ok(spec):
                self._current = engine_id
                return spec

            # Stop every other engine so only one audio model is resident.
            for other in list(self._procs.keys()):
                if other != engine_id:
                    self._stop_locked(other)

            # VRAM arbiter: let the sidecar unload image/video pipelines.
            if on_before_engine_start is not None:
                try:
                    on_before_engine_start(engine_id)
                except Exception as e:
                    log.warning("on_before_engine_start hook failed: %s", e)

            # If a stale instance for this id exists but is unhealthy, kill it.
            if self._running(engine_id):
                self._stop_locked(engine_id)

            argv, cwd, extra_env = spec.build_command()
            env = os.environ.copy()
            env.update(extra_env)
            log.info("starting audio engine %s: %s (cwd=%s)", engine_id, " ".join(argv), cwd)
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            proc = subprocess.Popen(
                argv, cwd=str(cwd), env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            self._procs[engine_id] = proc
            self._current = engine_id

            if not self._wait_health(spec, spec.start_timeout_s):
                self._stop_locked(engine_id)
                raise RuntimeError(
                    f"audio engine {spec.label!r} did not become healthy within "
                    f"{spec.start_timeout_s}s — check its logs."
                )
            log.info("audio engine %s is healthy at %s", engine_id, spec.base_url)
            return spec

    def _kill_tree(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                proc.terminate()
            proc.wait(timeout=20)
        except Exception as e:
            log.warning("kill_tree fallback for pid %s: %s", proc.pid, e)
            try:
                proc.kill()
            except Exception:
                pass

    def _stop_locked(self, engine_id: str) -> None:
        proc = self._procs.pop(engine_id, None)
        if proc is None:
            return
        log.info("stopping audio engine %s (pid %s)", engine_id, proc.pid)
        self._kill_tree(proc)
        if self._current == engine_id:
            self._current = None

    def stop(self, engine_id: str) -> None:
        with self._lock:
            self._stop_locked(engine_id)

    def stop_all(self) -> None:
        with self._lock:
            for engine_id in list(self._procs.keys()):
                self._stop_locked(engine_id)


# Process-wide singleton.
manager = EngineManager()
