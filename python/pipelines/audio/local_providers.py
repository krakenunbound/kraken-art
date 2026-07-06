"""Lightweight launchers for direct-run local audio providers."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from config import AUDIO_MODELS_ROOT, AUDIO_PROVIDERS_ROOT, OUTPUTS_ROOT


RUNNER = Path(__file__).with_name("local_provider_runner.py")


def _moss_tts_python() -> Path:
    return AUDIO_PROVIDERS_ROOT / "MOSS-TTS" / ".venv" / "Scripts" / "python.exe"


def _moss_sfx_python() -> Path:
    return AUDIO_PROVIDERS_ROOT / "MOSS-TTS" / "moss_soundeffect_v2" / ".venv" / "Scripts" / "python.exe"


def _heartmula_python() -> Path:
    return AUDIO_PROVIDERS_ROOT / "heartlib" / ".venv" / "Scripts" / "python.exe"


def _run_json(
    argv: list[str],
    cwd: Path,
    result_json: Path,
    timeout_s: int,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-40:])
        raise RuntimeError(tail or f"provider process exited with code {proc.returncode}")
    if not result_json.exists():
        tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-40:])
        raise RuntimeError(f"provider did not write result JSON. Output tail:\n{tail}")
    return json.loads(result_json.read_text(encoding="utf-8"))


def _out_dir(provider: str) -> Path:
    path = OUTPUTS_ROOT / "audio" / provider / time.strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_moss_sfx(
    prompt: str,
    duration_s: float,
    steps: int,
    cfg_scale: float = 4.0,
    sigma_shift: float = 5.0,
    seed: int = 0,
) -> dict[str, Any]:
    py = _moss_sfx_python()
    model_dir = AUDIO_MODELS_ROOT / "moss" / "MOSS-SoundEffect-v2.0"
    if not py.exists():
        raise RuntimeError(f"MOSS-SoundEffect venv Python not found: {py}")
    if not (model_dir / "transformer" / "diffusion_pytorch_model.safetensors").exists():
        raise RuntimeError(f"MOSS-SoundEffect weights not found: {model_dir}")

    out_dir = _out_dir("moss_sfx")
    output = out_dir / "moss_sfx.wav"
    result_json = out_dir / "result.json"
    seconds = max(0.5, min(float(duration_s), 30.0))
    effective_steps = int(steps) if int(steps) >= 10 else 50
    argv = [
        str(py), str(RUNNER), "moss-sfx",
        "--model-dir", str(model_dir),
        "--prompt", prompt,
        "--seconds", str(seconds),
        "--steps", str(effective_steps),
        "--cfg-scale", str(cfg_scale),
        "--sigma-shift", str(sigma_shift),
        "--seed", str(int(seed)),
        "--output", str(output),
        "--result-json", str(result_json),
    ]
    return _run_json(
        argv,
        AUDIO_PROVIDERS_ROOT / "MOSS-TTS" / "moss_soundeffect_v2",
        result_json,
        1800,
        extra_env={"TORCHDYNAMO_DISABLE": "1"},
    )


def generate_heartmula(
    prompt: str,
    lyrics: str,
    duration_s: float,
    temperature: float = 1.0,
    cfg_scale: float = 1.5,
    topk: int = 50,
) -> dict[str, Any]:
    py = _heartmula_python()
    model_path = AUDIO_MODELS_ROOT / "heartmula" / "ckpt"
    if not py.exists():
        raise RuntimeError(f"HeartMuLa venv Python not found: {py}")
    if not (model_path / "HeartMuLa-oss-3B" / "model.safetensors.index.json").exists():
        raise RuntimeError(f"HeartMuLa weights not found: {model_path}")

    out_dir = _out_dir("heartmula")
    output = out_dir / "heartmula.wav"
    result_json = out_dir / "result.json"
    with TemporaryDirectory(prefix="kraken-heartmula-") as tmp:
        tmp_dir = Path(tmp)
        lyrics_file = tmp_dir / "lyrics.txt"
        tags_file = tmp_dir / "tags.txt"
        lyric_text = lyrics.strip() or "[Verse]\nA local song rises from the machine,\nbright voices moving through a cinematic dream."
        lyrics_file.write_text(lyric_text, encoding="utf-8")
        tags_file.write_text((prompt or "cinematic pop, high quality vocals").strip(), encoding="utf-8")

        argv = [
            str(py), str(RUNNER), "heartmula",
            "--model-path", str(model_path),
            "--lyrics-file", str(lyrics_file),
            "--tags-file", str(tags_file),
            "--output", str(output),
            "--result-json", str(result_json),
            "--max-audio-length-ms", str(int(max(5.0, min(float(duration_s), 240.0)) * 1000)),
            "--topk", str(int(topk)),
            "--temperature", str(float(temperature)),
            "--cfg-scale", str(float(cfg_scale)),
            "--lazy-load",
        ]
        return _run_json(argv, AUDIO_PROVIDERS_ROOT / "heartlib", result_json, 2400)


def generate_moss_tts(
    text: str,
    language: str = "English",
    tokens: int = 0,
    audio_temperature: float = 1.2,
) -> dict[str, Any]:
    py = _moss_tts_python()
    model_dir = AUDIO_MODELS_ROOT / "moss" / "MOSS-TTS-Local-Transformer-v1.5"
    codec_dir = AUDIO_MODELS_ROOT / "moss" / "MOSS-Audio-Tokenizer-v2"
    if not py.exists():
        raise RuntimeError(f"MOSS-TTS venv Python not found: {py}")
    if not (model_dir / "model.safetensors").exists():
        raise RuntimeError(f"MOSS-TTS weights not found: {model_dir}")
    if not (codec_dir / "model.safetensors.index.json").exists():
        raise RuntimeError(f"MOSS Audio Tokenizer v2 weights not found: {codec_dir}")

    out_dir = _out_dir("moss_tts")
    output = out_dir / "moss_tts.wav"
    result_json = out_dir / "result.json"
    argv = [
        str(py), str(RUNNER), "moss-tts",
        "--model-dir", str(model_dir),
        "--codec-dir", str(codec_dir),
        "--text", text,
        "--language", language,
        "--tokens", str(int(tokens)),
        "--audio-temperature", str(float(audio_temperature)),
        "--output", str(output),
        "--result-json", str(result_json),
    ]
    return _run_json(argv, AUDIO_PROVIDERS_ROOT / "MOSS-TTS", result_json, 1200)
