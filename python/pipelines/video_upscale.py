"""Video upscaling / enhancement job runner.

SeedVR2 ships as a standalone CLI as well as ComfyUI nodes. Kraken Art uses the
CLI path directly so the feature stays native to the app and does not require a
running ComfyUI server. Frame-rate conversion is handled as a second ffmpeg pass
so the SeedVR2 stage can focus on restoration/upscale quality.
"""
from __future__ import annotations

import gc
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from base64 import b64encode
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from config import MODELS_ROOT, OUTPUTS_ROOT

log = logging.getLogger("kraken.video_upscale")

_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_ROOT = Path(__file__).resolve().parents[2]


def _seedvr2_candidates() -> list[Path]:
    env = os.environ.get("KRAKEN_SEEDVR2_CLI")
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    paths.extend([
        _ROOT / "external" / "SeedVR2" / "inference_cli.py",
        _ROOT / "external" / "seedvr2_videoupscaler" / "inference_cli.py",
        _ROOT / "external" / "ComfyUI-SeedVR2_VideoUpscaler" / "inference_cli.py",
        _ROOT / "external" / "ComfyUI" / "custom_nodes" / "seedvr2_videoupscaler" / "inference_cli.py",
        _ROOT / "external" / "ComfyUI" / "custom_nodes" / "ComfyUI-SeedVR2_VideoUpscaler" / "inference_cli.py",
    ])
    return paths


def _rife_candidates() -> list[Path]:
    env = os.environ.get("KRAKEN_RIFE_NCNN")
    paths: list[Path] = []
    if env:
        paths.append(Path(env))
    paths.extend([
        _ROOT / "external" / "rife-ncnn-vulkan" / "rife-ncnn-vulkan.exe",
        _ROOT / "external" / "rife-ncnn-vulkan" / "rife-ncnn-vulkan-20221029-windows" / "rife-ncnn-vulkan.exe",
    ])
    return paths


def find_seedvr2_cli() -> Path | None:
    for p in _seedvr2_candidates():
        if p.exists() and p.is_file():
            return p.resolve()
    return None


def find_rife_ncnn() -> Path | None:
    for p in _rife_candidates():
        if p.exists() and p.is_file():
            return p.resolve()
    return None


def find_seedvr2_python(cli: Path | None = None) -> Path:
    env = os.environ.get("KRAKEN_SEEDVR2_PYTHON")
    if env and Path(env).exists():
        return Path(env).resolve()
    kraken_venv = _ROOT / "python" / "venv" / "Scripts" / "python.exe"
    if kraken_venv.exists():
        return kraken_venv.resolve()
    if cli:
        for rel in (
            Path(".venv") / "Scripts" / "python.exe",
            Path("venv") / "Scripts" / "python.exe",
            Path(".venv") / "bin" / "python",
            Path("venv") / "bin" / "python",
        ):
            p = cli.parent / rel
            if p.exists():
                return p.resolve()
    return Path(sys.executable).resolve()


def seedvr2_models() -> list[dict[str, Any]]:
    root = MODELS_ROOT / "SEEDVR2"
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in {".safetensors", ".gguf", ".pth", ".pt"}:
            if "vae" in p.name.lower():
                continue
            items.append({
                "name": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
            })
    return items


def upscale_models() -> list[dict[str, Any]]:
    root = MODELS_ROOT / "upscale_models"
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*"), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in {".pth", ".pt", ".safetensors"} and p.stat().st_size > 0:
            items.append({
                "name": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
            })
    return items


def status() -> dict[str, Any]:
    cli = find_seedvr2_cli()
    return {
        "ffmpeg": _FFMPEG,
        "ffprobe": _FFPROBE,
        "rife_ncnn": str(find_rife_ncnn()) if find_rife_ncnn() else None,
        "seedvr2_cli": str(cli) if cli else None,
        "seedvr2_python": str(find_seedvr2_python(cli)) if cli else None,
        "seedvr2_model_dir": str(MODELS_ROOT / "SEEDVR2"),
        "seedvr2_models": seedvr2_models(),
        "upscale_model_dir": str(MODELS_ROOT / "upscale_models"),
        "upscale_models": upscale_models(),
    }


def _resolve_source(p: dict[str, Any]) -> Path:
    rel = p.get("source_rel_path")
    if rel:
        src = (OUTPUTS_ROOT / str(rel)).resolve()
        try:
            src.relative_to(OUTPUTS_ROOT.resolve())
        except ValueError:
            raise HTTPException(400, "source_rel_path escapes outputs root")
        if not src.exists():
            raise HTTPException(404, f"source video not found: {rel}")
        return src
    path = p.get("source_path")
    if path:
        src = Path(str(path)).expanduser()
        if not src.exists():
            raise HTTPException(404, f"source video not found: {path}")
        if not src.is_file():
            raise HTTPException(400, f"source is not a file: {path}")
        return src.resolve()
    raise HTTPException(400, "pick a source video or paste a source_path")


def _probe_video(path: Path) -> dict[str, Any]:
    if not _FFPROBE:
        return {"width": None, "height": None, "fps": None, "duration": None}
    cmd = [
        _FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "default=noprint_wrappers=1:nokey=0",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20)
    except Exception as e:
        log.warning("ffprobe failed for %s: %s", path, e)
        return {"width": None, "height": None, "fps": None, "duration": None}
    data: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    fps = None
    rate = data.get("r_frame_rate")
    if rate and "/" in rate:
        a, b = rate.split("/", 1)
        try:
            fps = float(a) / max(1.0, float(b))
        except ValueError:
            fps = None
    try:
        duration = float(data["duration"]) if data.get("duration") else None
    except ValueError:
        duration = None
    return {
        "width": int(data["width"]) if data.get("width", "").isdigit() else None,
        "height": int(data["height"]) if data.get("height", "").isdigit() else None,
        "fps": fps,
        "duration": duration,
    }


def _has_audio(path: Path) -> bool:
    if not _FFPROBE:
        return False
    cmd = [
        _FFPROBE, "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20)
    except Exception:
        return False
    return "audio" in out.lower()


def _emit(job, step: int, total: int, message: str) -> None:
    if job.cancel.is_set():
        raise RuntimeError("cancelled")
    step = max(int(step), int(getattr(job.progress, "step", 0) or 0))
    job.progress.step = step
    job.progress.total_steps = max(1, total)
    job.progress.message = message
    job.emit({
        "type": "progress",
        "step": step,
        "total_steps": max(1, total),
        "image_index": 0,
        "total_images": 1,
        "message": message,
    })


_PCT_RE = re.compile(r"(?<!\d)(\d{1,3})\s*%")
_COUNT_RE = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)")


def _progress_from_line(line: str) -> int | None:
    m = _PCT_RE.search(line)
    if m:
        pct = max(0, min(100, int(m.group(1))))
        return 5 + int(pct * 0.75)
    m = _COUNT_RE.search(line)
    if m:
        cur, total = int(m.group(1)), max(1, int(m.group(2)))
        if total > 1:
            return 5 + int(max(0, min(1, cur / total)) * 75)
    return None


def _run_process(job, cmd: list[str], cwd: Path | None, label: str, start: int, end: int) -> None:
    log.info("%s command: %s", label, " ".join(cmd))
    _emit(job, start, 100, f"{label}: starting")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    last_emit = time.time()
    tail: list[str] = []
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            if job.cancel.is_set():
                proc.kill()
                raise RuntimeError("cancelled")
            line = raw.strip()
            if not line:
                continue
            tail.append(line)
            tail = tail[-12:]
            parsed = _progress_from_line(line)
            if parsed is not None:
                span = end - start
                mapped = start + int((parsed / 100) * span)
                _emit(job, min(end, mapped), 100, f"{label}: {line[:180]}")
                last_emit = time.time()
            elif time.time() - last_emit > 3:
                _emit(job, min(end - 1, job.progress.step + 1), 100, f"{label}: {line[:180]}")
                last_emit = time.time()
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    if rc != 0:
        raise RuntimeError(f"{label} failed with exit code {rc}: {' | '.join(tail)[-1000:]}")
    _emit(job, end, 100, f"{label}: complete")


def _seedvr2_cmd(req: dict[str, Any], src: Path, dest: Path) -> tuple[list[str], Path]:
    cli = find_seedvr2_cli()
    if not cli:
        raise RuntimeError(
            "SeedVR2 CLI not found. Clone the standalone repo to external\\SeedVR2 "
            "or set KRAKEN_SEEDVR2_CLI to inference_cli.py. Models are expected in models\\SEEDVR2."
        )
    py = find_seedvr2_python(cli)
    resolution = int(req.get("resolution") or 2160)
    max_resolution = int(req.get("max_resolution") or 3840)
    batch_size = int(req.get("batch_size") or 5)
    temporal_overlap = int(req.get("temporal_overlap") or 2)
    input_noise_scale = float(req.get("input_noise_scale") or 0.0)
    latent_noise_scale = float(req.get("latent_noise_scale") or 0.0)
    attention_mode = str(req.get("attention_mode") or "sdpa")
    compile_mode = str(req.get("compile_mode") or "default")
    model = req.get("dit_model") or _default_seedvr2_model()

    cmd = [
        str(py), str(cli), str(src),
        "--output", str(dest),
        "--model_dir", str(MODELS_ROOT / "SEEDVR2"),
        "--resolution", str(resolution),
        "--max_resolution", str(max_resolution),
        "--batch_size", str(batch_size),
        "--temporal_overlap", str(temporal_overlap),
        "--video_backend", "ffmpeg",
        "--attention_mode", attention_mode,
    ]
    if model:
        cmd.extend(["--dit_model", str(model)])
    if req.get("uniform_batch_size", True):
        cmd.append("--uniform_batch_size")
    if input_noise_scale > 0:
        cmd.extend(["--input_noise_scale", str(input_noise_scale)])
    if latent_noise_scale > 0:
        cmd.extend(["--latent_noise_scale", str(latent_noise_scale)])
    if req.get("ten_bit"):
        cmd.append("--10bit")
    if req.get("vae_encode_tiled", True):
        cmd.append("--vae_encode_tiled")
        cmd.extend(["--vae_encode_tile_size", str(int(req.get("vae_encode_tile_size") or 1024))])
        cmd.extend(["--vae_encode_tile_overlap", str(int(req.get("vae_encode_tile_overlap") or 128))])
    if req.get("vae_decode_tiled", True):
        cmd.append("--vae_decode_tiled")
        cmd.extend(["--vae_decode_tile_size", str(int(req.get("vae_decode_tile_size") or 1024))])
        cmd.extend(["--vae_decode_tile_overlap", str(int(req.get("vae_decode_tile_overlap") or 128))])
    color = req.get("color_correction")
    if color:
        cmd.extend(["--color_correction", str(color)])
    blocks = int(req.get("blocks_to_swap") or 0)
    if blocks > 0:
        cmd.extend(["--dit_offload_device", "cpu", "--vae_offload_device", "cpu", "--blocks_to_swap", str(blocks)])
        if req.get("swap_io_components", True):
            cmd.append("--swap_io_components")
    chunk_size = int(req.get("chunk_size") or 0)
    if chunk_size > 0:
        cmd.extend(["--chunk_size", str(chunk_size)])
    if req.get("compile_dit"):
        cmd.append("--compile_dit")
    if req.get("compile_vae"):
        cmd.append("--compile_vae")
    if req.get("compile_dit") or req.get("compile_vae"):
        cmd.extend(["--compile_mode", compile_mode])
    seed = req.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])
    return cmd, cli.parent


def _default_seedvr2_model() -> str | None:
    models = seedvr2_models()
    if not models:
        return None
    for pat in ("3b_fp8", "3b", "fp8", "q8"):
        hit = next((m for m in models if pat in m["name"].lower()), None)
        if hit:
            return hit["name"]
    return models[0]["name"]


def _default_upscale_model() -> str | None:
    models = upscale_models()
    if not models:
        return None
    for pat in ("realesrgan_x2", "nmkd-siax", "siax", "ultrasharp", "remacri", "realesrgan"):
        hit = next((m for m in models if pat in m["name"].lower()), None)
        if hit:
            return hit["name"]
    return models[0]["name"]


def _even(n: float) -> int:
    i = max(2, int(round(n)))
    return i if i % 2 == 0 else i - 1


def _target_size(meta: dict[str, Any], resolution: int, max_resolution: int) -> tuple[int, int, float]:
    w = int(meta.get("width") or 0)
    h = int(meta.get("height") or 0)
    if w <= 0 or h <= 0:
        raise RuntimeError("Could not read source video dimensions.")
    scale = max(1.0, float(resolution) / max(1, min(w, h)))
    long_edge = max(w, h) * scale
    if max_resolution > 0 and long_edge > max_resolution:
        scale = float(max_resolution) / max(1, max(w, h))
    return _even(w * scale), _even(h * scale), scale


def _normalize_size_req(req: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(req)
    if str(out.get("scale_mode") or "resolution") != "factor":
        return out
    w = int(meta.get("width") or 0)
    h = int(meta.get("height") or 0)
    if w <= 0 or h <= 0:
        raise RuntimeError("Could not read source video dimensions for factor upscale.")
    factor = max(1.0, min(8.0, float(out.get("upscale_factor") or 2.0)))
    out["resolution"] = _even(min(w, h) * factor)
    out["max_resolution"] = _even(max(w, h) * factor)
    out["upscale_factor"] = factor
    return out


def _encode_frames_cmd(frames_dir: Path, fps: float, dest: Path) -> list[str]:
    if not _FFMPEG:
        raise RuntimeError("ffmpeg not found on PATH; cannot encode video.")
    return [
        _FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", f"{fps:.6f}",
        "-i", str(frames_dir / "%08d.png"),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(dest),
    ]


def _esrgan_video(job, req: dict[str, Any], src: Path, dest: Path, before: dict[str, Any], start: int, end: int) -> None:
    if not _FFMPEG:
        raise RuntimeError("ffmpeg not found on PATH; ESRGAN video upscaling needs ffmpeg.")
    model = req.get("upscale_model") or _default_upscale_model()
    if not model:
        raise RuntimeError("No ESRGAN/Real-ESRGAN upscale model found in models\\upscale_models.")
    resolution = int(req.get("resolution") or 1080)
    max_resolution = int(req.get("max_resolution") or 1920)
    target_w, target_h, factor = _target_size(before, resolution, max_resolution)
    tile_size = int(req.get("esrgan_tile_size", 0) or 0)
    tile_overlap = int(req.get("esrgan_tile_overlap", 64) or 0)
    fps = float(before.get("fps") or 24.0)

    with tempfile.TemporaryDirectory(prefix="kraken-esrgan-video-") as td:
        base = Path(td)
        in_dir = base / "in"
        out_dir = base / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        _run_process(
            job,
            [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), str(in_dir / "%08d.png")],
            None,
            "ESRGAN extract",
            start,
            min(end - 5, start + 4),
        )

        frames = sorted(in_dir.glob("*.png"))
        if not frames:
            raise RuntimeError("ffmpeg extracted no frames for ESRGAN video upscaling.")
        _emit(job, min(end - 5, start + 5), 100, f"ESRGAN: loading {model}")

        from PIL import Image
        from pipelines import upscale_esrgan

        t0 = time.time()
        for idx, frame in enumerate(frames, start=1):
            if job.cancel.is_set():
                raise RuntimeError("cancelled")
            img = Image.open(frame).convert("RGB")
            up = upscale_esrgan.upscale_tiled(
                img,
                str(model),
                factor=factor,
                tile_size=tile_size,
                overlap=tile_overlap,
            )
            if up.size != (target_w, target_h):
                up = up.resize((target_w, target_h), Image.Resampling.LANCZOS)
            up.save(out_dir / frame.name)
            if idx == 1 or idx == len(frames) or idx % 3 == 0:
                pct = idx / max(1, len(frames))
                mapped = min(end - 5, start + 5 + int(pct * max(1, end - start - 10)))
                fps_done = idx / max(0.001, time.time() - t0)
                _emit(job, mapped, 100, f"ESRGAN: frame {idx}/{len(frames)} ({fps_done:.2f} fps)")

        _run_process(job, _encode_frames_cmd(out_dir, fps, dest), None, "ESRGAN encode", end - 5, end)


def _blend_videos(job, base: Path, detail: Path, dest: Path, strength: float, start: int, end: int) -> None:
    if not _FFMPEG:
        raise RuntimeError("ffmpeg not found on PATH; cannot blend video outputs.")
    strength = max(0.0, min(1.0, float(strength)))
    _run_process(
        job,
        [
            _FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(base),
            "-i", str(detail),
            "-filter_complex", f"[0:v][1:v]blend=all_mode=normal:all_opacity={strength:.4f},format=yuv420p[v]",
            "-map", "[v]",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(dest),
        ],
        None,
        "AI detail blend",
        start,
        end,
    )


def _interpolate_cmd(src: Path, dest: Path, fps: int) -> list[str]:
    if not _FFMPEG:
        raise RuntimeError("ffmpeg not found on PATH; cannot convert frame rate.")
    return [
        _FFMPEG, "-y", "-hide_banner", "-loglevel", "info",
        "-i", str(src),
        "-vf", f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(dest),
    ]


def _rife_model_dir(exe: Path) -> Path:
    for name in ("rife-v4.6", "rife-v4", "rife-v3.1", "rife-v2.4", "rife-v2.3"):
        p = exe.parent / name
        if p.exists():
            return p
    return exe.parent / "rife-v2.3"


def _rife_interpolate(job, src: Path, dest: Path, fps: int, start: int, end: int) -> None:
    exe = find_rife_ncnn()
    if not exe:
        raise RuntimeError(
            "RIFE ncnn Vulkan not found. Install it under external\\rife-ncnn-vulkan "
            "or set KRAKEN_RIFE_NCNN to rife-ncnn-vulkan.exe."
        )
    meta = _probe_video(src)
    duration = float(meta.get("duration") or 0)
    target_frames = max(2, int(math.ceil(duration * fps))) if duration > 0 else 0
    with tempfile.TemporaryDirectory(prefix="kraken-rife-") as td:
        base = Path(td)
        in_dir = base / "in"
        out_dir = base / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        if not _FFMPEG:
            raise RuntimeError("ffmpeg not found on PATH; RIFE interpolation needs ffmpeg for frame extraction/encoding.")
        _run_process(
            job,
            [
                _FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(src), str(in_dir / "%08d.png"),
            ],
            None,
            "RIFE extract",
            start,
            min(end - 2, start + 3),
        )
        cmd = [
            str(exe),
            "-i", str(in_dir),
            "-o", str(out_dir),
            "-m", str(_rife_model_dir(exe)),
            "-g", "0",
            "-j", "2:4:2",
        ]
        if target_frames > 0:
            cmd.extend(["-n", str(target_frames)])
        _run_process(job, cmd, exe.parent, "RIFE interpolation", min(end - 2, start + 3), end - 2)
        _run_process(
            job,
            [
                _FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", str(out_dir / "%08d.png"),
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                str(dest),
            ],
            None,
            "RIFE encode",
            end - 2,
            end,
        )


def _copy_or_transcode(src: Path, dest: Path) -> None:
    if src.resolve() == dest.resolve():
        return
    shutil.copy2(src, dest)


def _mux_source_audio(job, video: Path, audio_src: Path, dest: Path, start: int, end: int) -> None:
    if not _FFMPEG:
        raise RuntimeError("ffmpeg not found on PATH; cannot mux source audio.")
    _run_process(
        job,
        [
            _FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video),
            "-i", str(audio_src),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(dest),
        ],
        None,
        "mux source audio",
        start,
        end,
    )


def _thumb_b64(path: Path) -> str:
    if not _FFMPEG:
        return ""
    with tempfile.TemporaryDirectory(prefix="kraken-vthumb-") as td:
        jpg = Path(td) / "thumb.jpg"
        cmd = [
            _FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(path), "-frames:v", "1", "-q:v", "4", str(jpg),
        ]
        try:
            subprocess.check_call(cmd, timeout=30)
            return b64encode(jpg.read_bytes()).decode("ascii") if jpg.exists() else ""
        except Exception:
            return ""


def _free_visual_pipelines() -> None:
    for mod_name in ("flux", "ideogram4", "sdxl", "z_image", "wan_video"):
        try:
            mod = __import__(f"pipelines.{mod_name}", fromlist=["unload"])
            if hasattr(mod, "unload"):
                mod.unload()
        except Exception:
            pass
    gc.collect()


def run(job) -> dict:
    _free_visual_pipelines()
    raw_req = job.params
    src = _resolve_source(raw_req)
    before = _probe_video(src)
    req = _normalize_size_req(raw_req, before)

    out_dir = OUTPUTS_ROOT / time.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%H%M%S") + "-" + job.id[:8] + "-vup"
    final_path = out_dir / f"{stem}.mp4"

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="kraken-video-upscale-") as td:
        tmp = Path(td)
        restored = tmp / "seedvr2.mp4"
        engine = str(req.get("engine") or "esrgan")
        if engine == "esrgan":
            restored = tmp / "esrgan.mp4"
            _esrgan_video(job, req, src, restored, before, 2, 82)
        else:
            ai_strength = max(0.0, min(1.0, float(req.get("ai_detail_strength", 1.0))))
            if ai_strength <= 0.0001:
                restored = tmp / "preserve.mp4"
                _esrgan_video(job, req, src, restored, before, 2, 82)
            elif ai_strength < 0.9999:
                detail = tmp / "seedvr2_detail.mp4"
                cmd, cwd = _seedvr2_cmd(req, src, detail)
                _run_process(job, cmd, cwd, "SeedVR2", 2, 72)
                preserve = tmp / "preserve.mp4"
                _esrgan_video(job, req, src, preserve, before, 73, 80)
                restored = tmp / "seedvr2_blend.mp4"
                _blend_videos(job, preserve, detail, restored, ai_strength, 81, 82)
            else:
                cmd, cwd = _seedvr2_cmd(req, src, restored)
                _run_process(job, cmd, cwd, "SeedVR2", 2, 82)

        produced = restored
        if not produced.exists():
            candidates = sorted(tmp.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise RuntimeError("SeedVR2 completed but no mp4 output was produced.")
            produced = candidates[0]

        target_fps = int(req.get("target_fps") or 0)
        interpolate = str(req.get("interpolation") or "none")
        source_fps = before.get("fps")
        should_interp = (
            target_fps > 0
            and interpolate != "none"
            and (source_fps is None or target_fps > float(source_fps) + 0.5)
        )
        if should_interp:
            interp = tmp / "interpolated.mp4"
            if interpolate == "rife_ncnn":
                _rife_interpolate(job, produced, interp, target_fps, 83, 96)
            else:
                _run_process(job, _interpolate_cmd(produced, interp, target_fps), None, "60fps interpolation", 83, 96)
            produced = interp

        if bool(req.get("keep_audio", True)) and _has_audio(src):
            with_audio = tmp / "with_audio.mp4"
            _mux_source_audio(job, produced, src, with_audio, 97, 99)
            produced = with_audio

        _emit(job, 99, 100, "saving output")
        _copy_or_transcode(produced, final_path)

    after = _probe_video(final_path)
    rel_path = final_path.relative_to(OUTPUTS_ROOT).as_posix()
    preview = _thumb_b64(final_path)
    elapsed = time.time() - t0
    job.emit({
        "type": "video",
        "image_index": 0,
        "path": str(final_path),
        "rel_path": rel_path,
        "filename": final_path.name,
        "seed": int(req.get("seed") or 42),
        "preview_b64": preview,
    })
    _emit(job, 100, 100, "complete")
    log.info("video upscale complete: %s -> %s in %.1fs", src.name, final_path.name, elapsed)
    return {
        "kind": "video_upscale",
        "source": str(src),
        "output": {
            "path": str(final_path),
            "rel_path": rel_path,
            "filename": final_path.name,
            "width": after.get("width"),
            "height": after.get("height"),
            "fps": after.get("fps"),
            "duration": after.get("duration"),
        },
        "input": before,
        "settings": {
            "engine": req.get("engine") or "esrgan",
            "scale_mode": req.get("scale_mode") or "resolution",
            "upscale_factor": float(req.get("upscale_factor") or 0),
            "ai_detail_strength": float(req.get("ai_detail_strength", 0.15)),
            "resolution": int(req.get("resolution") or 1080),
            "max_resolution": int(req.get("max_resolution") or 1920),
            "target_fps": int(req.get("target_fps") or 0),
            "interpolation": req.get("interpolation") or "none",
            "keep_audio": bool(req.get("keep_audio", True)),
            "dit_model": req.get("dit_model") or _default_seedvr2_model(),
            "upscale_model": req.get("upscale_model") or _default_upscale_model(),
        },
        "elapsed_sec": round(elapsed, 1),
        "output_dir": str(out_dir),
    }
