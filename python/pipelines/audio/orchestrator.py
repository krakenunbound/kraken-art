"""Audio job orchestrator.

This is the worker function registered with the central JobManager for kind="audio".

It respects the "finished product" rule:
- Never imports the heavy ACE/Stable-Audio training code (the transformers conflict stays isolated in the Kraken_Audio venvs).
- Talks only to a running ACE-Step API (the one the user already launches) via the thin ace_client.
- When the user asks for album art, generates it using Kraken Art's own internal image pipelines (FLUX or SDXL), exactly as requested.

The long-running music synthesis itself happens inside the ACE process on its own GPU context. Our job here only:
1. (optional) Generates a cover image first (blocking, using internal pipelines + VRAM managed by the existing image offload logic).
2. Submits the enriched request to the ACE /release_task endpoint.
3. Polls the ACE job status, emitting friendly progress.
4. On completion, records the returned audio file path(s) + the cover we generated (so the unified Library can show playable songs with nice art).
5. Calls /v1/free on the ACE service so VRAM is released symmetrically for the next image job.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jobs import Job

from pipelines.audio.ace_client import get_default_ace_client
from config import OUTPUTS_ROOT

# We import the image pipelines only when we actually need to generate a cover.
# This keeps the audio module light when the user is only doing music without covers.
# The heavy diffusers/torch stuff is already loaded by the image side anyway.

log = __import__("logging").getLogger("kraken.audio.orchestrator")


@dataclass
class AudioResult:
    audio_paths: list[str] = field(default_factory=list)
    cover_path: str | None = None
    ace_job_id: str | None = None
    ace_status: str | None = None
    message: str = ""


def _ensure_audio_out_dir() -> Path:
    d = OUTPUTS_ROOT / "audio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _generate_cover(job: Job, cover_prompt: str, width: int = 1024, height: int = 1024) -> str | None:
    """Generate a single square album cover using Kraken Art's internal pipelines.

    We deliberately reuse the exact same FLUX/SDXL run() functions that the user already trusts
    for image generation. This satisfies the "redirect ComfyUI cover calls to internal" requirement.
    """
    try:
        # Choose a sensible default for covers: prefer FLUX if the user has the models loaded,
        # otherwise fall back to SDXL. For the first slice we hard-pick a good path.
        # In a later pass this can read the user's lastGenerate or a dedicated "cover_profile".
        from pipelines import flux, sdxl
        from api.generate import GenerateRequest  # reuse the same validation shape

        # Build a minimal, high-quality cover request.
        # The actual model the user has selected in the Image tab (persisted) is ideal,
        # but for now we let the pipeline auto-pick or we can expose a "cover model" setting later.
        req = GenerateRequest(
            arch="flux1",  # or "sdxl" — FLUX usually gives prettier art for covers
            prompt=cover_prompt or "epic cinematic album cover art, high detail, dramatic lighting",
            negative="blurry, lowres, text, watermark, logo",
            width=width,
            height=height,
            steps=28,
            cfg=1.0,           # correct for FLUX
            count=1,
            seed=None,
        )

        # We create a lightweight throwaway Job so the image pipeline can emit progress
        # and we can surface "Generating cover (FLUX 12/28)..." in the music job.
        class _CoverJobShim:
            def __init__(self, parent: Job):
                self.parent = parent
                self.params = req.model_dump()
                self.progress = parent.progress  # share the same progress object for unified bar

            def emit(self, event: dict) -> None:
                # Translate image events into music-friendly messages
                if event.get("type") == "image":
                    self.parent.emit({
                        "type": "audio_cover_progress",
                        "message": f"Cover step {event.get('image_index', 0)+1}",
                        "preview_b64": event.get("preview_b64"),
                    })
                else:
                    self.parent.emit(event)

        shim = _CoverJobShim(job)
        job.emit({"type": "status", "message": "Generating album cover with internal FLUX pipeline..."})

        # Unload any audio-resident models first? The image pipelines already do aggressive unload
        # of the previous arch before loading the new one (see flux.run top).
        result = flux.run(shim)  # type: ignore[arg-type]

        if result and result.get("outputs"):
            first = result["outputs"][0]
            cover_full = first.get("path") or first.get("rel_path")
            job.emit({"type": "audio_cover_done", "path": cover_full})
            return cover_full

    except Exception as e:
        log.warning("Cover generation failed, continuing without cover: %s", e)
        job.emit({"type": "warning", "message": f"Cover generation failed: {e}"})
    return None


def run(job: Job) -> dict[str, Any]:
    """The actual worker executed by the central JobManager for kind='audio'."""
    p = job.params or {}
    client = get_default_ace_client()

    # 1. Optional album cover using Kraken Art's own image engines
    cover_path: str | None = None
    if p.get("generate_cover"):
        cover_prompt = p.get("cover_prompt") or p.get("prompt") or "beautiful album artwork"
        cover_path = _generate_cover(job, str(cover_prompt))

    # 2. Build the payload for the real ACE-Step /release_task
    # The parser in the finished product accepts a wide range of shapes (form or JSON).
    # We send the important musical fields + the model the user picked from the live list.
    ace_payload: dict[str, Any] = {
        "prompt": p.get("prompt", ""),
        "lyrics": p.get("lyrics", ""),
        "bpm": p.get("bpm", 120),
        "key_scale": p.get("key_scale", "C"),
        "time_signature": p.get("time_signature", "4/4"),
        "duration": p.get("duration", 60),           # seconds, if supported
        "temperature": p.get("temperature", 0.85),
        "model": p.get("ace_model") or p.get("model"),  # the DiT the user selected
        "thinking": p.get("thinking", False),
        "sample_mode": p.get("sample_mode", False),
    }

    # If we generated a cover, try to tell ACE about it (the engine may use it for metadata or Song Studio).
    # The exact field name may vary; we include the common candidates the original project used.
    if cover_path:
        ace_payload["cover_image_path"] = cover_path
        ace_payload["reference_cover"] = cover_path

    job.emit({"type": "status", "message": "Submitting to ACE-Step engine..."})

    try:
        submit_resp = client.submit_release_task(ace_payload)
    except Exception as e:
        job.emit({"type": "error", "message": f"Failed to reach ACE API: {e}"})
        raise

    ace_task_id = submit_resp.get("task_id") or submit_resp.get("id")
    if not ace_task_id:
        raise RuntimeError(f"ACE did not return a task_id. Response: {submit_resp}")

    job.emit({"type": "ace_submitted", "ace_task_id": ace_task_id})

    # 3. Poll the ACE service until the music job finishes (or we are cancelled)
    deadline = time.time() + float(p.get("ace_timeout", 600))  # 10 min default safety
    last_progress = 0.0
    audio_paths: list[str] = []

    while True:
        if job.cancel.is_set():
            # Best-effort: ask ACE to free (it may still be running in background)
            try:
                client.free_memory()
            except Exception:
                pass
            return {"kind": "audio", "cancelled": True, "ace_task_id": ace_task_id}

        try:
            status = client.get_job_status(ace_task_id)
        except Exception as e:
            job.emit({"type": "warning", "message": f"Poll error: {e} — retrying"})
            time.sleep(3)
            continue

        prog = float(status.get("progress", 0.0) or 0.0)
        stage = status.get("stage") or status.get("status") or "working"
        msg = status.get("progress_text") or status.get("message") or f"ACE: {stage}"

        if prog > last_progress or stage != "working":
            job.emit({
                "type": "audio_progress",
                "progress": prog,
                "stage": stage,
                "message": msg,
                "ace_task_id": ace_task_id,
            })
            last_progress = prog

        if stage in ("done", "succeeded", "completed", "finished"):
            # ACE finished — extract the audio file locations it produced
            metas = status.get("metas") or {}
            audio_paths = status.get("audio_paths") or metas.get("audio_paths") or []
            break

        if stage in ("failed", "error"):
            err = status.get("error") or "ACE job failed"
            raise RuntimeError(f"ACE generation failed: {err}")

        if time.time() > deadline:
            raise TimeoutError("ACE job did not finish within timeout")

        time.sleep(2.0)  # polite poll interval

    # 4. Symmetric VRAM release — tell the finished ACE engine to drop its models
    # so the user can immediately go back to heavy image generation (FLUX etc.).
    try:
        client.free_memory()
        job.emit({"type": "status", "message": "ACE VRAM released — ready for image work"})
    except Exception:
        pass

    # 5. Record everything the UI needs for playback + gallery
    result = {
        "kind": "audio",
        "ace_task_id": ace_task_id,
        "audio_paths": audio_paths,
        "cover_path": cover_path,
        "prompt": p.get("prompt"),
        "lyrics": p.get("lyrics"),
        "model": ace_payload.get("model"),
        "finished_at": time.time(),
    }

    job.emit({"type": "audio_complete", "result": result})
    return result
