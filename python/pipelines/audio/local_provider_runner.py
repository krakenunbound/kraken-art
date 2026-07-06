"""Standalone local audio provider runner.

This file is executed in provider-specific virtualenvs under F:\Kraken_Audio.
Keep it independent from the Kraken Art sidecar imports so dependency-conflicting
audio stacks never load into the sidecar process.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _write_result(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_tensor_audio(audio, output_path: str, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    wav = audio.detach().float().cpu().numpy()
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 2 and wav.shape[0] <= 8:
        wav = wav.T
    wav = np.asarray(wav, dtype=np.float32)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, wav, sample_rate)


def run_moss_sfx(args: argparse.Namespace) -> None:
    import torch
    from moss_soundeffect_v2 import MossSoundEffectPipeline

    started = time.monotonic()
    pipe = MossSoundEffectPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        device=args.device,
    )
    load_s = time.monotonic() - started

    started = time.monotonic()
    audio = pipe(
        prompt=args.prompt,
        seconds=args.seconds,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
    )
    gen_s = time.monotonic() - started
    _save_tensor_audio(audio, args.output, int(getattr(pipe, "sample_rate", 48000)))
    _write_result(args.result_json, {
        "ok": True,
        "provider": "moss_sfx",
        "path": args.output,
        "sample_rate": int(getattr(pipe, "sample_rate", 48000)),
        "load_seconds": round(load_s, 3),
        "generate_seconds": round(gen_s, 3),
    })


def run_heartmula(args: argparse.Namespace) -> None:
    import torch
    from heartlib import HeartMuLaGenPipeline

    lyrics_path = Path(args.lyrics_file)
    tags_path = Path(args.tags_file)
    started = time.monotonic()
    pipe = HeartMuLaGenPipeline.from_pretrained(
        args.model_path,
        device={"mula": torch.device(args.mula_device), "codec": torch.device(args.codec_device)},
        dtype={"mula": torch.bfloat16, "codec": torch.float32},
        version=args.version,
        lazy_load=args.lazy_load,
    )
    load_s = time.monotonic() - started

    started = time.monotonic()
    with torch.no_grad():
        pipe(
            {"lyrics": str(lyrics_path), "tags": str(tags_path)},
            max_audio_length_ms=args.max_audio_length_ms,
            save_path=args.output,
            topk=args.topk,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
        )
    gen_s = time.monotonic() - started
    _write_result(args.result_json, {
        "ok": True,
        "provider": "heartmula",
        "path": args.output,
        "sample_rate": 48000,
        "load_seconds": round(load_s, 3),
        "generate_seconds": round(gen_s, 3),
    })


def run_moss_tts(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModel, AutoProcessor

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    attn = "sdpa" if args.device.startswith("cuda") else "eager"

    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        codec_path=args.codec_dir,
        codec_weight_dtype=args.codec_weight_dtype,
        trust_remote_code=True,
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(args.device)
    model = AutoModel.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        attn_implementation=attn,
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()
    load_s = time.monotonic() - started

    started = time.monotonic()
    conversation = [[processor.build_user_message(
        text=args.text,
        tokens=args.tokens if args.tokens > 0 else None,
        language=args.language or None,
    )]]
    with torch.no_grad():
        batch = processor(conversation[0], mode="generation")
        outputs = model.generate(
            input_ids=batch["input_ids"].to(args.device),
            attention_mask=batch["attention_mask"].to(args.device),
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            audio_temperature=args.audio_temperature,
            audio_top_p=args.audio_top_p,
            audio_top_k=args.audio_top_k,
            audio_repetition_penalty=args.audio_repetition_penalty,
        )
    messages = [m for m in processor.decode(outputs) if m is not None]
    if not messages:
        raise RuntimeError("MOSS-TTS produced no decoded audio message")
    audio = messages[0].audio_codes_list[0]
    gen_s = time.monotonic() - started
    _save_tensor_audio(audio, args.output, int(processor.model_config.sampling_rate))
    _write_result(args.result_json, {
        "ok": True,
        "provider": "moss_tts",
        "path": args.output,
        "sample_rate": int(processor.model_config.sampling_rate),
        "load_seconds": round(load_s, 3),
        "generate_seconds": round(gen_s, 3),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sfx = sub.add_parser("moss-sfx")
    sfx.add_argument("--model-dir", required=True)
    sfx.add_argument("--prompt", required=True)
    sfx.add_argument("--seconds", type=float, default=5.0)
    sfx.add_argument("--steps", type=int, default=50)
    sfx.add_argument("--cfg-scale", type=float, default=4.0)
    sfx.add_argument("--sigma-shift", type=float, default=5.0)
    sfx.add_argument("--seed", type=int, default=0)
    sfx.add_argument("--device", default="cuda")
    sfx.add_argument("--output", required=True)
    sfx.add_argument("--result-json", required=True)
    sfx.set_defaults(func=run_moss_sfx)

    heart = sub.add_parser("heartmula")
    heart.add_argument("--model-path", required=True)
    heart.add_argument("--lyrics-file", required=True)
    heart.add_argument("--tags-file", required=True)
    heart.add_argument("--output", required=True)
    heart.add_argument("--result-json", required=True)
    heart.add_argument("--version", default="3B")
    heart.add_argument("--max-audio-length-ms", type=int, default=30000)
    heart.add_argument("--topk", type=int, default=50)
    heart.add_argument("--temperature", type=float, default=1.0)
    heart.add_argument("--cfg-scale", type=float, default=1.5)
    heart.add_argument("--mula-device", default="cuda")
    heart.add_argument("--codec-device", default="cuda")
    heart.add_argument("--lazy-load", action="store_true")
    heart.set_defaults(func=run_heartmula)

    tts = sub.add_parser("moss-tts")
    tts.add_argument("--model-dir", required=True)
    tts.add_argument("--codec-dir", required=True)
    tts.add_argument("--text", required=True)
    tts.add_argument("--language", default="English")
    tts.add_argument("--tokens", type=int, default=0)
    tts.add_argument("--max-new-tokens", type=int, default=2048)
    tts.add_argument("--audio-temperature", type=float, default=1.2)
    tts.add_argument("--audio-top-p", type=float, default=0.8)
    tts.add_argument("--audio-top-k", type=int, default=25)
    tts.add_argument("--audio-repetition-penalty", type=float, default=1.0)
    tts.add_argument("--codec-weight-dtype", default="bf16")
    tts.add_argument("--device", default="cuda")
    tts.add_argument("--output", required=True)
    tts.add_argument("--result-json", required=True)
    tts.set_defaults(func=run_moss_tts)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
