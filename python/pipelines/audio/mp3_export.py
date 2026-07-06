"""WAV → MP3 export with embedded cover art and ID3v2 tags (Phase D).

Why LAME VBR V0 instead of CBR 320 kbps:
  User preference, learned on a sibling project. LAME `-V 0` is the highest-
  quality VBR setting (target ~245 kbps avg, ranges ~220-280 kbps depending
  on content). It avoids two CBR 320 pitfalls the user has hit before:
    1. CBR allocates the same budget to silent / tonally-simple frames as
       to complex ones, wasting bits and inflating files for music that
       genuinely doesn't need them.
    2. Some downstream players + streaming pipelines have quirks with the
       320 sentinel (LAME's CBR 320 occasionally trips "needs re-encode"
       heuristics in mastering tools because it's the max LAME bitrate).
  VBR V0 sidesteps both and is sonically indistinguishable from CBR 320
  for the kind of generated music ACE-Step makes.

Why ffmpeg via subprocess instead of a pure-Python encoder:
  No good pure-Python MP3 encoder exists. lameenc is a wrapper that needs
  the LAME native lib; pydub is just an ffmpeg wrapper. Going straight to
  ffmpeg (which the user already has on PATH from Gyan's build, with
  libmp3lame compiled in) is one subprocess call and zero extra
  dependencies.

Tag writing happens with mutagen after the encode completes, so we get
real ID3v2.4 frames (TIT2, TPE1, TALB, TCON, COMM, TBPM, USLT, APIC).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mutagen.id3 import (
    APIC,
    COMM,
    ID3,
    ID3NoHeaderError,
    TALB,
    TBPM,
    TCON,
    TIT2,
    TKEY,
    TPE1,
    USLT,
)
from mutagen.mp3 import MP3

from config import OUTPUTS_ROOT

log = logging.getLogger("kraken.audio.mp3_export")

# Resolved once per process. ffmpeg is a system dep documented in the README;
# we look it up explicitly so the error on missing-ffmpeg is friendlier than
# the WinError 2 you'd get from `subprocess.run("ffmpeg", ...)`.
_FFMPEG = shutil.which("ffmpeg")

# LAME VBR quality. `-q:a 0` in ffmpeg's libmp3lame == `-V 0` in LAME CLI
# == "highest quality VBR" == ~245 kbps avg. Tested settings:
#   q=0  V0  ~245 kbps avg  (the default here)
#   q=2  V2  ~190 kbps avg  (transparent for most music, smaller files)
#   q=4  V4  ~165 kbps avg  (passable backup quality)
# We always use 0; the user explicitly asked for highest VBR.
_LAME_VBR_QUALITY = 0


@dataclass
class ExportResult:
    ok: bool
    mp3_path: str
    mp3_url: str
    source_wav: str
    bitrate_avg_kbps: int
    size_bytes: int
    duration_seconds: float
    cover_embedded: bool
    elapsed_s: float
    error: str | None = None


# ---------- filesystem helpers --------------------------------------------------------

_UNSAFE_FN_CHARS = re.compile(r'[<>:"|?*\\/\x00-\x1f]')


def _safe_filename(name: str, max_len: int = 90) -> str:
    """Strip path separators + control chars; collapse whitespace; cap length.

    The cap protects against the Windows 260-char path limit on very long
    workspace titles; 90 chars leaves comfortable headroom for the
    `outputs/exports/<safe>.mp3` prefix.
    """
    cleaned = _UNSAFE_FN_CHARS.sub("_", (name or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(". ")
    if not cleaned:
        cleaned = "untitled"
    return cleaned[:max_len].rstrip(". ")


def export_dir_for(workspace: str | None = None) -> Path:
    """Where the MP3 lands. Grouped by workspace title when given so a bulk
    export from "Codex gqom" doesn't merge with "Salt And Dust" b-sides.
    """
    root = OUTPUTS_ROOT / "exports"
    if workspace:
        root = root / _safe_filename(workspace, max_len=60)
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------- ffmpeg encode -------------------------------------------------------------

def _ensure_ffmpeg() -> str:
    if not _FFMPEG:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (Windows: `winget install Gyan.FFmpeg` "
            "or download from https://ffmpeg.org). The Phase-D MP3 export needs ffmpeg "
            "with libmp3lame compiled in."
        )
    return _FFMPEG


def encode_wav_to_mp3_vbr(source_wav: Path, dest_mp3: Path) -> None:
    """Encode `source_wav` to LAME VBR V0 at `dest_mp3`. Overwrites if it exists.

    Flags chosen carefully:
      -y                       overwrite without prompting (we own the dest path)
      -hide_banner -loglevel error
                               keep stdout/stderr clean so subprocess errors
                               surface the real problem when something fails
      -i <wav>                 source
      -vn                      no video stream (defensive — WAVs shouldn't have
                               one, but if some ACE engine ever produces an .mkv
                               wrapping the audio this guards us)
      -codec:a libmp3lame      pick the LAME encoder explicitly
      -q:a 0                   VBR V0 (highest quality VBR)
      -map_metadata -1         drop any pre-existing metadata; mutagen will
                               write a fresh ID3v2.4 block immediately after.
                               Without this, ffmpeg copies WAV LIST chunks
                               into ID3v1 which then conflicts with our v2.
    """
    ffmpeg = _ensure_ffmpeg()
    cmd = [
        ffmpeg,
        "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source_wav),
        "-vn",
        "-codec:a", "libmp3lame",
        "-q:a", str(_LAME_VBR_QUALITY),
        "-map_metadata", "-1",
        str(dest_mp3),
    ]
    log.info("ffmpeg encode: %s -> %s", source_wav.name, dest_mp3.name)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # stderr already gets the real error because we kept -loglevel error.
        raise RuntimeError(
            f"ffmpeg returned {proc.returncode} encoding {source_wav.name}: "
            f"{(proc.stderr or proc.stdout or '').strip()[:600]}"
        )
    if not dest_mp3.exists() or dest_mp3.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg ran cleanly but {dest_mp3} is missing or empty.")


# ---------- ID3 tag writer ------------------------------------------------------------

def write_id3_tags(
    mp3_path: Path,
    *,
    title: str,
    artist: str,
    album: str | None = None,
    genre: str | None = None,
    comment: str | None = None,
    bpm: int | None = None,
    key_scale: str | None = None,
    lyrics: str | None = None,
    cover_image_path: Path | None = None,
) -> bool:
    """Write ID3v2.4 frames + embed cover. Returns True if a cover was embedded.

    Frame choices:
      TIT2  title
      TPE1  artist     — Song Studio doesn't carry a real artist, so we put
                         the workspace title here (matches what the user is
                         organizing by) and the same value in TALB so most
                         players show it consistently.
      TALB  album
      TCON  genre      — from styleTags
      COMM  comment    — truncated prompt (so the "what was this prompt"
                         survives the export, useful when re-uploading or
                         curating)
      TBPM  bpm
      TKEY  key+scale  — e.g. "F Minor"
      USLT  lyrics     — unsynchronized lyrics
      APIC  cover      — type 3 = Cover (front)

    Mutagen creates the ID3 header if the file doesn't have one yet (which
    is the normal case right after a fresh ffmpeg encode with `-map_metadata -1`).
    """
    try:
        tags = ID3(mp3_path)
    except ID3NoHeaderError:
        tags = ID3()

    # Always overwrite the basics — this is a fresh export, not a tag-merge.
    tags.delall("TIT2"); tags.add(TIT2(encoding=3, text=title))
    tags.delall("TPE1"); tags.add(TPE1(encoding=3, text=artist))

    if album:
        tags.delall("TALB"); tags.add(TALB(encoding=3, text=album))
    if genre:
        tags.delall("TCON"); tags.add(TCON(encoding=3, text=genre))
    if comment:
        # Truncate generously; some players choke on multi-KB COMM frames.
        snippet = comment.strip().replace("\r\n", "\n")[:1500]
        tags.delall("COMM")
        tags.add(COMM(encoding=3, lang="eng", desc="prompt", text=snippet))
    if bpm:
        tags.delall("TBPM"); tags.add(TBPM(encoding=3, text=str(int(bpm))))
    if key_scale:
        tags.delall("TKEY"); tags.add(TKEY(encoding=3, text=str(key_scale)))
    if lyrics:
        tags.delall("USLT")
        tags.add(USLT(encoding=3, lang="eng", desc="lyrics", text=lyrics.strip()))

    cover_embedded = False
    if cover_image_path and cover_image_path.exists():
        try:
            mime = _mime_for(cover_image_path)
            data = cover_image_path.read_bytes()
            tags.delall("APIC")
            tags.add(APIC(
                encoding=3,
                mime=mime,
                type=3,       # Cover (front)
                desc="cover",
                data=data,
            ))
            cover_embedded = True
            log.info("embedded cover: %s (%d KB, %s)", cover_image_path.name, len(data) // 1024, mime)
        except Exception as exc:
            log.warning("cover embed failed (%s): %s — proceeding without cover", cover_image_path, exc)

    # `v2_version=4` writes ID3v2.4 (what most modern players prefer; some
    # very old portable players want v2.3, but for desktop / Suno-style use
    # v2.4 is the right default). `v1=2` strips any leftover ID3v1.
    tags.save(mp3_path, v2_version=4, v1=2)
    return cover_embedded


def _mime_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


# ---------- the high-level "export one song" entry point ------------------------------

def export_song(
    *,
    song_payload: dict[str, Any],
    overwrite: bool = False,
) -> ExportResult:
    """Convert one Song Studio library song to a tagged MP3.

    `song_payload` is the full dict as returned by Song Studio's
    /api/library entry (we just need the fields below). Caller resolves
    that — either by fetching from the live Song Studio (recommended,
    always up to date) or by passing a previously-fetched object.
    """
    t0 = time.time()

    title       = str(song_payload.get("title") or "Untitled").strip()
    workspace   = str(song_payload.get("workspaceTitle") or "").strip()
    style_tags  = str(song_payload.get("styleTags") or "").strip()
    prompt      = str(song_payload.get("prompt") or "").strip()
    lyrics      = str(song_payload.get("lyrics") or "").strip()
    bpm_raw     = song_payload.get("bpm")
    bpm         = int(bpm_raw) if isinstance(bpm_raw, (int, float)) else None
    key_scale   = str(song_payload.get("keyScale") or "").strip() or None

    # Source WAV — prefer masterPath, then first take.path, then any audio_path field.
    source_path = (
        song_payload.get("masterPath")
        or (song_payload.get("takes") or [{}])[0].get("path")
        or song_payload.get("audio_path")
    )
    if not source_path:
        raise FileNotFoundError(
            f"No source audio path on song {song_payload.get('id')!r} "
            f"(checked masterPath, takes[0].path, audio_path)."
        )
    source_wav = Path(source_path)
    if not source_wav.exists():
        raise FileNotFoundError(f"Source audio missing on disk: {source_wav}")

    # Cover path: same convention Song Studio uses + the Kraken Art Phase C
    # side-effect writer. Either way it lands in <folder>/cover_art/cover.png
    # (or .jpg / .webp).
    folder = Path(str(song_payload.get("folderPath") or source_wav.parent))
    cover = _find_cover(folder)

    # Destination: outputs/exports/<workspace>/<title>.mp3
    out_dir = export_dir_for(workspace or None)
    base = f"{_safe_filename(title)}"
    dest = out_dir / f"{base}.mp3"
    if dest.exists() and not overwrite:
        # Pick a sibling name with a numeric suffix so we never silently
        # clobber an earlier export. `(1)` styled to match common Windows
        # de-dupe naming.
        i = 1
        while True:
            candidate = out_dir / f"{base} ({i}).mp3"
            if not candidate.exists():
                dest = candidate
                break
            i += 1

    encode_wav_to_mp3_vbr(source_wav, dest)

    cover_embedded = write_id3_tags(
        dest,
        title=title,
        artist=workspace or "Kraken Studio",
        album=workspace or None,
        genre=style_tags or None,
        comment=prompt or None,
        bpm=bpm,
        key_scale=key_scale,
        lyrics=lyrics or None,
        cover_image_path=cover,
    )

    # Read back metadata from the freshly-tagged file for accurate reporting
    # (size, actual avg bitrate, duration). The MP3 object reads from the
    # XING/LAME tag ffmpeg embeds, so this is fast and accurate even for
    # long files.
    mp3 = MP3(dest)
    avg_kbps = int(round((mp3.info.bitrate or 0) / 1000))
    duration_s = float(mp3.info.length or 0.0)
    size_bytes = dest.stat().st_size

    # URL the UI can hit. The existing /api/outputs/{rel} route mounts
    # OUTPUTS_ROOT, so any path under it works.
    try:
        rel = dest.relative_to(OUTPUTS_ROOT).as_posix()
        mp3_url = f"/api/outputs/{rel}"
    except ValueError:
        mp3_url = ""

    return ExportResult(
        ok=True,
        mp3_path=str(dest),
        mp3_url=mp3_url,
        source_wav=str(source_wav),
        bitrate_avg_kbps=avg_kbps,
        size_bytes=size_bytes,
        duration_seconds=round(duration_s, 2),
        cover_embedded=cover_embedded,
        elapsed_s=round(time.time() - t0, 2),
    )


def _find_cover(folder: Path) -> Path | None:
    """Locate the song's cover image. Tries Song Studio's standard layout
    first, then a few common variants.
    """
    candidates = [
        folder / "cover_art" / "cover.png",
        folder / "cover_art" / "cover.jpg",
        folder / "cover_art" / "cover.jpeg",
        folder / "cover_art" / "cover.webp",
        folder / "cover.png",
        folder / "cover.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Last-ditch: any image file in cover_art/ — Song Studio occasionally
    # writes timestamped filenames there.
    cover_dir = folder / "cover_art"
    if cover_dir.exists():
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            for img in sorted(cover_dir.glob(f"*{ext}")):
                return img
    return None
