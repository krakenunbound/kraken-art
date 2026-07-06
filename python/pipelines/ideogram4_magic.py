"""Kraken-side prompt expansion for Ideogram 4.

Ideogram 4 responds best to its structured JSON caption schema. This module
turns ordinary Kraken prompts into that schema without loading a second LLM.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any


PHOTO_WORDS = (
    "photo",
    "photograph",
    "photographic",
    "photoreal",
    "realistic",
    "real photo",
    "professional photograph",
    "portrait photography",
    "fashion editorial",
    "dslr",
    "iphone",
    "35mm",
    "50mm",
    "85mm",
    "kodak",
    "fujifilm",
)

ART_WORDS = (
    "painting",
    "illustration",
    "watercolor",
    "oil painting",
    "digital painting",
    "anime",
    "manga",
    "comic",
    "cartoon",
    "vector",
    "graphic design",
    "logo",
    "poster",
    "book cover",
    "album cover",
    "3d render",
    "render",
)

LIGHTING_WORDS = (
    "dimly lit",
    "cinematic lighting",
    "soft light",
    "soft lighting",
    "natural light",
    "natural lighting",
    "golden hour",
    "studio lighting",
    "dramatic lighting",
    "rim lighting",
    "backlighting",
    "overcast",
    "sunset",
    "sunrise",
    "neon",
)

BACKGROUND_HINTS = (
    "background",
    "setting",
    "landscape",
    "forest",
    "beach",
    "river",
    "ocean",
    "city",
    "street",
    "kitchen",
    "room",
    "studio",
    "field",
    "mountain",
    "sky",
    "desert",
    "castle",
    "canyon",
)


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _sentence(text: str) -> str:
    text = _clean_spaces(text).strip(" ,;:")
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _extract_section(text: str, label: str, next_labels: tuple[str, ...]) -> str:
    labels = "|".join(re.escape(x) for x in next_labels)
    pattern = rf"(?is)\b{re.escape(label)}\b\s*[:\-]?\s*(.*?)(?=\n\s*(?:{labels})\b|\Z)"
    match = re.search(pattern, text)
    return _clean_spaces(match.group(1)) if match else ""


def _strip_structural_labels(text: str) -> str:
    text = re.sub(r"(?im)^\s*[•\-]\s*", "", text)
    text = re.sub(
        r"(?i)\b(high level description|compositional deconstruction|elements?|type\s*:\s*obj|desc\s*:|background)\b\s*[:\-]?",
        " ",
        text,
    )
    return _clean_spaces(text)


def _quoted_text(prompt: str) -> list[str]:
    spans = [
        m.strip()
        for m in re.findall(r"[\"'“”‘’]([^\"'“”‘’]{1,160})[\"'“”‘’]", prompt or "")
        if m.strip()
    ]
    seen: set[str] = set()
    out: list[str] = []
    for span in spans:
        key = span.casefold()
        if key not in seen:
            seen.add(key)
            out.append(span)
    return out[:8]


def _remove_quoted_text(prompt: str) -> str:
    return _clean_spaces(re.sub(r"[\"'“”‘’][^\"'“”‘’]{1,160}[\"'“”‘’]", " ", prompt or ""))


def _aspect(width: int, height: int) -> str:
    width = max(1, int(width or 1))
    height = max(1, int(height or 1))
    div = math.gcd(width, height)
    return f"{width // div}:{height // div}"


def _is_photo(prompt_l: str) -> bool:
    has_photo = any(word in prompt_l for word in PHOTO_WORDS)
    has_art = any(word in prompt_l for word in ART_WORDS)
    if "real photo" in prompt_l or "look like a real photo" in prompt_l:
        return True
    if "painting" in prompt_l and "photo" in prompt_l:
        return False
    if has_art and not has_photo:
        return False
    return True


def _style_description(prompt: str) -> dict[str, Any]:
    prompt_l = prompt.casefold()
    is_photo = _is_photo(prompt_l)

    if "gothic" in prompt_l:
        aesthetics = "surreal gothic fantasy, elegant decay, ethereal, cinematic"
        lighting = "dim atmospheric light, soft haze, controlled highlights on the face"
        palette = ["#1F241E", "#5B4B34", "#8A6F4B", "#C9B783", "#E8D7B5"]
    elif "beach" in prompt_l or "70s" in prompt_l:
        aesthetics = "naturalistic, sunlit, nostalgic 1970s fashion editorial"
        lighting = "neutral coastal daylight with soft shadows"
        palette = ["#F5F1E8", "#F2A7B8", "#E6D9C8", "#79A9B8", "#8B6F4E"]
    elif "mouse" in prompt_l or "children" in prompt_l:
        aesthetics = "photorealistic storybook realism, gentle, detailed, natural"
        lighting = "soft woodland daylight with clear readable foreground detail"
        palette = ["#2F4A2E", "#7A5C37", "#D9C7A4", "#C95F49", "#F4E8D8"]
    elif "logo" in prompt_l or "poster" in prompt_l or "graphic" in prompt_l:
        aesthetics = "clean, controlled, premium graphic design"
        lighting = "even studio-style illumination"
        palette = ["#061522", "#00D5FF", "#F8FAFC", "#F59E0B", "#111827"]
    else:
        aesthetics = "coherent, detailed, polished, high quality"
        lighting = "dimensional lighting with clear focal contrast"
        palette = ["#101820", "#2B5563", "#D8C7A3", "#F2F5F8", "#8A4F3D"]

    named_lighting = [word for word in LIGHTING_WORDS if word in prompt_l]
    if named_lighting:
        lighting = ", ".join(named_lighting[:3])

    if is_photo:
        photo = "natural perspective, crisp subject detail, realistic lens rendering"
        if any(lens in prompt_l for lens in ("35mm", "50mm", "85mm")):
            lenses = [lens for lens in ("35mm", "50mm", "85mm") if lens in prompt_l]
            photo = f"{', '.join(lenses)}, realistic depth of field, crisp subject detail"
        return {
            "aesthetics": aesthetics,
            "lighting": lighting,
            "photo": photo,
            "medium": "photograph",
            "color_palette": palette,
        }

    if "3d render" in prompt_l or "render" in prompt_l:
        medium = "3d_render"
        art_style = "high-end 3D render with realistic materials and controlled composition"
    elif "logo" in prompt_l or "poster" in prompt_l or "graphic" in prompt_l:
        medium = "graphic_design"
        art_style = "precise graphic design with deliberate typography and layout"
    else:
        medium = "painting"
        art_style = "cinematic digital painting with photographic depth and painterly texture"

    return {
        "aesthetics": aesthetics,
        "lighting": lighting,
        "medium": medium,
        "art_style": art_style,
        "color_palette": palette,
    }


def _background(prompt: str, explicit: str) -> str:
    if explicit:
        return _sentence(explicit)

    prompt_l = prompt.casefold()
    if "forest" in prompt_l or "woodland" in prompt_l:
        return (
            "A natural woodland setting with layered trees, earth-toned ground, scattered leaves, "
            "soft haze, and readable depth behind the foreground subject."
        )
    if "beach" in prompt_l or "ocean" in prompt_l:
        return (
            "A bright beach setting with pale sand, ocean surf behind the subject, a clear horizon, "
            "and natural coastal daylight."
        )
    if "gothic" in prompt_l or "ethereal landscape" in prompt_l:
        return (
            "A dim ethereal landscape with muted earth tones, dusty terrain, scattered rocks, sparse "
            "skeletal trees, atmospheric haze, and an overcast sky."
        )
    if "kitchen" in prompt_l:
        return "A lived-in kitchen interior with counters, cabinets, household details, and natural window light."
    if any(word in prompt_l for word in BACKGROUND_HINTS):
        return "A complete environment matching the requested setting, with coherent perspective and atmospheric depth."
    return "A complete atmospheric background that supports the requested subject and keeps the main focal elements readable."


def _main_bbox(width: int, height: int, has_text: bool) -> list[int]:
    ratio = max(1, width) / max(1, height)
    if ratio < 0.82:
        return [70 if not has_text else 150, 120, 970, 880]
    if ratio > 1.18:
        return [120 if not has_text else 210, 150, 900, 850]
    return [90 if not has_text else 180, 90, 940, 910]


def _text_bbox(index: int, width: int, height: int) -> list[int]:
    ratio = max(1, width) / max(1, height)
    if ratio < 0.82:
        y1 = 45 + index * 78
        return [y1, 100, min(y1 + 58, 980), 900]
    y1 = 45 + index * 85
    return [y1, 80, min(y1 + 68, 980), 920]


def _primary_description(prompt: str, explicit_desc: str, negative: str | None) -> str:
    desc = explicit_desc or _strip_structural_labels(_remove_quoted_text(prompt))
    if not desc:
        desc = "A polished, coherent subject matching the requested scene."
    desc = desc.strip(" .")
    if negative and negative.strip():
        desc = f"{desc}. Avoid {negative.strip().strip('.')}"
    return _sentence(desc)


def _high_level(prompt: str, explicit: str, negative: str | None) -> str:
    base = explicit or _strip_structural_labels(_remove_quoted_text(prompt))
    base = _sentence(base or prompt or "A polished image.")
    if negative and negative.strip():
        base = f"{base} Avoid {negative.strip().strip('.')}."
    return base


def _try_json_minify(raw: str) -> str | None:
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def build_local_magic_prompt(prompt: str, negative: str | None, width: int, height: int) -> str:
    """Return a minified Ideogram JSON caption for a plain prompt."""
    raw = (prompt or "").strip()
    if raw.startswith("{"):
        minified = _try_json_minify(raw)
        if minified:
            return minified
        return raw

    hld = _extract_section(
        raw,
        "high level description",
        ("background", "compositional deconstruction", "elements", "type", "desc"),
    )
    background = _extract_section(raw, "background", ("elements", "type", "desc"))
    desc = _extract_section(raw, "desc", ("background", "elements", "type"))

    texts = _quoted_text(raw)
    elements: list[dict[str, Any]] = []
    for idx, text in enumerate(texts):
        elements.append({
            "type": "text",
            "bbox": _text_bbox(idx, width, height),
            "text": text,
            "desc": (
                f"Readable, perfectly spelled text saying exactly '{text}', placed cleanly "
                "inside the composition with strong contrast."
            ),
            "color_palette": ["#F8FAFC", "#111827", "#00D5FF"],
        })

    elements.append({
        "type": "obj",
        "bbox": _main_bbox(width, height, bool(texts)),
        "desc": _primary_description(raw, desc, negative),
    })

    caption = {
        "high_level_description": _high_level(raw, hld, negative),
        "style_description": _style_description(raw),
        "compositional_deconstruction": {
            "background": _background(raw, background),
            "elements": elements,
        },
    }
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def preview_local_magic_prompt(prompt: str, negative: str | None, width: int, height: int) -> dict[str, str]:
    caption = build_local_magic_prompt(prompt, negative, width, height)
    try:
        pretty = json.dumps(json.loads(caption), indent=2, ensure_ascii=False)
    except Exception:
        pretty = caption
    return {
        "mode": "local",
        "aspect_ratio": _aspect(width, height),
        "prompt": caption,
        "pretty_prompt": pretty,
    }
