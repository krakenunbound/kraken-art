"""Deterministic, offline prompt builder for Kraken Art.

A single "scene" (the user's creative choices: subject, style, lighting,
camera, mood, in-image text) is rendered into each image model's *native*
dialect by a per-model formatter:

  - Ideogram 4   -> structured JSON caption (high_level_description +
                    compositional_deconstruction with bboxes + text elements).
                    This is the model's trained input format.
  - FLUX / Z-Image -> a flowing ~150-word natural-language paragraph following
                    the 4-pillar blueprint (subject -> environment ->
                    camera/lighting -> texture). T5/Qwen encoders want prose.
  - SDXL / Illustrious -> concise comma-separated descriptors + quality
                    boosters, kept short for CLIP's ~75-token budget.

No LLM, no network, no VRAM. Curated phrase libraries are drawn from
Ideogram's own prompting guide (concrete visual grounding over fluff) and
common pro-photography / art-direction vocabulary.

The phrase library is the heart of the builder: each option carries three
renderings so the same creative choice reads naturally in every format.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any


# ============================================================================
#  Phrase libraries — each option => {natural (prose), tags (sdxl), json
#  (ideogram style hint)}. Keep `tags` short; keep `natural` concrete.
# ============================================================================

# A phrase triple. `natural` flows into FLUX prose; `tags` is comma-joined for
# SDXL; `aes` feeds Ideogram's style_description.aesthetics/lighting fields.
def _p(natural: str, tags: str, aes: str) -> dict[str, str]:
    return {"natural": natural, "tags": tags, "aes": aes}


STYLES: dict[str, dict[str, str]] = {
    "auto": _p("", "", ""),
    "cinematic_photo": _p(
        "captured as a cinematic photograph on 35mm film, rich film-stock texture and filmic color grading",
        "cinematic photo, 35mm film, filmic color grading, photographic",
        "cinematic photography, 35mm film stock, filmic color grading"),
    "editorial_photo": _p(
        "shot as high-end editorial photography, crisp and minimal with clean balanced composition",
        "editorial photography, high-end, crisp, minimal, clean composition",
        "high-end editorial photography, crisp minimal aesthetic"),
    "portrait_photo": _p(
        "a professional portrait photograph with natural skin texture and a flattering lens rendering",
        "portrait photography, natural skin texture, 85mm, sharp focus",
        "professional portrait photography, natural skin texture"),
    "oil_painting": _p(
        "rendered as an oil painting with visible brushwork, layered impasto, and rich pigment",
        "oil painting, visible brushwork, impasto, painterly",
        "oil painting with visible brushwork and painterly depth"),
    "watercolor": _p(
        "painted in soft watercolor with bleeding washes, paper texture, and delicate edges",
        "watercolor painting, soft washes, paper texture, delicate",
        "soft watercolor with bleeding washes and paper texture"),
    "digital_painting": _p(
        "a cinematic digital painting with photographic depth and confident painterly texture",
        "digital painting, cinematic, painterly, detailed",
        "cinematic digital painting with photographic depth"),
    "anime": _p(
        "in clean modern anime style with crisp linework and cel shading",
        "anime style, clean lineart, cel shading, vibrant",
        "clean modern anime illustration with crisp linework"),
    "3d_render": _p(
        "a high-end 3D render with physically based materials, accurate reflections, and controlled studio composition",
        "3d render, octane, physically based materials, ray tracing",
        "high-end 3D render with physically based materials"),
    "vector_graphic": _p(
        "bold flat vector art with clean geometric shapes, ideal for layout and typography",
        "flat vector art, bold shapes, clean geometry, graphic design",
        "bold flat vector art with clean geometric shapes"),
    "poster": _p(
        "designed as a premium graphic-design poster with deliberate typography and strong negative space",
        "graphic design poster, bold typography, negative space, clean layout",
        "premium graphic-design poster with deliberate typography and layout"),
    "product_shot": _p(
        "a clean commercial product photograph on a seamless backdrop with crisp studio lighting",
        "product photography, seamless backdrop, studio lighting, commercial",
        "clean commercial product photography on a seamless backdrop"),
}

LIGHTING: dict[str, dict[str, str]] = {
    "auto": _p("", "", ""),
    "golden_hour": _p(
        "bathed in warm golden-hour sunlight with long soft shadows and gentle amber lens flares",
        "golden hour, warm sunlight, long shadows, amber lens flare",
        "warm golden-hour sunlight, long soft shadows, amber flares"),
    "studio_softbox": _p(
        "lit by soft high-key studio softboxes for clean even illumination and gentle shadows",
        "studio lighting, softbox, high-key, soft shadows",
        "soft high-key studio lighting with gentle shadows"),
    "volumetric": _p(
        "with moody volumetric lighting casting visible shafts of light through subtle haze",
        "volumetric lighting, god rays, haze, moody",
        "moody volumetric light with shafts through subtle haze"),
    "neon": _p(
        "illuminated by vibrant neon ambient light, cyan and magenta reflections glistening on wet surfaces",
        "neon lighting, cyan and magenta, wet reflections, cyberpunk glow",
        "vibrant neon ambient light with cyan and magenta reflections"),
    "natural_daylight": _p(
        "in soft natural daylight with realistic, even exposure",
        "natural daylight, soft light, even exposure",
        "soft natural daylight with realistic even exposure"),
    "dramatic_rim": _p(
        "with dramatic rim lighting separating the subject from a dark background",
        "rim lighting, dramatic, high contrast, dark background",
        "dramatic rim lighting against a dark background"),
    "overcast": _p(
        "under flat overcast light with soft diffuse shadows and muted contrast",
        "overcast light, soft diffuse shadows, muted",
        "flat overcast light with soft diffuse shadows"),
    "candlelit": _p(
        "lit by warm flickering candlelight with deep falloff into shadow",
        "candlelight, warm glow, deep shadows, chiaroscuro",
        "warm candlelight with deep falloff into shadow"),
}

CAMERAS: dict[str, dict[str, str]] = {
    "auto": _p("", "", ""),
    "low_angle": _p(
        "framed from a low angle to emphasize scale, shallow depth of field with the subject in razor focus",
        "low angle, dynamic, shallow depth of field, bokeh background",
        "low-angle perspective, shallow depth of field, sharp subject"),
    "macro": _p(
        "an extreme macro close-up with microscopic detail and a tight focus plane",
        "macro, extreme close-up, microscopic detail, bokeh",
        "extreme macro close-up with microscopic detail"),
    "wide_establishing": _p(
        "a wide establishing shot with the subject framed by a vast, story-rich environment",
        "wide angle, establishing shot, vast environment",
        "wide establishing composition framed by a vast environment"),
    "portrait_headshot": _p(
        "a tight headshot on a portrait lens with smooth background separation",
        "headshot, 85mm portrait lens, background separation",
        "tight portrait headshot with smooth background separation"),
    "flat_lay": _p(
        "a top-down flat-lay composition, neatly arranged and evenly lit",
        "flat lay, top-down, knolling, evenly lit",
        "top-down flat-lay composition, evenly arranged"),
    "centered_hero": _p(
        "a centered hero composition with the subject dominant and generous negative space",
        "centered composition, hero shot, negative space",
        "centered hero composition with generous negative space"),
    "rule_of_thirds": _p(
        "composed on the rule of thirds with balanced foreground and background interest",
        "rule of thirds, balanced composition, depth",
        "rule-of-thirds composition with balanced depth"),
}

MOODS: dict[str, dict[str, str]] = {
    "auto": _p("", "", ""),
    "warm_nostalgic": _p(
        "evoking a warm, nostalgic, timeless mood",
        "nostalgic, warm tones, timeless",
        "warm nostalgic timeless mood"),
    "epic_dramatic": _p(
        "with an epic, dramatic, high-stakes atmosphere",
        "epic, dramatic, cinematic, high contrast",
        "epic dramatic high-contrast atmosphere"),
    "serene_calm": _p(
        "conveying a serene, calm, contemplative stillness",
        "serene, calm, soft, peaceful",
        "serene calm contemplative stillness"),
    "dark_moody": _p(
        "with a dark, moody, mysterious tone",
        "dark, moody, mysterious, low-key",
        "dark moody mysterious tone"),
    "vibrant_playful": _p(
        "with a vibrant, playful, energetic feel and saturated color",
        "vibrant, playful, saturated, energetic",
        "vibrant playful saturated energy"),
    "cyberpunk": _p(
        "in a gritty high-tech cyberpunk atmosphere",
        "cyberpunk, neon-noir, gritty, futuristic",
        "gritty high-tech cyberpunk atmosphere"),
}

# Quality boosters appended per-format (concise for SDXL, woven for FLUX).
SDXL_QUALITY = "highly detailed, sharp focus, high quality, masterpiece"
FLUX_TEXTURE = ("Fine textures are rendered with care: believable materials, "
                "subtle surface imperfections, and crisp focal detail.")


# ============================================================================
#  Scene model
# ============================================================================

@dataclass
class SceneSpec:
    subject: str = ""
    texts: list[str] = field(default_factory=list)   # exact in-image strings
    style: str = "auto"
    lighting: str = "auto"
    camera: str = "auto"
    mood: str = "auto"
    negative: str = ""
    width: int = 1024
    height: int = 1024

    def picks(self) -> list[tuple[dict[str, str], str]]:
        """Return (phrase_dict, library_name) for each non-auto pick, in
        canonical order: style, camera, lighting, mood."""
        out = []
        for lib, key in ((STYLES, self.style), (CAMERAS, self.camera),
                         (LIGHTING, self.lighting), (MOODS, self.mood)):
            entry = lib.get((key or "auto").lower())
            if entry and entry.get("natural"):
                out.append(entry)
        return out


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _quoted(s: str) -> list[str]:
    spans = [m.strip() for m in re.findall(r"[\"'“”‘’]([^\"'“”‘’]{1,160})[\"'“”‘’]", s or "") if m.strip()]
    seen, out = set(), []
    for sp in spans:
        k = sp.casefold()
        if k not in seen:
            seen.add(k); out.append(sp)
    return out[:6]


def _strip_quotes(s: str) -> str:
    return _clean(re.sub(r"[\"'“”‘’][^\"'“”‘’]{1,160}[\"'“”‘’]", " ", s or ""))


# ============================================================================
#  Formatters
# ============================================================================

def format_flux(spec: SceneSpec) -> str:
    """Flowing natural-language paragraph (4-pillar blueprint). Good for
    FLUX, Z-Image, Qwen-Image, HunYuan — anything with a T5/Qwen encoder."""
    subject = _strip_quotes(spec.subject) or "a striking subject"
    parts: list[str] = [subject.rstrip(".")]
    for entry, _ in [(e, "") for e in spec.picks()]:
        parts.append(entry["natural"])
    body = ", ".join(p for p in parts if p)
    sentence = body[0].upper() + body[1:] if body else body
    out = sentence.rstrip(".") + ". " + FLUX_TEXTURE
    for t in spec.texts:
        out += f' The image includes the text "{t}" rendered cleanly and correctly.'
    if spec.negative.strip():
        out += f" Avoid {spec.negative.strip().rstrip('.')}"
    return _clean(out)


def format_sdxl(spec: SceneSpec) -> str:
    """Concise comma-separated descriptors + quality boosters (CLIP budget)."""
    subject = _strip_quotes(spec.subject) or "a striking subject"
    chunks = [subject.rstrip(".")]
    for entry in spec.picks():
        if entry.get("tags"):
            chunks.append(entry["tags"])
    for t in spec.texts:
        chunks.append(f'text "{t}"')
    chunks.append(SDXL_QUALITY)
    # Dedup words-ish while preserving order; trim to a sane length.
    seen, out = set(), []
    for c in ", ".join(chunks).split(", "):
        k = c.strip().casefold()
        if k and k not in seen:
            seen.add(k); out.append(c.strip())
    return ", ".join(out)


# ---- Ideogram structured-JSON formatter ------------------------------------

def _aspect(w: int, h: int) -> str:
    w, h = max(1, int(w or 1)), max(1, int(h or 1))
    d = math.gcd(w, h)
    return f"{w // d}:{h // d}"


def _main_bbox(w: int, h: int, has_text: bool) -> list[int]:
    r = max(1, w) / max(1, h)
    if r < 0.82:
        return [70 if not has_text else 150, 120, 970, 880]
    if r > 1.18:
        return [120 if not has_text else 210, 150, 900, 850]
    return [90 if not has_text else 180, 90, 940, 910]


def _text_bbox(i: int, w: int, h: int) -> list[int]:
    r = max(1, w) / max(1, h)
    y1 = 45 + i * (78 if r < 0.82 else 85)
    return [y1, 100 if r < 0.82 else 80, min(y1 + (58 if r < 0.82 else 68), 980), 900 if r < 0.82 else 920]


def format_ideogram(spec: SceneSpec) -> str:
    """Structured JSON caption — Ideogram 4's native trained format."""
    subject = _strip_quotes(spec.subject) or "a polished, coherent subject"
    picks = spec.picks()
    aes = ", ".join(e["aes"] for e in picks if e.get("aes"))

    # style_description: pull a lighting line out of the picks if present.
    light_entry = LIGHTING.get((spec.lighting or "auto").lower())
    style_desc: dict[str, Any] = {
        "aesthetics": aes or "coherent, detailed, polished, high quality",
        "lighting": (light_entry["aes"] if light_entry and light_entry.get("aes")
                     else "dimensional lighting with clear focal contrast"),
        "medium": _medium_for(spec.style),
        "color_palette": _palette_for(spec.mood),
    }

    elements: list[dict[str, Any]] = []
    has_text = bool(spec.texts)
    for i, t in enumerate(spec.texts):
        elements.append({
            "type": "text",
            "bbox": _text_bbox(i, spec.width, spec.height),
            "text": t,
            "desc": f"Readable, perfectly spelled text saying exactly '{t}', placed cleanly with strong contrast.",
        })
    primary = subject.rstrip(".")
    if aes:
        primary = f"{primary}, {aes}"
    if spec.negative.strip():
        primary = f"{primary}. Avoid {spec.negative.strip().rstrip('.')}"
    elements.append({"type": "obj", "bbox": _main_bbox(spec.width, spec.height, has_text),
                     "desc": primary.rstrip(".") + "."})

    hld = subject.rstrip(".")
    if picks:
        hld = f"{hld}, {', '.join(e['aes'] for e in picks if e.get('aes'))}"
    caption = {
        "high_level_description": hld.rstrip(".") + ".",
        "style_description": style_desc,
        "compositional_deconstruction": {
            "background": _background_for(spec),
            "elements": elements,
        },
    }
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def _medium_for(style: str) -> str:
    s = (style or "").lower()
    if s in ("oil_painting", "watercolor", "digital_painting", "anime"):
        return "painting"
    if s in ("vector_graphic", "poster"):
        return "graphic_design"
    if s == "3d_render":
        return "3d_render"
    return "photograph"


def _palette_for(mood: str) -> list[str]:
    return {
        "warm_nostalgic": ["#2B1F16", "#7A5C37", "#D9B27A", "#F2E2C4", "#C9794A"],
        "epic_dramatic":  ["#0B0D12", "#243044", "#7A4B3A", "#C9A24B", "#E8E2D4"],
        "serene_calm":    ["#1B2B33", "#3F6B73", "#A9C7CC", "#E6EFEF", "#D8C7A3"],
        "dark_moody":     ["#0A0A0D", "#1C2230", "#3A3F52", "#6B5B4A", "#B7A98C"],
        "vibrant_playful":["#1A1140", "#FF3D7F", "#FFC23D", "#3DDC97", "#00C2FF"],
        "cyberpunk":      ["#06121E", "#00E5FF", "#FF2E97", "#F8FAFC", "#111827"],
    }.get((mood or "").lower(), ["#101820", "#2B5563", "#D8C7A3", "#F2F5F8", "#8A4F3D"])


def _background_for(spec: SceneSpec) -> str:
    s = (spec.subject or "").lower()
    for key, desc in (
        ("forest", "A natural woodland setting with layered trees, earth-toned ground, and readable depth behind the subject."),
        ("beach", "A bright beach with pale sand, ocean surf, a clear horizon, and natural coastal daylight."),
        ("city", "A detailed urban environment with believable architecture, depth, and atmospheric perspective."),
        ("studio", "A clean seamless studio backdrop with controlled, even lighting."),
        ("mountain", "A vast mountain landscape with layered ridgelines and atmospheric haze."),
        ("space", "A deep-space backdrop with stars, soft nebula color, and a sense of vast scale."),
    ):
        if key in s:
            return desc
    return "A complete, atmospheric background that supports the subject and keeps the focal elements readable."


# ============================================================================
#  Public API
# ============================================================================

ARCH_FORMATTER = {
    "ideogram4": format_ideogram,
    "flux1": format_flux,
    "flux2": format_flux,
    "z_image": format_flux,
    "qwen_image": format_flux,
    "hunyuan": format_flux,
    "sdxl": format_sdxl,
    "illustrious": format_sdxl,
}


def build(spec: SceneSpec, arch: str) -> dict[str, Any]:
    """Render a scene into the target architecture's native prompt format.
    Returns {arch, format, prompt, pretty} (pretty = indented JSON for
    Ideogram, identical to prompt otherwise)."""
    fmt = ARCH_FORMATTER.get((arch or "").lower(), format_flux)
    prompt = fmt(spec)
    fmt_name = {"ideogram4": "json"}.get((arch or "").lower(),
               "tags" if fmt is format_sdxl else "natural")
    pretty = prompt
    if fmt_name == "json":
        try:
            pretty = json.dumps(json.loads(prompt), indent=2, ensure_ascii=False)
        except Exception:
            pass
    return {"arch": (arch or "").lower(), "format": fmt_name,
            "prompt": prompt, "pretty": pretty, "aspect_ratio": _aspect(spec.width, spec.height)}


def options() -> dict[str, list[dict[str, str]]]:
    """Library of selectable options for the UI (value + human label)."""
    def opts(lib: dict, labels: dict[str, str]) -> list[dict[str, str]]:
        return [{"value": k, "label": labels.get(k, k.replace("_", " ").title())} for k in lib]
    return {
        "style": opts(STYLES, {"auto": "Auto / from subject", "cinematic_photo": "Cinematic photo",
            "editorial_photo": "Editorial photo", "portrait_photo": "Portrait photo",
            "oil_painting": "Oil painting", "watercolor": "Watercolor", "digital_painting": "Digital painting",
            "anime": "Anime", "3d_render": "3D render", "vector_graphic": "Vector / graphic",
            "poster": "Poster", "product_shot": "Product shot"}),
        "lighting": opts(LIGHTING, {"auto": "Auto", "golden_hour": "Golden hour", "studio_softbox": "Studio softbox",
            "volumetric": "Volumetric / god rays", "neon": "Neon", "natural_daylight": "Natural daylight",
            "dramatic_rim": "Dramatic rim", "overcast": "Overcast", "candlelit": "Candlelit"}),
        "camera": opts(CAMERAS, {"auto": "Auto", "low_angle": "Low angle", "macro": "Macro close-up",
            "wide_establishing": "Wide establishing", "portrait_headshot": "Portrait headshot",
            "flat_lay": "Flat lay (top-down)", "centered_hero": "Centered hero", "rule_of_thirds": "Rule of thirds"}),
        "mood": opts(MOODS, {"auto": "Auto", "warm_nostalgic": "Warm / nostalgic", "epic_dramatic": "Epic / dramatic",
            "serene_calm": "Serene / calm", "dark_moody": "Dark / moody", "vibrant_playful": "Vibrant / playful",
            "cyberpunk": "Cyberpunk"}),
    }


def spec_from_payload(p: dict[str, Any]) -> SceneSpec:
    subject = str(p.get("subject") or p.get("prompt") or "")
    texts = p.get("texts")
    if not isinstance(texts, list) or not texts:
        texts = _quoted(subject)
    return SceneSpec(
        subject=subject,
        texts=[str(t) for t in texts][:6],
        style=str(p.get("style") or "auto"),
        lighting=str(p.get("lighting") or "auto"),
        camera=str(p.get("camera") or "auto"),
        mood=str(p.get("mood") or "auto"),
        negative=str(p.get("negative") or ""),
        width=int(p.get("width") or 1024),
        height=int(p.get("height") or 1024),
    )
