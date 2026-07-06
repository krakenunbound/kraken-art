"""Dependency check — reports installed pip packages + remediation hints for missing ones."""
from __future__ import annotations
from importlib import metadata
from fastapi import APIRouter

router = APIRouter()

REQUIRED = [
    ("torch",        ">=2.5",   "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"),
    ("torchvision",  ">=0.20",  "pip install torchvision --index-url https://download.pytorch.org/whl/cu124"),
    ("diffusers",    ">=0.32",  "pip install -U diffusers"),
    ("transformers", ">=4.46",  "pip install -U transformers"),
    ("accelerate",   ">=1.2",   "pip install -U accelerate"),
    ("safetensors",  ">=0.4",   "pip install -U safetensors"),
    ("bitsandbytes", ">=0.49",  "pip install -U bitsandbytes"),
    ("fastapi",      ">=0.115", "pip install -U fastapi"),
    ("uvicorn",      ">=0.32",  "pip install -U 'uvicorn[standard]'"),
    ("Pillow",       ">=10",    "pip install -U Pillow"),
    ("numpy",        ">=1.26",  "pip install -U numpy"),
]


def _installed_version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


@router.get("/deps")
def deps_status() -> dict:
    items = []
    missing = 0
    for name, want, fix in REQUIRED:
        v = _installed_version(name)
        ok = v is not None
        if not ok:
            missing += 1
        items.append({
            "name": name,
            "required": want,
            "installed": v,
            "ok": ok,
            "remedy": None if ok else fix,
        })
    return {"all_ok": missing == 0, "missing": missing, "packages": items}
