"""
VRAM Coordinator (PR 1 skeleton).

Responsible for cross-modality VRAM discipline (e.g. "make sure only the
requested modality's heavy models are resident before a job runs").

PR 1: Interface + minimal no-op implementation.
Real coordination logic will be wired in PR 1 / PR 2 / PR 3.
"""

from __future__ import annotations

from typing import Literal, Optional


Modality = Literal["image", "music", "video", "voice"]


class VRAMCoordinator:
    """
    PR 1 skeleton.

    Future responsibilities:
    - ensure_only(modality) before heavy work
    - track current resident heavy models
    - integrate with existing unload() patterns in flux.py / sdxl.py / audio proxies
    """

    def __init__(self) -> None:
        self._current: Optional[Modality] = None

    def ensure_only(self, modality: Modality, *, force: bool = False) -> None:
        """
        Best-effort guarantee that only `modality` heavy models are on GPU.

        PR 1: No-op stub. Real implementation comes when we have real
        ImageCapability + MusicCapability adapters.
        """
        # TODO(PR1/PR2): call existing unload paths, coordinate with JobManager
        self._current = modality

    @property
    def current_modality(self) -> Optional[Modality]:
        return self._current


# Singleton for early use
vram = VRAMCoordinator()