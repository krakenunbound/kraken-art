# Design Document Summary

**Document:** `Documentation/design-runs/kraken-unified-ecosystem-architecture-6f9f838e.md`
**Project:** Kraken Art (F:\Kraken Art, branch `feature/audio-integration`)
**Date:** 2026-05-27 (post full source + docs audit)

## What Was Produced
A complete, implementable systems architecture design document that transforms the current "chaos barely contained" state (rapid pragmatic audio port from F:\Kraken_Audio layered on mature image pipelines) into a smooth, unified, extensible creative workstation platform.

The design directly addresses the user's stated vision (one-stop shop for art + SFX + music, manual or fully AI/agent-controlled) and the documented North Star (robust local ComfyUI replacement + first-class MCP/agent control surface per TODO.md task #21).

## Core Deliverable
**Unified Capability / Modality Registry** (`python/platform/registry.py` + adapters under `python/platform/capabilities/`) as the single source of truth consumed by:
- The existing React GUI (tabs, forms, health)
- The new first-class MCP server surface (stdio primary + optional SSE)
- Future modalities (video, voice, SFX, 3D, etc.)

All generation paths share consistent JobManager lifecycle, progress events, cancel semantics, and VRAM coordination.

## Key Problems Solved (Grounded in Audit)
- B-011, B-013 (cover generation hardcodes + cancel blindness in `pipelines/audio/orchestrator.py`)
- B-012 (MP3 download re-glob instead of exact path in `api/audio.py`)
- B-014 (ffmpeg only checked at export time)
- B-015 (unhardened proxy streaming)
- Systemic duplication (health, models, VRAM, progress, cancel) between `api/generate.py` + pipelines vs `api/audio.py` + audio proxies
- No discoverable surface for MCP agents

## Architecture Highlights
- **Clean layers**: Platform services (JobManager, VRAMCoordinator, Registry, ModelScanner) vs modality adapters (ImageCapability, MusicCapability) vs presentation (GUI + MCP tools auto-derived from registry).
- **Migration**: Purely incremental and additive. Existing `/api/generate`, `/api/audio/*`, Song Studio `KRAKEN_COVER_URL` contracts, and all working image/audio UX remain intact at every step. No throwaway of the valuable audio port or FLUX streaming work.
- **MCP**: `python -m kraken.mcp.stdio` (or SSE) exposes registry-derived tools + high-level composers (e.g. `compose_full_song` demonstrating end-to-end chaining). Agents (Claude Desktop, Grok, etc.) can drive the workstation reliably.
- **GUI + Agent parity**: Both are peers using the same registry/jobs/WS surface. Human use stays polished and manual; agents get discoverability and reliable long-running workflows.
- **Diagrams**: Current (duplicated ad-hoc) vs target (registry-centric) Mermaid graphs included.
- **Quantified**: References exact lines (e.g. `orchestrator.py:69`, `audio.py:438`, `cover_art.py:139`), file paths, performance numbers from PIPELINES.md, and the full Cursor-audio-adaption.md history/contract.

## Mandatory Sections Included
- Title/Metadata, Overview, Background & Motivation (user quote + audit), Goals/Non-Goals
- Detailed Proposed Design with Mermaid diagrams and code snippets for critical interfaces
- API/Interface Changes, Data Model Changes (additive only)
- Alternatives Considered (3, with trade-off analysis vs concrete files)
- Security & Privacy (threat model for MCP + local-only mitigations)
- Observability, Rollout Plan (env-gated + PR-staged)
- Open Questions
- References (full citations to TODO.md, Cursor-audio-adaption.md, BUGS.md, ARCHITECTURE.md, specific source files)
- **Key Decisions** (6 major choices with rationale, e.g. registry in Python sidecar, keep thin audio proxies, stdio MCP)
- **PR Plan** (7 ordered, independently reviewable/mergeable PRs with titles, affected files, dependencies, short descriptions — starts with platform abstractions and ends with docs/tests)

## Grounding
Every claim cites real files and patterns from the 2026-05-27 audit:
- `python/jobs.py`, `python/api/{audio.py, generate.py, cover_art.py, models.py, settings.py}`, `python/pipelines/audio/orchestrator.py`, `python/pipelines/{flux.py, sdxl.py}`, `python/main.py`, `python/config_store.py`
- `src/Music.tsx`, `src/App.tsx`, `src/Generate.tsx`, `src/api/sidecar.ts`
- `src-tauri/src/lib.rs`
- All key Documentation/*.md files

The design is concrete enough that an engineer could begin implementation the following week.

## Summary Files
- Full design: `F:\Kraken Art\Documentation\design-runs\kraken-unified-ecosystem-architecture-6f9f838e.md`
- This summary: `F:\Kraken Art\Documentation\design-runs\kraken-unified-ecosystem-architecture-6f9f838e.summary.md`

Ready for reviewer subagent and user review / resolution of open questions before PR work begins.