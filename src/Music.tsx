import { useCallback, useEffect, useMemo, useState } from "react";
import {
  addSongsToPlaylist,
  audioGenerate,
  audioHealth,
  bulkDeleteSongs,
  cancelAudioJob,
  createPlaylist,
  createWorkspace,
  deleteSong as deleteSongApi,
  getAudioJob,
  getPlaylists,
  getSongLibrary,
  type AudioHealth,
  type ModelListing,
  type Playlist,
  type Song,
  type SongStudioHealth,
  type SongStudioModel,
  songStreamUrl,
  songStudioHealth,
} from "./api/sidecar";

// Filter that's currently active in the left sidebar. `null` = show everything.
type LibraryFilter =
  | { type: "workspace"; workspaceId: string; workspaceTitle: string }
  | { type: "playlist"; playlistId: string; playlistTitle: string }
  | null;

interface MusicProps {
  models: ModelListing | null;
  sidecar: string; // "up" | "down" | "checking"
}

// Format helpers --------------------------------------------------------------

function fmtDuration(seconds?: number | null): string {
  if (!seconds || seconds <= 0) return "?:??";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function coverUrl(song: Song): string | null {
  // Song Studio serves arbitrary files via /api/audio?path=... — we proxy that
  // through our /api/audio/stream. Works for both audio and image bytes because
  // the upstream just sets Content-Type from the file extension.
  const p = song.coverPath || song.cover_image || song.cover_path;
  return p ? songStreamUrl(p) : null;
}

function firstAudioPath(song: Song): string | null {
  if (song.audioPath) return song.audioPath;
  if (song.audio_path) return song.audio_path;
  if (Array.isArray(song.audioPaths) && song.audioPaths.length > 0) return song.audioPaths[0];
  if (Array.isArray((song as any).audio_paths) && (song as any).audio_paths.length > 0) {
    return (song as any).audio_paths[0];
  }
  return null;
}

// Main component --------------------------------------------------------------

export default function Music({ models, sidecar }: MusicProps) {
  // ---- Library state (B1) ---------------------------------------------------
  const [library, setLibrary] = useState<Song[]>([]);
  const [libraryLoading, setLibraryLoading] = useState(false);
  const [libraryError, setLibraryError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");
  const [activeSong, setActiveSong] = useState<Song | null>(null);
  const [bulkBusy, setBulkBusy] = useState(false);

  // ---- Sidebar state (B2) --------------------------------------------------
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [activeFilter, setActiveFilter] = useState<LibraryFilter>(null);
  const [newPlaylistName, setNewPlaylistName] = useState("");
  const [creatingPlaylist, setCreatingPlaylist] = useState(false);
  const [addToPlaylistOpen, setAddToPlaylistOpen] = useState(false);

  // ---- Song Studio health + model catalog ----------------------------------
  const [ssHealth, setSsHealth] = useState<SongStudioHealth | null>(null);
  const [ssModels, setSsModels] = useState<SongStudioModel[]>([]);

  // ---- Form state (generation; right pane — unchanged behavior) -----------
  const [prompt, setPrompt] = useState("cinematic space odyssey, vast choirs, pulsing synths, emotional climax");
  const [lyrics, setLyrics] = useState("");
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [bpm, setBpm] = useState(128);
  const [duration, setDuration] = useState(75);
  const [temperature, setTemperature] = useState(0.9);
  const [generateCover, setGenerateCover] = useState(true);
  const [coverPrompt, setCoverPrompt] = useState("");

  // ---- Job / progress state -----------------------------------------------
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [progressMsg, setProgressMsg] = useState("");
  const [busy, setBusy] = useState(false);

  // ---- ACE service health (legacy, bare engine) ---------------------------
  const [aceHealth, setAceHealth] = useState<AudioHealth | null>(null);
  const [aceChecking, setAceChecking] = useState(false);

  // ---- Library loader ------------------------------------------------------
  const reloadLibrary = useCallback(async () => {
    if (sidecar !== "up") return;
    setLibraryLoading(true);
    setLibraryError(null);
    try {
      const data = await getSongLibrary();
      const songs = Array.isArray(data.songs) ? data.songs : [];
      setLibrary(songs);
    } catch (e: any) {
      setLibraryError(String(e?.message ?? e));
    } finally {
      setLibraryLoading(false);
    }
  }, [sidecar]);

  const reloadPlaylists = useCallback(async () => {
    if (sidecar !== "up") return;
    try {
      const data = await getPlaylists();
      setPlaylists(Array.isArray(data.playlists) ? data.playlists : []);
    } catch (e) {
      // Non-fatal — sidebar just shows empty playlists section.
      setPlaylists([]);
    }
  }, [sidecar]);

  // Initial load + Song Studio health probe ----------------------------------
  useEffect(() => {
    if (sidecar !== "up") return;
    reloadLibrary();
    reloadPlaylists();
    (async () => {
      try {
        const h = await songStudioHealth();
        setSsHealth(h);
        if (h.ok && h.config?.generationModels) {
          setSsModels(h.config.generationModels);
          if (!selectedModelId && h.config.defaultGenerationModel) {
            setSelectedModelId(h.config.defaultGenerationModel);
          }
        }
      } catch (e: any) {
        setSsHealth({ ok: false, base_url: "?", error: String(e?.message ?? e) });
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sidecar]);

  // ---- Selection helpers ---------------------------------------------------
  const toggleSelected = useCallback((songId: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(songId)) next.delete(songId);
      else next.add(songId);
      return next;
    });
  }, []);
  const clearSelection = useCallback(() => setSelectedIds(new Set()), []);
  const selectAll = useCallback(() => {
    setSelectedIds(new Set(filtered.map(s => s.id)));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [library, search]);

  // ---- Workspaces derived from library (B2) --------------------------------
  // Song Studio doesn't expose a GET /workspaces — songs are the source of
  // truth, each carrying workspaceId + workspaceTitle. We derive distinct
  // entries here so the sidebar lists every workspace the user actually has
  // songs in.
  const workspaces = useMemo(() => {
    const seen = new Map<string, { id: string; title: string; count: number }>();
    for (const s of library) {
      const id = s.workspaceId || "";
      const title = s.workspaceTitle || "(no workspace)";
      const key = id || title;
      const entry = seen.get(key);
      if (entry) entry.count++;
      else seen.set(key, { id: id || title, title, count: 1 });
    }
    return Array.from(seen.values()).sort((a, b) => a.title.localeCompare(b.title));
  }, [library]);

  // ---- Filtered library (search + sidebar filter) --------------------------
  const filtered = useMemo(() => {
    let xs = library;

    // Apply sidebar filter first.
    if (activeFilter?.type === "workspace") {
      xs = xs.filter(s =>
        (s.workspaceId && s.workspaceId === activeFilter.workspaceId) ||
        s.workspaceTitle === activeFilter.workspaceTitle
      );
    } else if (activeFilter?.type === "playlist") {
      const playlist = playlists.find(p => p.id === activeFilter.playlistId);
      const ids = new Set(playlist?.songIds || playlist?.songs?.map(s => s.id) || []);
      xs = xs.filter(s => ids.has(s.id));
    }

    // Then search.
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      xs = xs.filter(s =>
        (s.title || "").toLowerCase().includes(q) ||
        (s.workspaceTitle || "").toLowerCase().includes(q) ||
        (s.summary || "").toLowerCase().includes(q) ||
        (s.prompt || "").toLowerCase().includes(q)
      );
    }
    return xs;
  }, [library, search, activeFilter, playlists]);

  // ---- Bulk actions --------------------------------------------------------
  async function onBulkDelete() {
    if (selectedIds.size === 0) return;
    const n = selectedIds.size;
    if (!window.confirm(`Delete ${n} song${n === 1 ? "" : "s"} permanently? This removes the audio files from disk.`)) return;
    setBulkBusy(true);
    try {
      const ids = Array.from(selectedIds);
      await bulkDeleteSongs(ids);
      // Optimistic: drop deleted ids from local library state, then reload to be sure.
      setLibrary(prev => prev.filter(s => !selectedIds.has(s.id)));
      setSelectedIds(new Set());
      if (activeSong && ids.includes(activeSong.id)) setActiveSong(null);
      reloadLibrary();
    } catch (e: any) {
      window.alert(`Bulk delete failed: ${e?.message ?? e}`);
    } finally {
      setBulkBusy(false);
    }
  }

  async function onCreatePlaylist() {
    const name = newPlaylistName.trim();
    if (!name) return;
    setCreatingPlaylist(true);
    try {
      await createPlaylist(name);
      setNewPlaylistName("");
      await reloadPlaylists();
    } catch (e: any) {
      window.alert(`Create playlist failed: ${e?.message ?? e}`);
    } finally {
      setCreatingPlaylist(false);
    }
  }

  async function onAddSelectedToPlaylist(playlistId: string, playlistTitle: string) {
    if (selectedIds.size === 0) return;
    const ids = Array.from(selectedIds);
    try {
      await addSongsToPlaylist(playlistId, ids);
      setAddToPlaylistOpen(false);
      // Reload playlists so the in-memory songIds reflect the new state for
      // the playlist filter to work immediately.
      await reloadPlaylists();
      // Friendly confirmation in the toolbar status spot — UX nicety; could
      // be a toast in a future iteration.
      window.alert(`Added ${ids.length} song${ids.length === 1 ? "" : "s"} to "${playlistTitle}".`);
    } catch (e: any) {
      window.alert(`Add to playlist failed: ${e?.message ?? e}`);
    }
  }

  async function onCreateWorkspace() {
    const name = window.prompt("New workspace name:");
    if (!name?.trim()) return;
    try {
      await createWorkspace(name.trim());
      // Workspaces are derived from library — new empty workspace won't show
      // until a song is generated into it. Tell the user.
      window.alert(
        `Workspace "${name.trim()}" created. It will appear in this sidebar once a song is generated into it ` +
        `(Song Studio derives workspace lists from songs).`
      );
    } catch (e: any) {
      window.alert(`Create workspace failed: ${e?.message ?? e}`);
    }
  }

  async function onDeleteSingle(song: Song) {
    if (!window.confirm(`Delete "${song.title || song.id}" permanently?`)) return;
    try {
      await deleteSongApi(song.id);
      setLibrary(prev => prev.filter(s => s.id !== song.id));
      setSelectedIds(prev => {
        const next = new Set(prev);
        next.delete(song.id);
        return next;
      });
      if (activeSong?.id === song.id) setActiveSong(null);
    } catch (e: any) {
      window.alert(`Delete failed: ${e?.message ?? e}`);
    }
  }

  // ---- ACE engine health (right-pane helper) -------------------------------
  async function checkAce() {
    setAceChecking(true);
    try {
      const resp = await audioHealth();
      setAceHealth(resp);
    } catch (e: any) {
      setAceHealth({ ok: false, detail: String(e?.message ?? e) });
    } finally {
      setAceChecking(false);
    }
  }

  // ---- Generation flow -----------------------------------------------------
  async function submit() {
    if (sidecar !== "up") return;
    setBusy(true);
    setJobId(null);
    setJobStatus(null);
    setProgressMsg("Submitting to orchestrator...");

    try {
      const payload = {
        prompt,
        lyrics,
        ace_model: selectedModelId || null,
        bpm,
        key_scale: "C",
        duration,
        temperature,
        generate_cover: generateCover,
        cover_prompt: coverPrompt || null,
        thinking: false,
        sample_mode: false,
      };
      const resp = await audioGenerate(payload);
      const jid = resp.job_id;
      setJobId(jid);
      setJobStatus(resp.status);

      const poll = async () => {
        try {
          const snap = await getAudioJob(jid);
          setJobStatus(snap.status);
          if (snap.progress?.message) setProgressMsg(snap.progress.message);
          if (snap.result) {
            setProgressMsg("Complete — refreshing library...");
            setBusy(false);
            await reloadLibrary();
            return;
          }
          if (snap.error) {
            setProgressMsg(`Error: ${snap.error}`);
            setBusy(false);
            return;
          }
          if (snap.status === "failed" || snap.status === "cancelled") {
            setBusy(false);
            return;
          }
          setTimeout(poll, 1200);
        } catch (e: any) {
          setProgressMsg(`Poll error: ${e?.message ?? e}`);
          setBusy(false);
        }
      };
      setTimeout(poll, 800);
    } catch (e: any) {
      setProgressMsg(`Submit failed: ${e?.message ?? e}`);
      setBusy(false);
    }
  }

  async function cancel() {
    if (!jobId) return;
    try {
      await cancelAudioJob(jobId);
      setProgressMsg("Cancel requested");
    } catch (e: any) {
      setProgressMsg(`Cancel error: ${e?.message ?? e}`);
    }
  }

  // ---- Render --------------------------------------------------------------
  // Render: legacy `audio` model list (from filesystem scan) is unused in the
  // dropdown now — we use the Song Studio's live catalog instead. The scan
  // count stays in the left pane for parity but is informational.
  const fsAudioModelCount = models?.categories?.audio?.length ?? 0;

  return (
    <>
      {/* Left pane — workspaces + playlists sidebar (B2) ------------------- */}
      <aside className="pane left-system music-sidebar">
        <div className="music-sidebar-section">
          <div className="music-sidebar-header">
            <span>Library</span>
          </div>
          <button
            className={"music-filter-button" + (!activeFilter ? " active" : "")}
            onClick={() => setActiveFilter(null)}
          >
            All songs
            <span className="music-filter-count">{library.length}</span>
          </button>
        </div>

        <div className="music-sidebar-section">
          <div className="music-sidebar-header">
            <span>Workspaces</span>
            <button
              className="music-sidebar-add"
              onClick={onCreateWorkspace}
              title="Create a new workspace (it appears here once a song is in it)"
            >+</button>
          </div>
          {workspaces.length === 0 ? (
            <div className="muted small" style={{ padding: "4px 8px" }}>No songs yet.</div>
          ) : (
            workspaces.map((w) => {
              const isActive = activeFilter?.type === "workspace" && activeFilter.workspaceTitle === w.title;
              return (
                <button
                  key={w.id}
                  className={"music-filter-button" + (isActive ? " active" : "")}
                  onClick={() => setActiveFilter({ type: "workspace", workspaceId: w.id, workspaceTitle: w.title })}
                  title={w.title}
                >
                  <span className="music-filter-label">{w.title}</span>
                  <span className="music-filter-count">{w.count}</span>
                </button>
              );
            })
          )}
        </div>

        <div className="music-sidebar-section">
          <div className="music-sidebar-header">
            <span>Playlists</span>
            <button onClick={reloadPlaylists} className="music-sidebar-add" title="Refresh">⟳</button>
          </div>
          {playlists.length === 0 ? (
            <div className="muted small" style={{ padding: "4px 8px" }}>None yet.</div>
          ) : (
            playlists.map((p) => {
              const isActive = activeFilter?.type === "playlist" && activeFilter.playlistId === p.id;
              const count = p.songIds?.length ?? p.songs?.length ?? 0;
              return (
                <button
                  key={p.id}
                  className={"music-filter-button" + (isActive ? " active" : "")}
                  onClick={() => setActiveFilter({ type: "playlist", playlistId: p.id, playlistTitle: p.title })}
                  title={p.title}
                >
                  <span className="music-filter-label">{p.title}</span>
                  <span className="music-filter-count">{count}</span>
                </button>
              );
            })
          )}
          <div className="new-playlist-form">
            <input
              type="text"
              placeholder="New playlist name..."
              value={newPlaylistName}
              onChange={(e) => setNewPlaylistName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") onCreatePlaylist(); }}
              disabled={creatingPlaylist}
            />
            <button
              onClick={onCreatePlaylist}
              disabled={creatingPlaylist || !newPlaylistName.trim()}
            >
              {creatingPlaylist ? "..." : "Add"}
            </button>
          </div>
        </div>
      </aside>

      {/* Center pane — Suno-style library grid + toolbar ------------------- */}
      <main className="pane center gen-center music-center">
        <div className="music-toolbar">
          <div className="music-toolbar-title">
            <span style={{ fontSize: 22 }}>♪</span>
            <strong>
              {activeFilter?.type === "workspace" && activeFilter.workspaceTitle}
              {activeFilter?.type === "playlist" && activeFilter.playlistTitle}
              {!activeFilter && "All songs"}
            </strong>
            <span className="muted small" style={{ marginLeft: 6 }}>
              {libraryLoading
                ? "loading..."
                : `${filtered.length}${filtered.length !== library.length ? ` of ${library.length}` : ""} song${filtered.length === 1 ? "" : "s"}`}
            </span>
            {activeFilter && (
              <button
                className="filter-clear"
                onClick={() => setActiveFilter(null)}
                title="Clear sidebar filter"
              >×</button>
            )}
          </div>
          <input
            className="music-search"
            type="text"
            placeholder="Search titles, workspaces, prompts..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <div className="music-toolbar-actions">
            {selectedIds.size > 0 ? (
              <>
                <span className="muted small">{selectedIds.size} selected</span>
                <div className="add-to-playlist-wrap">
                  <button
                    onClick={() => setAddToPlaylistOpen(o => !o)}
                    disabled={playlists.length === 0}
                    title={playlists.length === 0 ? "Create a playlist first (left sidebar)" : "Add selected to a playlist"}
                  >
                    Add to playlist ▾
                  </button>
                  {addToPlaylistOpen && playlists.length > 0 && (
                    <div className="add-to-playlist-menu">
                      {playlists.map((p) => (
                        <button
                          key={p.id}
                          className="add-to-playlist-item"
                          onClick={() => onAddSelectedToPlaylist(p.id, p.title)}
                        >
                          {p.title}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <button onClick={onBulkDelete} disabled={bulkBusy} className="danger">
                  {bulkBusy ? "Deleting..." : "Delete"}
                </button>
                <button onClick={clearSelection}>Clear</button>
              </>
            ) : (
              <>
                <button onClick={selectAll} disabled={filtered.length === 0}>Select all</button>
                <button onClick={reloadLibrary} disabled={libraryLoading}>
                  {libraryLoading ? "..." : "Refresh"}
                </button>
              </>
            )}
          </div>
        </div>

        {libraryError && (
          <div className="error-box">
            Library load failed: {libraryError}
            <button onClick={reloadLibrary} style={{ marginLeft: 8 }}>Retry</button>
          </div>
        )}

        {!libraryError && !libraryLoading && library.length === 0 && (
          <div className="muted" style={{ padding: 30, textAlign: "center" }}>
            No songs yet. Use the generation form on the right to create one — or check that
            Song Studio is running on port 8010.
          </div>
        )}

        <div className="song-grid">
          {filtered.map((song) => {
            const isSelected = selectedIds.has(song.id);
            const isActive = activeSong?.id === song.id;
            const cover = coverUrl(song);
            return (
              <div
                key={song.id}
                className={"song-card" + (isSelected ? " selected" : "") + (isActive ? " active" : "")}
                onClick={() => setActiveSong(song)}
              >
                <label
                  className="song-checkbox"
                  onClick={(e) => e.stopPropagation()}
                >
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => toggleSelected(song.id)}
                  />
                </label>

                <div className="song-cover">
                  {cover ? (
                    <img
                      src={cover}
                      alt=""
                      loading="lazy"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = "none";
                      }}
                    />
                  ) : (
                    <div className="song-cover-fallback">♪</div>
                  )}
                </div>

                <div className="song-info">
                  <div className="song-title" title={song.title}>
                    {song.title || "(untitled)"}
                  </div>
                  <div className="song-meta">
                    <span title={song.workspaceTitle}>{song.workspaceTitle || "—"}</span>
                    <span>·</span>
                    <span>{fmtDuration(song.duration)}</span>
                  </div>
                  {song.summary && (
                    <div className="song-summary muted small">{song.summary}</div>
                  )}
                </div>

                <div className="song-actions" onClick={(e) => e.stopPropagation()}>
                  <button
                    title="Delete this song"
                    className="icon danger-ghost"
                    onClick={() => onDeleteSingle(song)}
                  >
                    ✕
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Inline player (temporary — replaced by the persistent bottom bar in B3) */}
        {activeSong && (() => {
          const ap = firstAudioPath(activeSong);
          return (
            <div className="inline-player">
              <div style={{ fontSize: 13, marginBottom: 4 }}>
                <b>{activeSong.title || activeSong.id}</b>
                <span className="muted small" style={{ marginLeft: 8 }}>
                  {activeSong.workspaceTitle} · {fmtDuration(activeSong.duration)}
                </span>
              </div>
              {ap ? (
                <audio controls autoPlay src={songStreamUrl(ap)} style={{ width: "100%" }} />
              ) : (
                <div className="muted small">No playable audio file for this song.</div>
              )}
            </div>
          );
        })()}
      </main>

      {/* Right pane — generation form (unchanged behavior, scrollable) ----- */}
      <aside className="pane right-params" style={{ padding: 14, overflowY: "auto" }}>
        <div className="section-title">Song Studio</div>
        <div className="field">
          {ssHealth?.ok ? (
            <div style={{ color: "var(--c-good)", fontSize: 12 }}>
              Song Studio ✓ ({ssHealth.base_url})
              {ssHealth.config?.coverArtStatus?.toLowerCase().includes("offline") && (
                <div className="muted small" style={{ marginTop: 4 }}>
                  Cover-art legacy ComfyUI offline — Phase C will reroute to Kraken Art's FLUX/SDXL.
                </div>
              )}
            </div>
          ) : (
            <div style={{ color: "var(--c-bad)", fontSize: 12 }}>
              Song Studio not reachable at port 8010. Start it via the Kraken_Audio launcher.
              {ssHealth?.error && <div className="muted small" style={{ marginTop: 4 }}>{ssHealth.error}</div>}
            </div>
          )}
        </div>

        <div className="section-title" style={{ marginTop: 12 }}>Model</div>
        <div className="field">
          <select
            value={selectedModelId}
            onChange={(e) => setSelectedModelId(e.target.value)}
            style={{ width: "100%" }}
            disabled={ssModels.length === 0}
          >
            {ssModels.length === 0 && <option value="">(Song Studio not ready)</option>}
            {ssModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}{m.params ? ` — ${m.params}` : ""}{m.recommended ? " — recommended" : ""}
              </option>
            ))}
          </select>
          <div className="muted small" style={{ marginTop: 4 }}>
            {ssModels.length > 0
              ? `${ssModels.length} model(s) — live from Song Studio's catalog`
              : `${fsAudioModelCount} filesystem model(s) found; Song Studio not reporting catalog yet`}
          </div>
        </div>

        <div className="section-title" style={{ marginTop: 12 }}>Prompt</div>
        <div className="field">
          <textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        </div>

        <div className="section-title" style={{ marginTop: 8 }}>Lyrics (optional)</div>
        <div className="field">
          <textarea
            rows={5}
            value={lyrics}
            onChange={(e) => setLyrics(e.target.value)}
            placeholder="[Verse 1]&#10;..."
            style={{ fontFamily: "monospace", fontSize: 12 }}
          />
        </div>

        <div className="section-title" style={{ marginTop: 12 }}>Album Cover</div>
        <div className="field">
          <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}>
            <input
              type="checkbox"
              checked={generateCover}
              onChange={(e) => setGenerateCover(e.target.checked)}
            />
            Generate cover with Kraken Art (internal FLUX/SDXL)
          </label>
          {generateCover && (
            <input
              style={{ marginTop: 6, width: "100%" }}
              placeholder="Cover prompt override (optional)"
              value={coverPrompt}
              onChange={(e) => setCoverPrompt(e.target.value)}
            />
          )}
        </div>

        <div className="section-title" style={{ marginTop: 12 }}>Parameters</div>
        <div className="field-row">
          <div className="field">
            <label>BPM</label>
            <input type="number" value={bpm} onChange={(e) => setBpm(parseInt(e.target.value) || 120)} />
          </div>
          <div className="field">
            <label>Duration (s)</label>
            <input type="number" value={duration} onChange={(e) => setDuration(parseInt(e.target.value) || 60)} />
          </div>
          <div className="field">
            <label>Temperature</label>
            <input
              type="number"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value) || 0.85)}
            />
          </div>
        </div>

        <div className="action-row" style={{ marginTop: 14 }}>
          <button className="primary big" onClick={submit} disabled={busy || sidecar !== "up"}>
            {busy ? "Working..." : "♪ Generate"}
          </button>
          {jobId && <button onClick={cancel}>Cancel</button>}
        </div>

        {jobId && (
          <div className="job-status-box" style={{ marginTop: 12 }}>
            <div style={{ fontSize: 12 }}>
              <b>Job:</b> {jobId.slice(0, 8)}… &nbsp; <b>Status:</b> {jobStatus}
            </div>
            <div className="muted small" style={{ marginTop: 4 }}>{progressMsg}</div>
          </div>
        )}

        <div className="section-title" style={{ marginTop: 16 }}>Bare ACE-Step (legacy)</div>
        <div className="field">
          <button onClick={checkAce} disabled={aceChecking || sidecar !== "up"} style={{ fontSize: 12, width: "100%" }}>
            {aceChecking ? "Checking..." : "Check ACE on port 8001"}
          </button>
          {aceHealth && (
            <div
              style={{
                color: aceHealth.ok ? "var(--c-good)" : "var(--c-bad)",
                fontSize: 11,
                marginTop: 4,
              }}
            >
              {aceHealth.ok ? "ACE healthy" : "ACE unreachable"}
            </div>
          )}
        </div>

        <div style={{ marginTop: 18, fontSize: 11, color: "var(--muted)" }}>
          Heavy synthesis runs in the existing Kraken_Audio process. Kraken Art only orchestrates +
          generates covers. Library views the same files Song Studio knows about.
        </div>
      </aside>
    </>
  );
}
