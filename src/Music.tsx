import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  addSongsToPlaylist,
  audioGenerate,
  audioHealth,
  bulkDeleteSongs,
  cancelAudioJob,
  createPlaylist,
  createWorkspace,
  deleteSong as deleteSongApi,
  exportSongMp3,
  exportSongsBulk,
  exportedMp3DownloadUrl,
  generateDialogue,
  generateSpeech,
  getAudioEngines,
  getAudioProviders,
  getAudioJob,
  getPlaylists,
  getSongLibrary,
  getTtsVoices,
  localAudioFileUrl,
  openJobWS,
  type AudioProvider,
  type AudioEngine,
  type AudioHealth,
  type ModelListing,
  type Playlist,
  type Song,
  type SongStudioHealth,
  type SongStudioModel,
  type TtsVoice,
  songStreamUrl,
  songStudioHealth,
  uploadTtsVoice,
} from "./api/sidecar";

// Filter that's currently active in the left sidebar. `null` = show everything.
type LibraryFilter =
  | { type: "workspace"; workspaceId: string; workspaceTitle: string }
  | { type: "playlist"; playlistId: string; playlistTitle: string }
  | null;

type AudioWorkflow = "song" | "instrumental" | "sfx" | "speech" | "dialogue";

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

function resultAudioSrc(result: string | null): string | null {
  if (!result) return null;
  if (/^https?:\/\//i.test(result)) return result;
  if (/^[A-Za-z]:\\/.test(result) || result.startsWith("/")) return localAudioFileUrl(result);
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

  // ---- Persistent player bar state (B3) ------------------------------------
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [volume, setVolume] = useState(0.8);

  // ---- MP3 export state (Phase D) ------------------------------------------
  // singleExportBusy is keyed by song.id so multiple cards can show their own
  // spinners (you can fire a row export while a bulk job is also running).
  const [singleExportBusy, setSingleExportBusy] = useState<Set<string>>(new Set());
  // Bulk export tracking — shows a banner above the grid with live progress.
  type BulkProgress = {
    jobId: string;
    total: number;
    done: number;
    failed: number;
    currentTitle: string;
    finished: boolean;
  };
  const [bulkExport, setBulkExport] = useState<BulkProgress | null>(null);

  // ---- Song Studio health + model catalog ----------------------------------
  const [ssHealth, setSsHealth] = useState<SongStudioHealth | null>(null);
  const [ssModels, setSsModels] = useState<SongStudioModel[]>([]);

  // ---- Audio engine selector (sidecar-managed, lazy) -----------------------
  // ACE-Step (vocals) vs Stable Audio 3 (instrumental). The picked engine loads
  // on Generate; the other is killed first. Nothing here loads a model.
  const [engines, setEngines] = useState<AudioEngine[]>([]);
  const [engineId, setEngineId] = useState<string>("ace_step");
  const [workflow, setWorkflow] = useState<AudioWorkflow>("song");
  const [providers, setProviders] = useState<AudioProvider[]>([]);
  const selectedEngine = useMemo(() => engines.find((e) => e.id === engineId), [engines, engineId]);
  const engineChoices = useMemo(() => {
    return engines.filter((e) => {
      if (workflow === "song") return e.capability === "song";
      if (workflow === "instrumental") return e.id === "stable_audio_3";
      if (workflow === "sfx") return e.capability === "sfx" || e.id === "stable_audio_3";
      if (workflow === "speech") return e.capability === "speech";
      if (workflow === "dialogue") return e.id === "tts_luxtts";
      return true;
    });
  }, [engines, workflow]);

  // ---- Form state (generation; right pane — unchanged behavior) -----------
  const [prompt, setPrompt] = useState("cinematic space odyssey, vast choirs, pulsing synths, emotional climax");
  const [lyrics, setLyrics] = useState("");
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [bpm, setBpm] = useState(128);
  const [duration, setDuration] = useState(75);
  const [steps, setSteps] = useState(8);
  const [temperature, setTemperature] = useState(0.9);
  const [generateCover, setGenerateCover] = useState(true);
  const [coverPrompt, setCoverPrompt] = useState("");
  const [ttsVoices, setTtsVoices] = useState<TtsVoice[]>([]);
  const [ttsVoicesLoading, setTtsVoicesLoading] = useState(false);
  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const [voiceName, setVoiceName] = useState("");
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [speechText, setSpeechText] = useState("The tavern fell silent as the door creaked open.");
  const [dialogueScript, setDialogueScript] = useState("[Narrator] The tavern fell silent as the door creaked open.\n[Guard] Halt! Who goes there?\n[Traveler] Just a weary wanderer, seeking shelter from the storm.");
  const [speakerMapText, setSpeakerMapText] = useState("Narrator=\nGuard=\nTraveler=");
  const [ttsSteps, setTtsSteps] = useState(4);
  const [ttsShift, setTtsShift] = useState(0.5);
  const [ttsSpeed, setTtsSpeed] = useState(1.0);
  const [audioLabResult, setAudioLabResult] = useState<string | null>(null);

  const reloadEngines = useCallback(async () => {
    if (sidecar !== "up") return;
    try {
      const data = await getAudioEngines();
      setEngines(data.engines);
      // Keep the selection valid; prefer an available engine.
      setEngineId((cur) => {
        if (data.engines.some((e) => e.id === cur && e.available)) return cur;
        return data.engines.find((e) => e.available)?.id ?? cur;
      });
    } catch {
      /* sidecar not ready — selector shows nothing until it is */
    }
  }, [sidecar]);

  const reloadProviders = useCallback(async () => {
    if (sidecar !== "up") return;
    try {
      const data = await getAudioProviders();
      setProviders(data.providers);
    } catch {
      setProviders([]);
    }
  }, [sidecar]);

  async function loadTtsVoices() {
    setTtsVoicesLoading(true);
    try {
      const data = await getTtsVoices();
      const voices = [...(data.preset || []), ...(data.custom || [])];
      setTtsVoices(voices);
      if (!selectedVoiceId && voices.length > 0) setSelectedVoiceId(voices[0].id);
    } catch (e: any) {
      window.alert(`Load voices failed: ${e?.message ?? e}`);
    } finally {
      setTtsVoicesLoading(false);
    }
  }

  async function onUploadVoice() {
    if (!voiceFile) {
      window.alert("Choose a clean reference audio clip first.");
      return;
    }
    const name = voiceName.trim() || voiceFile.name.replace(/\.[^.]+$/, "");
    setTtsVoicesLoading(true);
    try {
      const voice = await uploadTtsVoice(voiceFile, name);
      setTtsVoices(prev => [...prev, voice]);
      setSelectedVoiceId(voice.id);
      setVoiceName("");
      setVoiceFile(null);
    } catch (e: any) {
      window.alert(`Upload voice failed: ${e?.message ?? e}`);
    } finally {
      setTtsVoicesLoading(false);
    }
  }

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

  // Initial load -------------------------------------------------------------
  // Keep launch-side effects to a minimum: library/playlists come from the
  // local Kraken sidecar, while Song Studio / ACE probes only happen on
  // explicit user action or when a generation is actually submitted.
  useEffect(() => {
    if (sidecar !== "up") return;
    reloadLibrary();
    reloadPlaylists();
    reloadEngines();
    reloadProviders();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sidecar]);

  async function loadSongStudioCatalog() {
    try {
      const h = await songStudioHealth();
      setSsHealth(h);
      if (h.ok && h.config?.generationModels) {
        setSsModels(h.config.generationModels);
        if (!selectedModelId && h.config.defaultGenerationModel) {
          setSelectedModelId(h.config.defaultGenerationModel);
        }
      }
      return h;
    } catch (e: any) {
      const err = String(e?.message ?? e);
      setSsHealth({ ok: false, base_url: "?", error: err });
      throw e;
    }
  }

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

  // ---- MP3 export handlers (Phase D) ---------------------------------------
  // Single-song flow:
  //   1. POST /api/audio/songs/{id}/export-mp3   — synchronous, returns the
  //      result blob with mp3_url + size + bitrate + cover_embedded.
  //   2. Open mp3_url in a new tab so the browser saves it (the endpoint
  //      sets Content-Disposition: attachment, so it's a true download).
  // Per-card spinner is keyed on song.id so simultaneous exports are fine.
  async function onExportSingle(song: Song) {
    if (singleExportBusy.has(song.id)) return;
    setSingleExportBusy(prev => new Set(prev).add(song.id));
    try {
      const r = await exportSongMp3(song.id, { overwrite: false });
      // Kick off the browser download via a synthetic <a download>. We use
      // the dedicated download endpoint (not the mp3_url static file) so
      // Content-Disposition is honoured even on browsers that prefer to
      // open MP3s inline.
      const a = document.createElement("a");
      a.href = exportedMp3DownloadUrl(song.id);
      a.download = `${song.title || song.id}.mp3`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      // Small toast-style alert with the useful facts. Could be a real
      // toast component later; alert keeps this commit minimal.
      window.alert(
        `Exported "${song.title || song.id}":\n` +
        `  ${Math.round(r.size_bytes / 1024)} KB at ${r.bitrate_avg_kbps} kbps (VBR V0)\n` +
        `  Cover embedded: ${r.cover_embedded ? "yes" : "no"}\n` +
        `  Encoded in ${r.elapsed_s}s`
      );
    } catch (e: any) {
      window.alert(`Export failed for "${song.title || song.id}":\n${e?.message ?? e}`);
    } finally {
      setSingleExportBusy(prev => {
        const next = new Set(prev);
        next.delete(song.id);
        return next;
      });
    }
  }

  // Bulk-export flow:
  //   1. POST /api/audio/songs/export-mp3 with the selected IDs.
  //   2. Subscribe to /ws/jobs/{job_id} for per-song progress + final summary.
  // We render a small banner above the song grid with "8 of 30 exported".
  async function onExportSelected() {
    if (selectedIds.size === 0) return;
    const ids = Array.from(selectedIds);
    try {
      const { job_id } = await exportSongsBulk(ids, false);
      const initial: BulkProgress = {
        jobId: job_id, total: ids.length, done: 0, failed: 0,
        currentTitle: "", finished: false,
      };
      setBulkExport(initial);

      const ws = await openJobWS(job_id);
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "bulk_progress") {
            setBulkExport(prev => prev && ({
              ...prev,
              currentTitle: String(msg.message || "").replace(/^Exporting \d+\/\d+: /, ""),
            }));
          } else if (msg.type === "bulk_song_done") {
            setBulkExport(prev => prev && ({ ...prev, done: prev.done + 1 }));
          } else if (msg.type === "bulk_song_failed") {
            setBulkExport(prev => prev && ({ ...prev, failed: prev.failed + 1 }));
          } else if (msg.type === "status" && msg.status === "succeeded") {
            setBulkExport(prev => prev && ({ ...prev, finished: true }));
            ws.close();
          } else if (msg.type === "status" && (msg.status === "failed" || msg.status === "cancelled")) {
            setBulkExport(prev => prev && ({ ...prev, finished: true }));
            ws.close();
          }
        } catch {
          // Non-JSON frames (shouldn't happen) — ignore.
        }
      };
      ws.onerror = () => {
        // The WS may close before the job's final 'status' event reaches us
        // (browser shutting it down, network blip). Don't change finished —
        // a stale banner is better than a misleading "complete" claim.
      };
    } catch (e: any) {
      window.alert(`Bulk export failed to start: ${e?.message ?? e}`);
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

  // ---- Player bar (B3) -----------------------------------------------------
  // Keep <audio>.volume in sync with the slider state. Native audio holds its
  // own volume so we set it imperatively on the element ref.
  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = volume;
  }, [volume]);

  // When the active song changes, reset transport state and let the audio
  // element kick off a new load. The autoPlay attribute on the element will
  // start playback once loadedmetadata fires (browsers handle the rest).
  useEffect(() => {
    setCurrentTime(0);
    setAudioDuration(0);
    setPlaying(false);
  }, [activeSong?.id]);

  function togglePlayPause() {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) {
      el.play().catch(() => {});
    } else {
      el.pause();
    }
  }

  function playPrev() {
    if (!activeSong) return;
    const idx = filtered.findIndex((s) => s.id === activeSong.id);
    if (idx < 0) return;
    const prev = filtered[(idx - 1 + filtered.length) % filtered.length];
    setActiveSong(prev);
  }
  function playNext() {
    if (!activeSong) return;
    const idx = filtered.findIndex((s) => s.id === activeSong.id);
    if (idx < 0) return;
    const next = filtered[(idx + 1) % filtered.length];
    setActiveSong(next);
  }

  function onSeek(e: React.ChangeEvent<HTMLInputElement>) {
    const el = audioRef.current;
    if (!el || !isFinite(audioDuration) || audioDuration <= 0) return;
    const pct = parseFloat(e.target.value);
    const t = (pct / 100) * audioDuration;
    el.currentTime = t;
    setCurrentTime(t);
  }

  // ---- ACE engine health (right-pane helper) -------------------------------
  async function checkAce() {
    setAceChecking(true);
    try {
      if (ssHealth === null) {
        await loadSongStudioCatalog();
      }
      const resp = await audioHealth();
      setAceHealth(resp);
    } catch (e: any) {
      setAceHealth({ ok: false, detail: String(e?.message ?? e) });
    } finally {
      setAceChecking(false);
    }
  }

  function preferredEngine(ids: string[]): string {
    return ids.find((id) => engines.some((e) => e.id === id && e.available)) || ids[0];
  }

  function onWorkflowChange(next: AudioWorkflow) {
    setWorkflow(next);
    if (next === "song") {
      setEngineId(preferredEngine(["ace_step", "heartmula"]));
      setDuration(75);
    }
    if (next === "instrumental") {
      setEngineId("stable_audio_3");
      setDuration(60);
      setSteps(8);
      setGenerateCover(false);
    }
    if (next === "sfx") {
      setEngineId(preferredEngine(["moss_sfx", "stable_audio_3"]));
      setDuration(5);
      setSteps(50);
      setGenerateCover(false);
    }
    if (next === "speech") {
      setEngineId(preferredEngine(["moss_tts", "tts_luxtts"]));
      setGenerateCover(false);
    }
    if (next === "dialogue") {
      setEngineId("tts_luxtts");
      setGenerateCover(false);
    }
  }

  function parseSpeakerMap(): Record<string, string> {
    const out: Record<string, string> = {};
    for (const raw of speakerMapText.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line) continue;
      const idx = line.indexOf("=");
      if (idx < 0) continue;
      const role = line.slice(0, idx).trim();
      const voice = line.slice(idx + 1).trim();
      if (role && voice) out[role] = voice;
    }
    return out;
  }

  // ---- Generation flow -----------------------------------------------------
  async function submit() {
    if (sidecar !== "up") return;
    setBusy(true);
    setJobId(null);
    setJobStatus(null);
    setProgressMsg("Submitting to orchestrator...");
    setAudioLabResult(null);

    try {
      if (workflow === "speech") {
        const provider = engineId === "moss_tts" ? "moss_tts" : "tts_luxtts";
        if (provider !== "moss_tts" && !selectedVoiceId) throw new Error("Select or upload a voice reference first.");
        setProgressMsg(provider === "moss_tts" ? "Generating speech with MOSS-TTS..." : "Generating speech with LuxTTS...");
        const r = await generateSpeech({
          text: speechText,
          voice_id: selectedVoiceId,
          provider,
          speed: ttsSpeed,
          num_steps: ttsSteps,
          t_shift: ttsShift,
          ref_duration: 5,
        });
        setAudioLabResult(r?.result?.audio_url || r?.result?.path || r?.result?.url || JSON.stringify(r.result || r));
        setProgressMsg("Speech complete.");
        setBusy(false);
        reloadEngines();
        return;
      }

      if (workflow === "dialogue") {
        if (engineId !== "tts_luxtts") throw new Error("Multi-voice dialogue is currently wired to LuxTTS voice mapping.");
        const speakers = parseSpeakerMap();
        setProgressMsg("Generating multi-voice dialogue with LuxTTS...");
        const r = await generateDialogue({
          script: dialogueScript,
          speakers,
          speed: ttsSpeed,
          num_steps: ttsSteps,
          t_shift: ttsShift,
          ref_duration: 5,
          gap_seconds: 0.35,
        });
        setAudioLabResult(r?.path || JSON.stringify(r));
        setProgressMsg(`Dialogue complete (${r?.line_count ?? "?"} lines).`);
        setBusy(false);
        reloadEngines();
        return;
      }

      const payload = {
        engine_id: engineId,
        prompt,
        lyrics: workflow === "song" ? lyrics : "",
        ace_model: workflow === "song" && engineId === "ace_step" ? (selectedModelId || null) : null,
        bpm,
        key_scale: "C",
        duration,
        steps,
        temperature,
        generate_cover: workflow === "song" && generateCover,
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
            reloadEngines();
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
                <button
                  onClick={onExportSelected}
                  disabled={bulkBusy || (bulkExport !== null && !bulkExport.finished)}
                  title="Export selected songs to MP3 (LAME VBR V0 + embedded cover)"
                >
                  Export MP3
                </button>
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

        {bulkExport && (
          <div className={"bulk-export-banner" + (bulkExport.finished ? " finished" : "")}>
            {bulkExport.finished ? (
              <>
                <span>
                  ✓ Exported {bulkExport.done} of {bulkExport.total}
                  {bulkExport.failed > 0 && <> &nbsp;·&nbsp; {bulkExport.failed} failed</>}
                </span>
                <span className="muted small">
                  Files in <code>outputs/exports/</code>
                </span>
                <button onClick={() => setBulkExport(null)}>Dismiss</button>
              </>
            ) : (
              <>
                <span>
                  Exporting MP3s — {bulkExport.done + bulkExport.failed} of {bulkExport.total}
                  {bulkExport.currentTitle && <> &nbsp;·&nbsp; <em>{bulkExport.currentTitle}</em></>}
                </span>
                <span className="muted small">VBR V0 (~240 kbps) with embedded cover</span>
              </>
            )}
          </div>
        )}

        {!libraryError && !libraryLoading && library.length === 0 && (
          <div className="muted" style={{ padding: 30, textAlign: "center" }}>
            No songs yet. Pick an engine and use the generation form on the right to create one.
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
                    title="Export to MP3 (LAME VBR V0 with embedded cover)"
                    className="icon"
                    onClick={() => onExportSingle(song)}
                    disabled={singleExportBusy.has(song.id)}
                  >
                    {singleExportBusy.has(song.id) ? "…" : "⬇"}
                  </button>
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

      </main>

      {/* Right pane — generation form (unchanged behavior, scrollable) ----- */}
      <aside className="pane right-params" style={{ padding: 14, overflowY: "auto" }}>
        <div className="section-title">Create</div>
        <div className="field">
          <select value={workflow} onChange={(e) => onWorkflowChange(e.target.value as AudioWorkflow)} style={{ width: "100%" }}>
            <option value="song">Music with vocals</option>
            <option value="instrumental">Instrumental music</option>
            <option value="sfx">Sound effect / ambience</option>
            <option value="speech">Text to voice</option>
            <option value="dialogue">Multi-voice dialogue</option>
          </select>
          <div className="muted small" style={{ marginTop: 4 }}>
            {workflow === "song" && "ACE-Step for local Suno-style songs, including duets when prompted in lyrics/style."}
            {workflow === "instrumental" && "Stable Audio 3 for instrumental beds and loops."}
            {workflow === "sfx" && "MOSS-SoundEffect for prompted effects and ambience; Stable Audio remains available as backup."}
            {workflow === "speech" && "MOSS-TTS for high-quality direct narration, or LuxTTS for reference-voice cloning."}
            {workflow === "dialogue" && "LuxTTS renders each [Role] line with the mapped voice and stitches the result."}
          </div>
        </div>

        {providers.length > 0 && (
          <div className="field" style={{ display: "grid", gap: 4 }}>
            {providers.filter(p => p.recommended || !p.installed).slice(0, 7).map((p) => (
              <div key={p.id} className="muted small">
                <b style={{ color: p.available ? "var(--c-good)" : "var(--muted)" }}>
                  {p.available ? "✓" : "○"} {p.label}
                </b>
                {" — "}{p.capability}
              </div>
            ))}
          </div>
        )}

        <div className="section-title">Engine</div>
        <div className="field">
          <select
            value={engineId}
            onChange={(e) => setEngineId(e.target.value)}
            style={{ width: "100%" }}
            disabled={engines.length === 0}
          >
            {engineChoices.length === 0 && <option value="">(loading engines…)</option>}
            {engineChoices.map((e) => (
              <option key={e.id} value={e.id} disabled={!e.available}>
                {e.label}{!e.available ? " — not installed" : ""}{e.running ? " ● loaded" : ""}{e.direct_runner ? " — direct" : ""}
              </option>
            ))}
          </select>
          <div className="muted small" style={{ marginTop: 4 }}>
            {selectedEngine?.description ?? "Pick an audio engine."}
            {selectedEngine && (
              <> Loads on Generate (~{selectedEngine.vram_hint_gb} GB); resident audio/image models are unloaded first.</>
            )}
          </div>
        </div>

        {workflow === "song" && engineId === "ace_step" && (<>
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
            <div style={{ color: "var(--muted)", fontSize: 12 }}>
              The sidecar starts Song Studio automatically when you open the library or
              load the catalog — no separate launcher needed. First start takes a few seconds.
              {ssHealth?.error && <div style={{ color: "var(--c-bad)", marginTop: 4 }}>{ssHealth.error}</div>}
            </div>
          )}
        </div>
        </>)}

        {workflow === "song" && engineId === "ace_step" && (<>
        <div className="section-title" style={{ marginTop: 12 }}>ACE-Step model (voice / DiT)</div>
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
          <button onClick={() => { void loadSongStudioCatalog(); }} disabled={sidecar !== "up"} style={{ marginTop: 8, fontSize: 12, width: "100%" }}>
            {ssModels.length > 0 ? "Refresh Song Studio catalog" : "Load Song Studio catalog"}
          </button>
        </div>
        </>)}

        {(workflow === "speech" || workflow === "dialogue") && engineId !== "moss_tts" && (<>
          <div className="section-title" style={{ marginTop: 12 }}>Voice Library</div>
          <div className="field">
            <button onClick={() => { void loadTtsVoices(); }} disabled={ttsVoicesLoading || sidecar !== "up"} style={{ width: "100%", fontSize: 12 }}>
              {ttsVoicesLoading ? "Loading voices..." : "Load voices / start TTS"}
            </button>
            <select
              value={selectedVoiceId}
              onChange={(e) => setSelectedVoiceId(e.target.value)}
              style={{ width: "100%", marginTop: 6 }}
              disabled={ttsVoices.length === 0}
            >
              {ttsVoices.length === 0 && <option value="">No voices loaded</option>}
              {ttsVoices.map((v) => <option key={v.id} value={v.id}>{v.name || v.id}</option>)}
            </select>
            <div style={{ display: "grid", gap: 6, marginTop: 8 }}>
              <input placeholder="Voice name" value={voiceName} onChange={(e) => setVoiceName(e.target.value)} />
              <input type="file" accept="audio/*" onChange={(e) => setVoiceFile(e.target.files?.[0] || null)} />
              <button onClick={() => { void onUploadVoice(); }} disabled={!voiceFile || ttsVoicesLoading} style={{ fontSize: 12 }}>
                Upload reference voice
              </button>
            </div>
          </div>
        </>)}

        {(workflow === "song" || workflow === "instrumental" || workflow === "sfx") && (<>
        <div className="section-title" style={{ marginTop: 12 }}>Prompt</div>
        <div className="field">
          <textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        </div>
        </>)}

        {(workflow === "instrumental" || workflow === "sfx") ? (
          <div className="muted small" style={{ marginTop: 2 }}>
            {workflow === "sfx" && engineId === "moss_sfx"
              ? "MOSS-SoundEffect is strongest around 1-30 second acoustic descriptions. Use 30-75 steps for quality."
              : "Stable Audio 3 is instrumental / SFX only. Use descriptive acoustic prompts and short durations for effects."}
          </div>
        ) : workflow === "song" ? (<>
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
        </>) : null}

        {workflow === "speech" && (<>
          <div className="section-title" style={{ marginTop: 12 }}>Text</div>
          <div className="field">
            <textarea rows={7} value={speechText} onChange={(e) => setSpeechText(e.target.value)} />
          </div>
        </>)}

        {workflow === "dialogue" && (<>
          <div className="section-title" style={{ marginTop: 12 }}>Script</div>
          <div className="field">
            <textarea rows={8} value={dialogueScript} onChange={(e) => setDialogueScript(e.target.value)} style={{ fontFamily: "monospace", fontSize: 12 }} />
          </div>
          <div className="section-title" style={{ marginTop: 8 }}>Voice Map</div>
          <div className="field">
            <textarea
              rows={4}
              value={speakerMapText}
              onChange={(e) => setSpeakerMapText(e.target.value)}
              placeholder={"Narrator=custom_voice_id\nGuard=custom_voice_id"}
              style={{ fontFamily: "monospace", fontSize: 12 }}
            />
          </div>
        </>)}

        {workflow === "song" && (<>
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
        </>)}

        <div className="section-title" style={{ marginTop: 12 }}>Parameters</div>
        {(workflow === "instrumental" || workflow === "sfx") ? (
          <div className="field-row">
            <div className="field">
              <label>Duration (s)</label>
              <input type="number" value={duration} onChange={(e) => setDuration(parseInt(e.target.value) || 30)} />
            </div>
            <div className="field">
              <label title="Sampling steps for Stable Audio 3">Steps</label>
              <input type="number" value={steps} onChange={(e) => setSteps(parseInt(e.target.value) || 8)} />
            </div>
          </div>
        ) : workflow === "song" ? (
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
        ) : (
          <div className="field-row">
            <div className="field">
              <label>Steps</label>
              <input type="number" value={ttsSteps} onChange={(e) => setTtsSteps(parseInt(e.target.value) || 4)} />
            </div>
            <div className="field">
              <label>Speed</label>
              <input type="number" step="0.05" value={ttsSpeed} onChange={(e) => setTtsSpeed(parseFloat(e.target.value) || 1)} />
            </div>
            <div className="field">
              <label>T-shift</label>
              <input type="number" step="0.05" value={ttsShift} onChange={(e) => setTtsShift(parseFloat(e.target.value) || 0.5)} />
            </div>
          </div>
        )}

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

        {!jobId && progressMsg && (
          <div className="job-status-box" style={{ marginTop: 12 }}>
            <div className="muted small">{progressMsg}</div>
            {audioLabResult && (
              <>
                {resultAudioSrc(audioLabResult) && (
                  <audio controls src={resultAudioSrc(audioLabResult) || undefined} style={{ width: "100%", marginTop: 8 }} />
                )}
                <div className="muted small" style={{ marginTop: 4, wordBreak: "break-all" }}>
                  Output: {audioLabResult}
                </div>
              </>
            )}
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

      {/* Persistent player bar (B3) — position: fixed; bottom: 0 in CSS so it
          floats over the rest of the page. Hidden when no song is active. */}
      {activeSong && (() => {
        const audioPath = firstAudioPath(activeSong);
        const cover = coverUrl(activeSong);
        const pct = audioDuration > 0 ? (currentTime / audioDuration) * 100 : 0;
        const totalForDisplay = audioDuration > 0
          ? audioDuration
          : (activeSong.duration ?? 0);
        return (
          <div className="music-player-bar" role="region" aria-label="Now playing">
            <div className="player-info">
              <div className="player-cover">
                {cover ? (
                  <img src={cover} alt="" onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
                ) : (
                  <div className="player-cover-fallback">♪</div>
                )}
              </div>
              <div className="player-meta">
                <div className="player-title" title={activeSong.title}>
                  {activeSong.title || activeSong.id}
                </div>
                <div className="player-sub muted small">
                  {activeSong.workspaceTitle || "—"}
                </div>
              </div>
              <button
                className="player-close"
                title="Close player"
                onClick={() => {
                  audioRef.current?.pause();
                  setActiveSong(null);
                }}
              >×</button>
            </div>

            <div className="player-transport">
              <div className="transport-buttons">
                <button onClick={playPrev} title="Previous" className="transport-btn">⏮</button>
                <button onClick={togglePlayPause} title={playing ? "Pause" : "Play"} className="transport-btn play">
                  {playing ? "⏸" : "▶"}
                </button>
                <button onClick={playNext} title="Next" className="transport-btn">⏭</button>
              </div>
              <div className="transport-scrubber">
                <span className="transport-time">{fmtDuration(currentTime)}</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={0.1}
                  value={pct}
                  onChange={onSeek}
                  disabled={!audioPath || audioDuration <= 0}
                  className="transport-range"
                />
                <span className="transport-time">{fmtDuration(totalForDisplay)}</span>
              </div>
            </div>

            <div className="player-volume">
              <span className="muted small">🔊</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={volume}
                onChange={(e) => setVolume(parseFloat(e.target.value))}
                className="volume-range"
                title={`Volume: ${Math.round(volume * 100)}%`}
              />
            </div>

            {/* Hidden controlled audio element — actual playback. */}
            {audioPath && (
              <audio
                ref={audioRef}
                src={songStreamUrl(audioPath)}
                autoPlay
                preload="auto"
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onEnded={() => { setPlaying(false); playNext(); }}
                onTimeUpdate={(e) => setCurrentTime((e.target as HTMLAudioElement).currentTime)}
                onLoadedMetadata={(e) => {
                  const el = e.target as HTMLAudioElement;
                  setAudioDuration(isFinite(el.duration) ? el.duration : 0);
                  el.volume = volume;
                }}
                style={{ display: "none" }}
              />
            )}
            {!audioPath && (
              <div className="muted small player-noaudio">No playable audio for this song.</div>
            )}
          </div>
        );
      })()}
    </>
  );
}
