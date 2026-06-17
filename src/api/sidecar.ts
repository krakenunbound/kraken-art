import { invoke } from "@tauri-apps/api/core";

// Resolved once after Tauri reports the sidecar URL.
let baseUrl: string | null = null;

async function base(): Promise<string> {
  if (baseUrl) return baseUrl;
  try {
    baseUrl = await invoke<string>("sidecar_url");
  } catch {
    // Outside Tauri (e.g. `npm run dev` in the browser) — fall back.
    baseUrl = "http://127.0.0.1:7780";
  }
  return baseUrl;
}

async function readError(r: Response, path: string): Promise<Error> {
  let detail = `${r.status} ${r.statusText}`.trim();
  try {
    const body = await r.json();
    if (body?.detail) {
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
  } catch { /* not JSON, leave status */ }
  return new Error(`${path}: ${detail}`);
}

async function getJSON<T>(path: string): Promise<T> {
  const u = (await base()) + path;
  const r = await fetch(u);
  if (!r.ok) throw await readError(r, path);
  return r.json() as Promise<T>;
}

async function postJSON<T>(path: string, body?: unknown): Promise<T> {
  const u = (await base()) + path;
  const r = await fetch(u, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw await readError(r, path);
  return r.json() as Promise<T>;
}

async function patchJSON<T>(path: string, body?: unknown): Promise<T> {
  const u = (await base()) + path;
  const r = await fetch(u, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw await readError(r, path);
  return r.json() as Promise<T>;
}

// ---------- Types ----------

export type Health = { ok: boolean; service: string; models_root: string; models_root_exists: boolean; outputs_root: string };

export type GpuInfo = {
  detected: boolean;
  vendor: string | null;
  name: string | null;
  vram_total_mb: number | null;
  vram_free_mb: number | null;
  gpu_utilization_percent: number | null;
  gpu_temperature_c: number | null;
  gpu_clock_mhz: number | null;
  gpu_clock_max_mhz: number | null;
  power_watts: number | null;
  power_limit_watts: number | null;
  throttled: boolean;
  throttle_reasons: string[];
  driver: string | null;
  cuda_runtime: string | null;
  cuda_available: boolean;
  torch_version: string | null;
  errors: string[];
};

export type DepPackage = { name: string; required: string; installed: string | null; ok: boolean; remedy: string | null };
export type DepsStatus = { all_ok: boolean; missing: number; packages: DepPackage[] };

export type ModelEntry = {
  name: string;
  filename: string;
  subdir: string;
  abs_path: string;
  size_bytes: number;
  ext: string;
  base_model?: string | null;
  source_type?: string | null;
  detected_arch?: string | null;
  experimental?: boolean | null;
  disabled_by_default?: boolean | null;
  warning?: string | null;
  recommended_settings?: {
    cfg?: number;
    cfg_range?: [number, number];
    steps?: number;
    steps_range?: [number, number];
    width?: number;
    height?: number;
    resolutions?: [number, number][];
    sampler?: string;
    scheduler?: string;
    source?: string;
  } | null;
};
export type ModelListing = {
  root: string;
  exists: boolean;
  audio_ace_root?: string;
  audio_ace_exists?: boolean;
  categories: Record<string, ModelEntry[]>;
  counts: Record<string, number>;
};

// ---------- Endpoints ----------

export const health         = () => getJSON<Health>("/health");
export const getGpu         = () => getJSON<GpuInfo>("/api/gpu");
export const getDeps        = () => getJSON<DepsStatus>("/api/deps");
export const getModels      = () => getJSON<ModelListing>("/api/models");
export const refreshModels  = () => postJSON<{ refreshed: boolean; counts: Record<string, number> }>("/api/models/refresh");

// ---------- Audio (ACE-Step bridge) ----------
export type AudioHealth = { ok: boolean; detail: any };
export const audioHealth = () => getJSON<AudioHealth>("/api/audio/health");
export const audioModels = () => getJSON<any>("/api/audio/models");

export interface AudioGeneratePayload {
  engine_id?: string;          // "ace_step" (vocals) | "stable_audio_3" (instrumental)
  prompt?: string;
  lyrics?: string;
  ace_model?: string | null;
  bpm?: number;
  key_scale?: string;
  duration?: number;
  steps?: number;              // Stable Audio 3 sampling steps
  temperature?: number;
  generate_cover?: boolean;
  cover_prompt?: string | null;
  thinking?: boolean;
  sample_mode?: boolean;
}

export const audioGenerate = (payload: AudioGeneratePayload) =>
  postJSON<{ job_id: string; status: string; kind: string }>("/api/audio/generate", payload);

// ---------- Audio engines (sidecar-managed, lazy-loaded) ----------
// The sidecar owns each engine's lifecycle. `available` = files on disk (NOT
// loaded); `running` = its process is alive. The picked engine loads on
// Generate and the other is killed first — so only one audio model is resident.
export type AudioEngine = {
  id: string;
  label: string;
  description: string;
  available: boolean;
  running: boolean;
  instrumental_only: boolean;
  vram_hint_gb: number;
};

export const getAudioEngines = () =>
  getJSON<{ engines: AudioEngine[]; current: string | null }>("/api/audio/engines");

export const stopAudioEngines = () =>
  postJSON<{ ok: boolean; current: string | null }>("/api/audio/engines/stop");

export const getAudioJob = (jobId: string) => getJSON<any>(`/api/audio/jobs/${jobId}`);
export const cancelAudioJob = (jobId: string) =>
  postJSON<any>(`/api/audio/jobs/${jobId}/cancel`, {});

export const audioFileUrl = (absPath: string) =>
  `/api/audio/file?path=${encodeURIComponent(absPath)}`;  // relative to the sidecar — works in Tauri webview

// ---------- Song Studio (port 8010 — workstation: library, playlists, etc.) ----------
//
// All these go through our /api/audio/* proxy. The webview never talks to
// 8010 directly. See python/api/audio.py for the proxy implementations.

export type SongStudioModel = {
  id: string;
  label: string;
  provider: string;
  params?: string;
  local?: boolean;
  recommended?: boolean;
  status?: string;
  notes?: string;
};

export type SongStudioHealth = {
  ok: boolean;
  base_url: string;
  config?: {
    assistantReady?: boolean;
    assistantProvider?: string;
    assistantStatus?: string;
    voiceCloneReady?: boolean;
    voiceCloneStatus?: string;
    coverArtReady?: boolean;
    coverArtStatus?: string;
    generationModels?: SongStudioModel[];
    defaultGenerationModel?: string;
    defaultWorkspace?: string;
    aceApiBaseUrl?: string;
  };
  error?: string;
};

export type Song = {
  id: string;
  title?: string;
  summary?: string;
  prompt?: string;
  lyrics?: string;
  workspaceId?: string;
  workspaceTitle?: string;
  workspaceKind?: string;
  bpm?: number | null;
  key?: string | null;
  language?: string | null;
  duration?: number | null;
  status?: string;
  createdAt?: string;
  coverPath?: string;
  audioPath?: string;
  audioPaths?: string[];
  styleTags?: string;
  folder?: string;
  // Song Studio uses camelCase OR snake_case in different fields — keep loose:
  [extra: string]: any;
};

export type Playlist = {
  id: string;
  title: string;
  songIds?: string[];
  songs?: Song[];
  [extra: string]: any;
};

export const songStudioHealth = () =>
  getJSON<SongStudioHealth>("/api/audio/song-studio/health");

export const getSongLibrary = () =>
  getJSON<{ songs: Song[] }>("/api/audio/library");

export const getPlaylists = () =>
  getJSON<{ playlists: Playlist[] }>("/api/audio/playlists");

export const createPlaylist = (title: string) =>
  postJSON<Playlist>("/api/audio/playlists", { title });

export const addSongsToPlaylist = (playlistId: string, songIds: string[]) =>
  postJSON<any>(`/api/audio/playlists/${playlistId}/songs`, { songIds });

export const createWorkspace = (title: string) =>
  postJSON<any>("/api/audio/workspaces", { title });

export const renameWorkspace = (workspaceId: string, title: string) =>
  patchJSON<any>(`/api/audio/workspaces/${workspaceId}`, { title });

export const deleteSong = async (songId: string) => {
  // Reuse the same JSON util but with DELETE method
  const url = `/api/audio/songs/${songId}`;
  const r = await fetch(url, { method: "DELETE" });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json().catch(() => ({ ok: true }));
};

export const bulkDeleteSongs = (songIds: string[]) =>
  postJSON<any>("/api/audio/songs/bulk-delete", { songIds });

export const songStreamUrl = (path: string) =>
  `/api/audio/stream?path=${encodeURIComponent(path)}`;

export const songDownloadUrl = (songId: string) =>
  `/api/audio/songs/${songId}/download`;

// ---------- Phase D: MP3 export (LAME VBR V0 + embedded cover) ----------

export type ExportMp3Result = {
  ok: boolean;
  song_id: string;
  mp3_path: string;
  mp3_url: string;
  source_wav: string;
  bitrate_avg_kbps: number;
  size_bytes: number;
  duration_seconds: number;
  cover_embedded: boolean;
  elapsed_s: number;
};

export type BulkExportMp3Result = { job_id: string; queued: number };

/** Sync single-song export. Returns when the MP3 is on disk and tagged. */
export const exportSongMp3 = (songId: string, opts: {
  overwrite?: boolean;
  title?: string;
  artist?: string;
  album?: string;
  genre?: string;
  comment?: string;
} = {}) => postJSON<ExportMp3Result>(`/api/audio/songs/${songId}/export-mp3`, opts);

/** Async bulk export. Returns a job_id immediately; subscribe via openJobWS. */
export const exportSongsBulk = (songIds: string[], overwrite = false) =>
  postJSON<BulkExportMp3Result>("/api/audio/songs/export-mp3", { song_ids: songIds, overwrite });

/** Browser-download URL for an already-exported MP3 (most recent match). */
export const exportedMp3DownloadUrl = (songId: string) =>
  `/api/audio/songs/${songId}/export-mp3/download`;

// ---------- Generation ----------

export type LoraEntry = { name: string; weight?: number; model_weight?: number };

export type GenerateParams = {
  arch: string;
  checkpoint?: string | null;
  diffusion_model?: string | null;
  vae?: string | null;
  text_encoders?: string[];
  clip_vision?: string | null;
  loras?: LoraEntry[];
  embeddings?: string[];
  prompt: string;
  negative?: string;
  ideogram_magic?: boolean;
  ideogram_magic_mode?: "local" | "api" | "raw";
  ideogram_speed_mode?: "max" | "high" | "fast";
  width: number;
  height: number;
  steps: number;
  cfg: number;
  sampler: string;
  scheduler?: string;
  clip_skip?: number | null;
  count: number;
  seed?: number | null;
  upscale_enabled?: boolean;
  upscale_mode?: "esrgan" | "usdu" | "iterative";
  upscale_model?: string | null;
  upscale_factor?: number;
  upscale_denoise?: number;
  upscale_tile_size?: number;
};

export type JobSnapshot = {
  id: string;
  kind: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  progress: { step: number; total_steps: number; image_index: number; total_images: number; message: string };
  result: any;
  error: string | null;
};

export const startGenerate = (p: GenerateParams) =>
  postJSON<{ job_id: string; status: string }>("/api/generate", p);
export const getJob = (id: string) => getJSON<JobSnapshot>(`/api/jobs/${id}`);
export const cancelJob = (id: string) => postJSON<{ status: string }>(`/api/jobs/${id}/cancel`);
export const ideogramMagicPrompt = (p: {
  prompt: string;
  negative?: string;
  width: number;
  height: number;
  mode?: "local" | "api" | "raw";
}) => postJSON<{ mode: string; prompt: string; pretty_prompt: string; aspect_ratio?: string }>("/api/ideogram4/magic-prompt", p);

// ---------- Prompt Builder (deterministic, shared across image archs) ----------
export type PromptBuilderOption = { value: string; label: string };
export type PromptBuilderOptions = Record<"style" | "lighting" | "camera" | "mood", PromptBuilderOption[]>;
export type PromptBuildResult = { arch: string; format: string; prompt: string; pretty: string; aspect_ratio: string };

export const promptBuilderOptions = () => getJSON<PromptBuilderOptions>("/api/prompt-builder/options");
export const promptBuilderBuild = (p: {
  arch: string; subject: string; texts?: string[];
  style?: string; lighting?: string; camera?: string; mood?: string;
  negative?: string; width?: number; height?: number;
}) => postJSON<PromptBuildResult>("/api/prompt-builder/build", p);

// ---------- Video (WAN i2v) ----------

export type VideoGenerateParams = {
  arch: string;                         // "wan"
  diffusion_model?: string | null;      // high-noise expert
  diffusion_model_2?: string | null;    // low-noise expert
  vae?: string | null;
  text_encoders?: string[];             // slot 1 = UMT5-XXL
  loras?: LoraEntry[];
  prompt: string;
  negative?: string;
  input_image?: string | null;          // data URL / raw base64 / path
  width: number;
  height: number;
  num_frames: number;
  fps: number;
  steps: number;
  cfg: number;
  seed?: number | null;
};

export const startGenerateVideo = (p: VideoGenerateParams) =>
  postJSON<{ job_id: string; status: string }>("/api/generate/video", p);

// ---------- Logs ----------

export type LogEntry = { id: number; ts: number; level: string; logger: string; message: string };

export const getLogs = (since_id?: number, limit = 500) =>
  getJSON<{ items: LogEntry[]; last_id: number }>(
    `/api/logs?limit=${limit}` + (since_id !== undefined ? `&since_id=${since_id}` : "")
  );
export const clearLogs = () => postJSON<{ cleared: boolean }>("/api/logs/clear");

// ---------- System ----------

export type ClearMemoryResult = {
  vram_free_mb_before: number | null;
  vram_free_mb_after: number | null;
  freed_mb: number | null;
  details: Record<string, any>;
};

export const clearMemory = () => postJSON<ClearMemoryResult>("/api/clear_memory");

// ---------- Settings ----------

export type Settings = {
  civitai: { api_token: string; api_token_set: boolean; nsfw_visible: boolean; default_sort: string };
  downloads: { category_paths: Record<string, string> };
  performance: {
    // 'auto': keep transformer fully on GPU when free VRAM allows, else stream
    //         (Forge-style per-Linear offload). Default.
    // 'on':   always try fully resident (may OOM on tight VRAM cards).
    // 'off':  always stream (safest, slightly slower if model would have fit).
    flux_fast_inference: "auto" | "on" | "off";
    // Headroom (GB) reserved for activations when auto-deciding fast vs stream.
    // 2.0 GB is safe on a 24 GB 3090. Lower it to let larger transformers stay
    // fully resident (faster) at higher OOM risk mid-step.
    flux_fast_inference_buffer_gb: number;
  };
  // Last Generate-tab setup. Restored on next launch if the named models still
  // exist in the current scan. Written by Generate.tsx, persisted via
  // config_store.lastGenerate.
  lastGenerate?: {
    archId?: string;
    checkpoint?: string;
    diffusionModel?: string;
    vae?: string;
    te?: string[];
    [extra: string]: any;
  } | null;
};

// Shape saved by the Generate tab so the user's last choices are restored on next launch.
export interface LastGenerate {
  archId?: string;
  checkpoint?: string;
  diffusionModel?: string;
  vae?: string;
  te?: string[];
  loras?: LoraEntry[];
  embeddings?: string[];
  prompt?: string;
  negative?: string;
  ideogramMagic?: boolean;
  ideogramMagicMode?: "local" | "api" | "raw";
  ideogramSpeedMode?: "max" | "high" | "fast";
  preset?: string;
  w?: number;
  h?: number;
  steps?: number;
  cfg?: number;
  sampler?: string;
  scheduler?: string;
  clipSkip?: number;
  count?: number;
  randomSeed?: boolean;
  seed?: number;
  upEnabled?: boolean;
  upMode?: "esrgan" | "usdu" | "iterative";
  upModel?: string;
  upFactor?: number;
  upDenoise?: number;
  upTile?: number;
}

export const getSettings   = () => getJSON<Settings>("/api/settings");
export const patchSettings = (patch: Partial<{
  civitai: Partial<Settings["civitai"]>;
  downloads: Partial<Settings["downloads"]>;
  performance: Partial<Settings["performance"]>;
  lastGenerate?: LastGenerate | null;
}>) => patchJSON<Settings>("/api/settings", patch);

export const saveLastGenerate = (data: LastGenerate) =>
  patchSettings({ lastGenerate: data });

// ---------- Civitai ----------

export type CivitaiCreator = { username?: string; image?: string | null };
export type CivitaiFile = {
  id: number; name: string; sizeKB?: number; primary?: boolean;
  hashes?: { SHA256?: string }; downloadUrl?: string; type?: string;
  metadata?: { fp?: string; size?: string; format?: string };
};
export type CivitaiImage = { url: string; nsfw?: string | boolean | null; width?: number; height?: number };
export type CivitaiVersion = {
  id: number; modelId?: number; name: string; baseModel?: string;
  trainedWords?: string[]; files?: CivitaiFile[]; images?: CivitaiImage[];
  description?: string | null;
};
export type CivitaiModel = {
  id: number; name: string; description?: string; type: string; nsfw?: boolean;
  tags?: string[]; creator?: CivitaiCreator; stats?: { downloadCount?: number; rating?: number; thumbsUpCount?: number };
  modelVersions?: CivitaiVersion[];
};
export type CivitaiSearchResult = {
  items: CivitaiModel[];
  metadata?: {
    totalItems?: number;
    currentPage?: number;
    pageSize?: number;
    totalPages?: number;
    nextPage?: string;
    nextCursor?: string;
  };
};

export type CivitaiSearchParams = {
  query?: string; types?: string; baseModels?: string;
  nsfw?: boolean; limit?: number; page?: number; sort?: string; period?: string; cursor?: string;
};

export const civitaiSearch = (p: CivitaiSearchParams) => {
  const q = new URLSearchParams();
  if (p.query)      q.set("query", p.query);
  if (p.types)      q.set("types", p.types);
  if (p.baseModels) q.set("baseModels", p.baseModels);
  if (p.nsfw === false) q.set("nsfw", "false");
  if (p.limit)      q.set("limit", String(p.limit));
  if (p.page)       q.set("page", String(p.page));
  if (p.sort)       q.set("sort", p.sort);
  if (p.period)     q.set("period", p.period);
  if (p.cursor)     q.set("cursor", p.cursor);
  return getJSON<CivitaiSearchResult>(`/api/civitai/search?${q.toString()}`);
};

export const civitaiGetModel   = (id: number) => getJSON<CivitaiModel>(`/api/civitai/model/${id}`);
export const civitaiGetVersion = (id: number) => getJSON<CivitaiVersion>(`/api/civitai/version/${id}`);
export const civitaiDownload   = (modelVersionId: number, fileId?: number) =>
  postJSON<{ job_id: string; dest: string; category: string }>("/api/civitai/download", { modelVersionId, fileId });

// ---------- Outputs ----------

export type OutputDeleteResult = { deleted: string[]; errors: { path: string; error: string }[] };
export type LatestImageOutput = {
  rel_path: string;
  filename: string;
  size_bytes: number;
  mtime: number;
  model_label: string;
  arch?: string | null;
  seed?: number | null;
  width?: number | null;
  height?: number | null;
};

export async function outputFileUrl(relPath: string): Promise<string> {
  return (await base()) + "/api/outputs/file/" + relPath.split("/").map(encodeURIComponent).join("/");
}

export const deleteOutputs = (paths: string[]) =>
  postJSON<OutputDeleteResult>("/api/outputs/delete", { paths });

export const getOutputSettings = (relPath: string) =>
  getJSON<GenerateParams>("/api/outputs/settings/" + relPath.split("/").map(encodeURIComponent).join("/"));

export const getLatestImages = (limit = 9) =>
  getJSON<{ root: string; items: LatestImageOutput[] }>(`/api/outputs/latest-images?limit=${limit}`);

export async function openJobWS(jobId: string): Promise<WebSocket> {
  let wsBase: string;
  try {
    wsBase = await invoke<string>("sidecar_ws_url");
  } catch {
    wsBase = "ws://127.0.0.1:7780";
  }
  return new WebSocket(`${wsBase}/ws/jobs/${jobId}`);
}
