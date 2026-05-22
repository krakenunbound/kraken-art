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
  driver: string | null;
  cuda_runtime: string | null;
  cuda_available: boolean;
  torch_version: string | null;
  errors: string[];
};

export type DepPackage = { name: string; required: string; installed: string | null; ok: boolean; remedy: string | null };
export type DepsStatus = { all_ok: boolean; missing: number; packages: DepPackage[] };

export type ModelEntry = { name: string; filename: string; subdir: string; abs_path: string; size_bytes: number; ext: string };
export type ModelListing = {
  root: string;
  exists: boolean;
  categories: Record<string, ModelEntry[]>;
  counts: Record<string, number>;
};

// ---------- Endpoints ----------

export const health         = () => getJSON<Health>("/health");
export const getGpu         = () => getJSON<GpuInfo>("/api/gpu");
export const getDeps        = () => getJSON<DepsStatus>("/api/deps");
export const getModels      = () => getJSON<ModelListing>("/api/models");
export const refreshModels  = () => postJSON<{ refreshed: boolean; counts: Record<string, number> }>("/api/models/refresh");

// ---------- Generation ----------

export type LoraEntry = { name: string; weight: number };

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
  width: number;
  height: number;
  steps: number;
  cfg: number;
  sampler: string;
  scheduler?: string;
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
};

export const getSettings   = () => getJSON<Settings>("/api/settings");
export const patchSettings = (patch: Partial<{
  civitai: Partial<Settings["civitai"]>;
  downloads: Partial<Settings["downloads"]>;
  performance: Partial<Settings["performance"]>;
}>) => patchJSON<Settings>("/api/settings", patch);

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
export type CivitaiSearchResult = { items: CivitaiModel[]; metadata?: { totalItems?: number; currentPage?: number; pageSize?: number; totalPages?: number; nextPage?: string } };

export type CivitaiSearchParams = {
  query?: string; types?: string; baseModels?: string;
  nsfw?: boolean; limit?: number; page?: number; sort?: string;
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
  return getJSON<CivitaiSearchResult>(`/api/civitai/search?${q.toString()}`);
};

export const civitaiGetModel   = (id: number) => getJSON<CivitaiModel>(`/api/civitai/model/${id}`);
export const civitaiGetVersion = (id: number) => getJSON<CivitaiVersion>(`/api/civitai/version/${id}`);
export const civitaiDownload   = (modelVersionId: number, fileId?: number) =>
  postJSON<{ job_id: string; dest: string; category: string }>("/api/civitai/download", { modelVersionId, fileId });

// ---------- Outputs ----------

export type OutputDeleteResult = { deleted: string[]; errors: { path: string; error: string }[] };

export async function outputFileUrl(relPath: string): Promise<string> {
  return (await base()) + "/api/outputs/file/" + relPath.split("/").map(encodeURIComponent).join("/");
}

export const deleteOutputs = (paths: string[]) =>
  postJSON<OutputDeleteResult>("/api/outputs/delete", { paths });

export async function openJobWS(jobId: string): Promise<WebSocket> {
  let wsBase: string;
  try {
    wsBase = await invoke<string>("sidecar_ws_url");
  } catch {
    wsBase = "ws://127.0.0.1:7780";
  }
  return new WebSocket(`${wsBase}/ws/jobs/${jobId}`);
}
