import { useEffect, useMemo, useRef, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { openPath } from "@tauri-apps/plugin-opener";
import {
  cancelJob,
  deleteOutputs,
  getLatestVideos,
  getVideoUpscaleStatus,
  importVideoFile,
  importVideoPath,
  openJobWS,
  outputFileUrl,
  startVideoUpscale,
  type LatestVideoOutput,
  type VideoUpscaleParams,
  type VideoUpscaleStatus,
} from "./api/sidecar";

type Source = LatestVideoOutput & { url: string };
type Result = { rel_path: string; filename: string; path?: string; seed?: number; preview_b64?: string };
const VIDEO_ACCEPT = ".mp4,.webm,.mov,.mkv,.avi,.m4v";
const VIDEO_EXT_RE = /\.(mp4|webm|mov|mkv|avi|m4v)$/i;
const UPSCALE_RESULT_RE = /-vup\.(mp4|webm|mov|mkv|avi|m4v)$/i;
type NumericControlProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  format?: (n: number) => string;
  onChange: (n: number) => void;
};

const SIZE_PRESETS = [
  {
    label: "HD / 1080p",
    mode: "resolution",
    resolution: 1080,
    max: 1920,
    factor: 2,
    note: "Best first test. Good quality and predictable runtime on this GPU.",
  },
  {
    label: "2K / 1440p",
    mode: "resolution",
    resolution: 1440,
    max: 2560,
    factor: 2,
    note: "Best balance target for finished clips. Slower than HD, much lighter than 4K.",
  },
  {
    label: "4K / 2160p",
    mode: "resolution",
    resolution: 2160,
    max: 3840,
    factor: 2,
    note: "Highest output target. Expect long SeedVR2 runs and possible memory limits.",
  },
  {
    label: "2x source",
    mode: "factor",
    resolution: 1080,
    max: 1920,
    factor: 2,
    note: "Doubles source dimensions. Useful when you want exact scale instead of a named format.",
  },
  {
    label: "3x source",
    mode: "factor",
    resolution: 1080,
    max: 1920,
    factor: 3,
    note: "Large upscale. Better for ESRGAN preservation than SeedVR2 speed.",
  },
  {
    label: "4x source",
    mode: "factor",
    resolution: 1080,
    max: 1920,
    factor: 4,
    note: "Maximum simple scale. Use for short tests before committing a full render.",
  },
  {
    label: "Custom resolution",
    mode: "custom_resolution",
    resolution: 1080,
    max: 1920,
    factor: 2,
    note: "Set short side and max edge manually.",
  },
  {
    label: "Custom factor",
    mode: "custom_factor",
    resolution: 1080,
    max: 1920,
    factor: 2,
    note: "Set an exact source multiplier manually.",
  },
];

function SliderNumber({ label, value, min, max, step, disabled, format, onChange }: NumericControlProps) {
  const display = format ? format(value) : String(value);
  return (
    <label className="field numeric-control">
      <span>{label} {display}</span>
      <div className="slider-row">
        <input
          type="range"
          value={value}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      </div>
    </label>
  );
}

function seedBatch(n: number): number {
  return Math.max(1, Math.min(81, Math.round((n - 1) / 4) * 4 + 1));
}

function seedModelNote(name: string): string {
  const n = name.toLowerCase();
  if (/7b.*sharp|sharp.*7b/.test(n)) {
    return "Highest detail and strongest polish. Best for AI-generated clips, but can repaint faces, logos, fabric, and small background elements.";
  }
  if (/3b.*fp8|fp8.*3b/.test(n)) {
    return "Recommended default. Faster than 7B, strong detail, good temporal stability, still more generative than a classic upscaler.";
  }
  if (/q4|gguf/.test(n)) {
    return "Quantized model for fitting larger SeedVR2 variants into VRAM. Good detail, slower and more stylized than preservation upscalers.";
  }
  return "SeedVR2 diffusion upscale. Best when you want reconstruction and polish, not strict pixel-level preservation.";
}

function esrganModelNote(name: string): string {
  const n = name.toLowerCase();
  if (/siax/.test(n)) {
    return "Best sharp preservation pick from the tested local models. Keeps composition, adds crisp detail, slower than RealESRGAN_x2 and can make edges crunchy.";
  }
  if (/ultrasharp/.test(n)) {
    return "Punchy detail and crisp edges. Good for sci-fi panels and suits; can add halos or over-sharpen skin.";
  }
  if (/remacri/.test(n)) {
    return "Natural-looking texture with moderate sharpening. Good middle ground when UltraSharp is too harsh.";
  }
  if (/realesrgan.*x2|realesrgan_x2/.test(n)) {
    return "Default from the test clip. Fastest clean preservation choice for 1080p; avoids hallucination, but is softer than Siax, UltraSharp, or SeedVR2.";
  }
  if (/realesrgan/.test(n)) {
    return "General-purpose restoration. Usually faithful, sometimes smooths fine detail.";
  }
  return "Frame upscaler loaded through Spandrel. Fast and source-preserving compared with SeedVR2.";
}

const COLOR_MODES: VideoUpscaleParams["color_correction"][] = [
  "lab",
  "wavelet",
  "wavelet_adaptive",
  "hsv",
  "adain",
  "none",
];

export default function VideoUpscale({ active }: { active: boolean }) {
  const [statusInfo, setStatusInfo] = useState<VideoUpscaleStatus | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [selected, setSelected] = useState<Source | null>(null);
  const [checkedSources, setCheckedSources] = useState<Set<string>>(new Set());
  const [resultVideos, setResultVideos] = useState<Source[]>([]);
  const [selectedResult, setSelectedResult] = useState<Source | null>(null);
  const [checkedResults, setCheckedResults] = useState<Set<string>>(new Set());
  const [sourcePath, setSourcePath] = useState("");
  const [sourceNotice, setSourceNotice] = useState("");
  const [importing, setImporting] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const [engine, setEngine] = useState<VideoUpscaleParams["engine"]>("esrgan");
  const [presetIdx, setPresetIdx] = useState(0);
  const [scaleMode, setScaleMode] = useState<VideoUpscaleParams["scale_mode"]>("resolution");
  const [upscaleFactor, setUpscaleFactor] = useState(2);
  const [resolution, setResolution] = useState(1080);
  const [maxResolution, setMaxResolution] = useState(1920);
  const [ditModel, setDitModel] = useState("");
  const [upscaleModel, setUpscaleModel] = useState("");
  const [batchSize, setBatchSize] = useState(33);
  const [temporalOverlap, setTemporalOverlap] = useState(3);
  const [uniformBatch, setUniformBatch] = useState(true);
  const [chunkSize, setChunkSize] = useState(0);
  const [colorCorrection, setColorCorrection] = useState<VideoUpscaleParams["color_correction"]>("lab");
  const [aiDetailStrength, setAiDetailStrength] = useState(0.15);
  const [inputNoiseScale, setInputNoiseScale] = useState(0);
  const [latentNoiseScale, setLatentNoiseScale] = useState(0);
  const [tenBit, setTenBit] = useState(false);
  const [vaeEncodeTiled, setVaeEncodeTiled] = useState(true);
  const [vaeEncodeTileSize, setVaeEncodeTileSize] = useState(1024);
  const [vaeEncodeTileOverlap, setVaeEncodeTileOverlap] = useState(128);
  const [vaeDecodeTiled, setVaeDecodeTiled] = useState(true);
  const [vaeDecodeTileSize, setVaeDecodeTileSize] = useState(1024);
  const [vaeDecodeTileOverlap, setVaeDecodeTileOverlap] = useState(128);
  const [blocksToSwap, setBlocksToSwap] = useState(0);
  const [swapIo, setSwapIo] = useState(true);
  const [attentionMode, setAttentionMode] = useState<VideoUpscaleParams["attention_mode"]>("sdpa");
  const [compileDit, setCompileDit] = useState(false);
  const [compileVae, setCompileVae] = useState(false);
  const [compileMode, setCompileMode] = useState<VideoUpscaleParams["compile_mode"]>("default");
  const [esrganTileSize, setEsrganTileSize] = useState(0);
  const [esrganTileOverlap, setEsrganTileOverlap] = useState(64);
  const [targetFps, setTargetFps] = useState(60);
  const [interpolation, setInterpolation] = useState<VideoUpscaleParams["interpolation"]>("rife_ncnn");
  const [keepAudio, setKeepAudio] = useState(true);
  const [seed, setSeed] = useState(42);

  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState("idle");
  const [step, setStep] = useState({ step: 0, total: 0 });
  const [message, setMessage] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [resultUrl, setResultUrl] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  async function refreshSources(select?: { relPath: string; lane: "source" | "result" }) {
    const data = await getLatestVideos(80);
    const items = await Promise.all(
      data.items.map(async (it) => ({ ...it, url: await outputFileUrl(it.rel_path) })),
    );
    const nextResults = items.filter((it) => UPSCALE_RESULT_RE.test(it.filename));
    const nextSources = items.filter((it) => !UPSCALE_RESULT_RE.test(it.filename));
    setSources(nextSources);
    setResultVideos(nextResults);
    const liveSources = new Set(nextSources.map((it) => it.rel_path));
    const liveResults = new Set(nextResults.map((it) => it.rel_path));
    setCheckedSources((cur) => new Set([...cur].filter((rel) => liveSources.has(rel))));
    setCheckedResults((cur) => new Set([...cur].filter((rel) => liveResults.has(rel))));
    if (select?.lane === "source") {
      const hit = nextSources.find((it) => it.rel_path === select.relPath) ?? nextSources[0] ?? null;
      setSelected(hit);
      setSourcePath("");
    }
    if (select?.lane === "result") {
      const hit = nextResults.find((it) => it.rel_path === select.relPath) ?? nextResults[0] ?? null;
      setSelectedResult(hit);
    }
    setSelected((cur) => {
      if (select?.lane === "source") return cur;
      if (cur) {
        const hit = nextSources.find((it) => it.rel_path === cur.rel_path);
        if (hit) return hit;
      }
      return nextSources[0] ?? null;
    });
    setSelectedResult((cur) => {
      if (select?.lane === "result") return cur;
      if (cur) {
        const hit = nextResults.find((it) => it.rel_path === cur.rel_path);
        if (hit) return hit;
      }
      return nextResults[0] ?? null;
    });
  }

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    getVideoUpscaleStatus()
      .then((s) => {
        if (cancelled) return;
        setStatusInfo(s);
        if (!ditModel && s.seedvr2_models.length) {
          const model =
            s.seedvr2_models.find((m) => /3b.*fp8|fp8.*3b/i.test(m.name)) ??
            s.seedvr2_models[0];
          setDitModel(model.name);
        }
        if (!upscaleModel && s.upscale_models?.length) {
          const model =
            s.upscale_models.find((m) => /realesrgan.*x2|realesrgan_x2/i.test(m.name)) ??
            s.upscale_models.find((m) => /nmkd.*siax|siax/i.test(m.name)) ??
            s.upscale_models.find((m) => /ultrasharp/i.test(m.name)) ??
            s.upscale_models.find((m) => /remacri/i.test(m.name)) ??
            s.upscale_models.find((m) => /realesrgan/i.test(m.name)) ??
            s.upscale_models[0];
          setUpscaleModel(model.name);
        }
      })
      .catch(() => {});
    refreshSources()
      .then(() => {
        if (cancelled) return;
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [active]);

  useEffect(() => {
    if (!result?.rel_path) return;
    let cancelled = false;
    outputFileUrl(result.rel_path).then((url) => {
      if (!cancelled) setResultUrl(url);
    });
    return () => {
      cancelled = true;
    };
  }, [result]);

  useEffect(() => {
    if (!active) return;
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");
    const onDragEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      setDragActive(true);
    };
    const onDragOver = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
      setDragActive(true);
    };
    const onDragLeave = (e: DragEvent) => {
      if (e.clientX <= 0 || e.clientY <= 0 || e.clientX >= window.innerWidth || e.clientY >= window.innerHeight) {
        setDragActive(false);
      }
    };
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      setDragActive(false);
      if (e.dataTransfer?.files?.length) void importFiles(e.dataTransfer.files);
    };
    window.addEventListener("dragenter", onDragEnter);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("drop", onDrop);

    let disposed = false;
    let unlisten: (() => void) | null = null;
    getCurrentWindow().onDragDropEvent((event) => {
      const payload = event.payload;
      if (payload.type === "enter" || payload.type === "over") {
        setDragActive(true);
      } else if (payload.type === "leave") {
        setDragActive(false);
      } else if (payload.type === "drop") {
        setDragActive(false);
        void importPathList(payload.paths);
      }
    }).then((fn) => {
      if (disposed) fn();
      else unlisten = fn;
    }).catch(() => {});

    return () => {
      disposed = true;
      unlisten?.();
      window.removeEventListener("dragenter", onDragEnter);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, [active]);

  const busy = jobStatus === "queued" || jobStatus === "running" || jobStatus === "submitting";
  const pct = step.total > 0 ? Math.floor((step.step / step.total) * 100) : 0;
  const cliReady = !!statusInfo?.seedvr2_cli;
  const ffmpegReady = !!statusInfo?.ffmpeg;
  const rifeReady = !!statusInfo?.rife_ncnn;
  const hasSource = !!sourcePath.trim() || !!selected;
  const interpReady = interpolation !== "rife_ncnn" || rifeReady;
  const esrganReady = !!upscaleModel;
  const seedNeedsDetail = aiDetailStrength > 0.0001;
  const seedNeedsBase = aiDetailStrength < 0.9999;
  const seedReady = (!seedNeedsDetail || (cliReady && !!ditModel)) && (!seedNeedsBase || esrganReady);
  const ready = ffmpegReady && interpReady && hasSource && (engine === "seedvr2" ? seedReady : esrganReady) && !busy;
  const sizePreset = SIZE_PRESETS[presetIdx] ?? SIZE_PRESETS[0];
  const customRes = sizePreset.mode === "custom_resolution";
  const customFactor = sizePreset.mode === "custom_factor";
  const sizeSummary =
    scaleMode === "factor"
      ? `${upscaleFactor.toFixed(upscaleFactor % 1 ? 1 : 0)}x source`
      : `${resolution}px short side / max ${maxResolution}px`;
  const checkedCount = checkedSources.size;
  const checkedResultCount = checkedResults.size;
  const activeResult = resultUrl
    ? { url: resultUrl, filename: result?.filename ?? "Upscaled result", path: result?.path }
    : selectedResult;

  const sourceLabel = useMemo(() => {
    if (sourcePath.trim()) return sourcePath.trim();
    if (selected) return selected.filename;
    return "No source selected";
  }, [sourcePath, selected]);

  function applyPreset(idx: number) {
    setPresetIdx(idx);
    const p = SIZE_PRESETS[idx];
    if (p.mode === "factor" || p.mode === "custom_factor") {
      setScaleMode("factor");
      setUpscaleFactor(p.factor);
    } else {
      setScaleMode("resolution");
      setResolution(p.resolution);
      setMaxResolution(p.max);
    }
  }

  function setSourceChecked(relPath: string, checked: boolean) {
    setCheckedSources((cur) => {
      const next = new Set(cur);
      if (checked) next.add(relPath);
      else next.delete(relPath);
      return next;
    });
  }

  function checkAllSources() {
    setCheckedSources(new Set(sources.map((s) => s.rel_path)));
  }

  function setResultChecked(relPath: string, checked: boolean) {
    setCheckedResults((cur) => {
      const next = new Set(cur);
      if (checked) next.add(relPath);
      else next.delete(relPath);
      return next;
    });
  }

  function checkAllResults() {
    setCheckedResults(new Set(resultVideos.map((s) => s.rel_path)));
  }

  async function importFiles(filesLike: FileList | File[]) {
    const files = Array.from(filesLike).filter((file) => file.type.startsWith("video/") || VIDEO_EXT_RE.test(file.name));
    if (!files.length) {
      setErrorMsg("Choose a video file: mp4, webm, mov, mkv, avi, or m4v.");
      setDragActive(false);
      return;
    }
    setErrorMsg("");
    setSourceNotice(`Importing ${files.length} video${files.length === 1 ? "" : "s"}...`);
    setImporting(true);
    let lastRel = "";
    try {
      for (const file of files) {
        const r = await importVideoFile(file);
        lastRel = r.item.rel_path;
      }
      await refreshSources({ relPath: lastRel, lane: "source" });
      setResult(null);
      setResultUrl("");
      setSourceNotice(`Imported ${files.length} video${files.length === 1 ? "" : "s"}.`);
    } catch (e: any) {
      setErrorMsg(`video import failed: ${e?.message ?? e}`);
      setSourceNotice("");
    } finally {
      setImporting(false);
      setDragActive(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function importPathList(pathsLike: string[]) {
    const paths = pathsLike.map((p) => p.trim().replace(/^["']|["']$/g, "")).filter((p) => VIDEO_EXT_RE.test(p));
    if (!paths.length) {
      setErrorMsg("Drop or paste a video path: mp4, webm, mov, mkv, avi, or m4v.");
      return;
    }
    setErrorMsg("");
    setSourceNotice(`Importing ${paths.length} video${paths.length === 1 ? "" : "s"}...`);
    setImporting(true);
    let lastRel = "";
    try {
      for (const path of paths) {
        const r = await importVideoPath(path);
        lastRel = r.item.rel_path;
      }
      await refreshSources({ relPath: lastRel, lane: "source" });
      setResult(null);
      setResultUrl("");
      setSourceNotice(`Imported ${paths.length} video${paths.length === 1 ? "" : "s"}.`);
    } catch (e: any) {
      setErrorMsg(`path import failed: ${e?.message ?? e}`);
      setSourceNotice("");
    } finally {
      setImporting(false);
    }
  }

  async function importPath() {
    const path = sourcePath.trim();
    if (!path) {
      setErrorMsg("Paste a video path first.");
      return;
    }
    await importPathList([path]);
  }

  async function deleteOneVideo(source: Source) {
    setErrorMsg("");
    setSourceNotice("Deleting video...");
    setImporting(true);
    try {
      const r = await deleteOutputs([source.rel_path]);
      if (r.errors.length) {
        throw new Error(r.errors.map((e) => `${e.path}: ${e.error}`).join("; "));
      }
      if (selected?.rel_path === source.rel_path) {
        setSelected(null);
      }
      if (selectedResult?.rel_path === source.rel_path || result?.rel_path === source.rel_path) {
        setSelectedResult(null);
        setResult(null);
        setResultUrl("");
      }
      setCheckedSources((cur) => {
        const next = new Set(cur);
        next.delete(source.rel_path);
        return next;
      });
      setCheckedResults((cur) => {
        const next = new Set(cur);
        next.delete(source.rel_path);
        return next;
      });
      await refreshSources();
      setSourceNotice("Deleted video.");
    } catch (e: any) {
      setErrorMsg(`delete failed: ${e?.message ?? e}`);
      setSourceNotice("");
    } finally {
      setImporting(false);
    }
  }

  async function deletePaths(paths: string[]) {
    if (!paths.length) return;
    setErrorMsg("");
    setSourceNotice(`Deleting ${paths.length} video${paths.length === 1 ? "" : "s"}...`);
    setImporting(true);
    try {
      const r = await deleteOutputs(paths);
      if (r.errors.length) {
        throw new Error(r.errors.map((e) => `${e.path}: ${e.error}`).join("; "));
      }
      const deleted = new Set(r.deleted);
      if (selected && deleted.has(selected.rel_path)) {
        setSelected(null);
      }
      if ((selectedResult && deleted.has(selectedResult.rel_path)) || (result && deleted.has(result.rel_path))) {
        setSelectedResult(null);
        setResult(null);
        setResultUrl("");
      }
      setCheckedSources((cur) => new Set([...cur].filter((rel) => !deleted.has(rel))));
      setCheckedResults((cur) => new Set([...cur].filter((rel) => !deleted.has(rel))));
      await refreshSources();
      setSourceNotice(`Deleted ${r.deleted.length} videos.`);
    } catch (e: any) {
      setErrorMsg(`delete failed: ${e?.message ?? e}`);
      setSourceNotice("");
    } finally {
      setImporting(false);
    }
  }

  async function deleteCheckedSources() {
    await deletePaths(sources.filter((s) => checkedSources.has(s.rel_path)).map((s) => s.rel_path));
  }

  async function deleteAllSources() {
    await deletePaths(sources.map((s) => s.rel_path));
  }

  async function deleteCheckedResults() {
    await deletePaths(resultVideos.filter((s) => checkedResults.has(s.rel_path)).map((s) => s.rel_path));
  }

  async function deleteAllResults() {
    await deletePaths(resultVideos.map((s) => s.rel_path));
  }

  function clearSource() {
    setSelected(null);
    setCheckedSources(new Set());
    setSelectedResult(null);
    setCheckedResults(new Set());
    setSourcePath("");
    setResult(null);
    setResultUrl("");
    setSourceNotice("");
  }

  async function run() {
    setErrorMsg("");
    setResult(null);
    setResultUrl("");
    setMessage("");
    setStep({ step: 0, total: 100 });
    setJobStatus("submitting");

    const params: VideoUpscaleParams = {
      source_rel_path: sourcePath.trim() ? null : selected?.rel_path ?? null,
      source_path: sourcePath.trim() || null,
      engine,
      dit_model: engine === "seedvr2" ? (ditModel || null) : null,
      upscale_model: upscaleModel || null,
      scale_mode: scaleMode,
      upscale_factor: upscaleFactor,
      resolution,
      max_resolution: maxResolution,
      batch_size: batchSize,
      temporal_overlap: temporalOverlap,
      uniform_batch_size: uniformBatch,
      chunk_size: chunkSize,
      color_correction: colorCorrection,
      ai_detail_strength: aiDetailStrength,
      input_noise_scale: inputNoiseScale,
      latent_noise_scale: latentNoiseScale,
      ten_bit: tenBit,
      vae_encode_tiled: vaeEncodeTiled,
      vae_encode_tile_size: vaeEncodeTileSize,
      vae_encode_tile_overlap: vaeEncodeTileOverlap,
      vae_decode_tiled: vaeDecodeTiled,
      vae_decode_tile_size: vaeDecodeTileSize,
      vae_decode_tile_overlap: vaeDecodeTileOverlap,
      blocks_to_swap: blocksToSwap,
      swap_io_components: swapIo,
      attention_mode: attentionMode,
      compile_dit: compileDit,
      compile_vae: compileVae,
      compile_mode: compileMode,
      seed,
      esrgan_tile_size: esrganTileSize,
      esrgan_tile_overlap: esrganTileOverlap,
      keep_audio: keepAudio,
      target_fps: interpolation === "none" ? 0 : targetFps,
      interpolation,
    };

    try {
      const r = await startVideoUpscale(params);
      setJobId(r.job_id);
      setJobStatus(r.status);
      const ws = await openJobWS(r.job_id);
      wsRef.current = ws;
      ws.onmessage = (e) => {
        const evt = JSON.parse(e.data);
        if (evt.type === "snapshot" || evt.type === "status") {
          setJobStatus(evt.status);
          if (evt.error) setErrorMsg(evt.error);
          const out = evt.result?.output;
          if (evt.status === "succeeded" && out?.rel_path) {
            setResult({ rel_path: out.rel_path, filename: out.filename, path: out.path });
            void refreshSources({ relPath: out.rel_path, lane: "result" });
          }
        }
        if (evt.type === "progress") {
          setStep({ step: evt.step, total: evt.total_steps });
          if (evt.message) setMessage(evt.message);
        }
        if (evt.type === "video") {
          setResult({
            rel_path: evt.rel_path ?? "",
            filename: evt.filename,
            path: evt.path,
            seed: evt.seed,
            preview_b64: evt.preview_b64,
          });
          if (evt.rel_path) void refreshSources({ relPath: evt.rel_path, lane: "result" });
        }
      };
      const resetIfStuck = (msg: string) => {
        wsRef.current = null;
        setJobStatus((cur) => {
          if (cur === "queued" || cur === "running" || cur === "submitting") {
            setErrorMsg(msg);
            return "idle";
          }
          return cur;
        });
      };
      ws.onclose = () => resetIfStuck("Connection to sidecar lost - check Logs.");
      ws.onerror = () => resetIfStuck("WebSocket error - sidecar may have died. Check Logs.");
    } catch (e: any) {
      setJobStatus("idle");
      setErrorMsg(String(e?.message ?? e));
    }
  }

  async function cancel() {
    if (!jobId) return;
    try {
      await cancelJob(jobId);
    } catch {
      /* ignore */
    }
  }

  async function openResultVideo(openWith?: string) {
    const path = activeResult?.path;
    if (!path) {
      setErrorMsg("No local result file path is available yet.");
      return;
    }
    try {
      await openPath(path, openWith);
    } catch (e: any) {
      setErrorMsg(`open result failed: ${e?.message ?? e}`);
    }
  }

  return (
    <>
      <main className={"pane center gen-center" + (active ? "" : " tab-hidden")}>
        <div className="source-head">
          <div className="section-title">Recent / imported videos</div>
          <div className="source-actions">
            <input
              ref={fileInputRef}
              type="file"
              accept={VIDEO_ACCEPT}
              multiple
              hidden
              onChange={(e) => void importFiles(e.target.files ?? [])}
            />
            <button type="button" onClick={() => fileInputRef.current?.click()} disabled={importing || busy}>
              Add videos
            </button>
            <button type="button" onClick={() => void refreshSources()} disabled={importing}>
              Refresh
            </button>
            <button type="button" onClick={checkAllSources} disabled={importing || sources.length === 0 || checkedCount === sources.length}>
              Check all
            </button>
            <button
              type="button"
              onClick={clearSource}
              disabled={importing || (!selected && !selectedResult && !sourcePath.trim() && !result && checkedCount === 0 && checkedResultCount === 0)}
            >
              Clear selection
            </button>
            <button type="button" className="danger" onClick={() => void deleteCheckedSources()}
              disabled={importing || busy || checkedCount === 0}>
              Delete checked{checkedCount ? ` (${checkedCount})` : ""}
            </button>
            <button type="button" className="danger" onClick={() => void deleteAllSources()}
              disabled={importing || busy || sources.length === 0}>
              Delete all
            </button>
          </div>
        </div>
        <div
          className={"card video-drop" + (dragActive ? " drag" : "")}
          onDragEnter={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
            setDragActive(true);
          }}
          onDragLeave={(e) => {
            if (e.currentTarget === e.target) setDragActive(false);
          }}
          onDrop={(e) => {
            e.preventDefault();
            void importFiles(e.dataTransfer.files);
          }}
        >
          {sources.length === 0 ? (
            <div className="placeholder small">Drop videos here or choose Add videos.</div>
          ) : (
            <div className="video-source-grid">
              {sources.map((s) => (
                <div key={s.rel_path} className={"video-source-wrap" + (checkedSources.has(s.rel_path) ? " checked" : "")}>
                  <button
                    className={"video-source" + (selected?.rel_path === s.rel_path && !sourcePath.trim() ? " sel" : "")}
                    onClick={() => {
                      setSelected(s);
                      setSourcePath("");
                    }}
                    title={s.filename}
                  >
                    <video src={s.url} muted preload="metadata" />
                    <span>{s.filename}</span>
                  </button>
                  <label className="video-source-check" title={`Mark ${s.filename} for deletion`}>
                    <input
                      type="checkbox"
                      checked={checkedSources.has(s.rel_path)}
                      onChange={(e) => setSourceChecked(s.rel_path, e.target.checked)}
                    />
                  </label>
                  <button
                    type="button"
                    className="video-source-remove"
                    title={`Delete ${s.filename}`}
                    onClick={() => void deleteOneVideo(s)}
                    disabled={importing || busy}
                  >
                    x
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
        {sourceNotice && <div className="muted small source-notice">{sourceNotice}</div>}

        <div className="path-row">
          <label className="field">
            <span className="lbl">External video path</span>
            <input
              value={sourcePath}
              onChange={(e) => setSourcePath(e.target.value)}
              placeholder="F:\\Downloads\\grok-imagine-clip.mp4"
            />
          </label>
          <button type="button" onClick={() => void importPath()} disabled={importing || !sourcePath.trim()}>
            Import path
          </button>
        </div>

        <div className="section-title">Compare preview</div>
        <div className="card video-compare">
          <div className="compare-pane">
            <div className="compare-title">Source</div>
            {selected && !sourcePath.trim() ? (
              <video src={selected.url} controls loop />
            ) : (
              <div className="placeholder small">{sourceLabel}</div>
            )}
            <div className="muted small compare-caption">{selected && !sourcePath.trim() ? selected.filename : sourceLabel}</div>
          </div>
          <div className="compare-pane">
            <div className="compare-title">Result</div>
            {activeResult?.url ? (
              <video src={activeResult.url} controls loop />
            ) : (
              <div className="placeholder small">No result selected</div>
            )}
            <div className="result-actions">
              <div className="muted small compare-caption">
                {activeResult?.filename ?? `${engine === "seedvr2" ? "SeedVR2" : "ESRGAN"} ${sizeSummary}`}
              </div>
              {activeResult?.path && (
                <div className="mini-actions">
                  <button type="button" onClick={() => void openResultVideo()}>Open result</button>
                  <button type="button" onClick={() => void openResultVideo("vlc")}>VLC</button>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="source-head result-head">
          <div className="section-title">Upscale results</div>
          <div className="source-actions">
            <button type="button" onClick={() => void refreshSources()} disabled={importing}>
              Refresh
            </button>
            <button type="button" onClick={checkAllResults}
              disabled={importing || resultVideos.length === 0 || checkedResultCount === resultVideos.length}>
              Check all
            </button>
            <button type="button" className="danger" onClick={() => void deleteCheckedResults()}
              disabled={importing || busy || checkedResultCount === 0}>
              Delete checked{checkedResultCount ? ` (${checkedResultCount})` : ""}
            </button>
            <button type="button" className="danger" onClick={() => void deleteAllResults()}
              disabled={importing || busy || resultVideos.length === 0}>
              Delete all
            </button>
          </div>
        </div>
        <div className="card video-drop result-drop">
          {resultVideos.length === 0 ? (
            <div className="placeholder small">Finished upscale outputs will appear here.</div>
          ) : (
            <div className="video-source-grid">
              {resultVideos.map((s) => (
                <div key={s.rel_path} className={"video-source-wrap" + (checkedResults.has(s.rel_path) ? " checked" : "")}>
                  <button
                    className={"video-source" + (selectedResult?.rel_path === s.rel_path ? " sel" : "")}
                    onClick={() => {
                      setSelectedResult(s);
                      setResult(null);
                      setResultUrl("");
                    }}
                    title={s.filename}
                  >
                    <video src={s.url} muted preload="metadata" />
                    <span>{s.filename}</span>
                  </button>
                  <label className="video-source-check" title={`Mark ${s.filename} for deletion`}>
                    <input
                      type="checkbox"
                      checked={checkedResults.has(s.rel_path)}
                      onChange={(e) => setResultChecked(s.rel_path, e.target.checked)}
                    />
                  </label>
                  <button
                    type="button"
                    className="video-source-remove"
                    title={`Delete ${s.filename}`}
                    onClick={() => void deleteOneVideo(s)}
                    disabled={importing || busy}
                  >
                    x
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {busy && (
          <div className="card" style={{ marginTop: 12 }}>
            <div className="progress-row">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${pct}%` }} />
              </div>
              <span className="muted small">{pct}%</span>
            </div>
            <div className="muted small" style={{ marginTop: 6 }}>{message || jobStatus}</div>
          </div>
        )}

        {errorMsg && <div className="card error-card" style={{ marginTop: 12 }}>{errorMsg}</div>}
      </main>

      <aside className={"pane right-params" + (active ? "" : " tab-hidden")}>
        <div className="section-title">Video upscale status</div>
        <div className="card">
          <div className="kv">
            <div className="k">cli</div>
            <div className="v" style={{ color: cliReady ? "var(--c-good)" : "var(--c-bad)" }}>
              {cliReady ? "found" : "missing"}
            </div>
            <div className="k">ffmpeg</div>
            <div className="v" style={{ color: ffmpegReady ? "var(--c-good)" : "var(--c-bad)" }}>
              {ffmpegReady ? "found" : "missing"}
            </div>
            <div className="k">models</div>
            <div className="v">{statusInfo?.seedvr2_models.length ?? 0}</div>
            <div className="k">upscalers</div>
            <div className="v">{statusInfo?.upscale_models?.length ?? 0}</div>
            <div className="k">rife</div>
            <div className="v" style={{ color: rifeReady ? "var(--c-good)" : "var(--c-warn)" }}>
              {rifeReady ? "found" : "missing"}
            </div>
          </div>
          {!cliReady && (
            <div className="muted small" style={{ marginTop: 8 }}>
              Put SeedVR2 standalone at external\SeedVR2 or set KRAKEN_SEEDVR2_CLI.
            </div>
          )}
        </div>

        <div className="section-title">Engine</div>
        <select value={engine} onChange={(e) => setEngine(e.target.value as VideoUpscaleParams["engine"])}>
          <option value="seedvr2">SeedVR2 - cinematic detail</option>
          <option value="esrgan">ESRGAN - preserve source</option>
        </select>
        <div className="muted small" style={{ marginTop: 6 }}>
          {engine === "seedvr2"
            ? "Slowest and highest-detail path. Reconstructs missing detail, but can change exact source details."
            : "Fast frame upscaling path. Preserves composition and identity better, but cannot invent convincing missing detail like SeedVR2."}
        </div>

        <div className="section-title">Model</div>
        {engine === "seedvr2" ? (
          <>
            <select value={ditModel} onChange={(e) => setDitModel(e.target.value)}>
              {statusInfo?.seedvr2_models.length === 0 && <option value="">(no SeedVR2 models)</option>}
              {statusInfo?.seedvr2_models.map((m) => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
            <div className="muted small" style={{ marginTop: 6 }}>{seedModelNote(ditModel)}</div>
            <label className="field" style={{ marginTop: 10 }}>
              <span>Preservation base</span>
              <select value={upscaleModel} onChange={(e) => setUpscaleModel(e.target.value)}>
                {(statusInfo?.upscale_models?.length ?? 0) === 0 && <option value="">(no upscale models)</option>}
                {statusInfo?.upscale_models?.map((m) => (
                  <option key={m.name} value={m.name}>{m.name}</option>
                ))}
              </select>
            </label>
            <div className="muted small" style={{ marginTop: 6 }}>{esrganModelNote(upscaleModel)}</div>
          </>
        ) : (
          <>
            <select value={upscaleModel} onChange={(e) => setUpscaleModel(e.target.value)}>
              {(statusInfo?.upscale_models?.length ?? 0) === 0 && <option value="">(no upscale models)</option>}
              {statusInfo?.upscale_models?.map((m) => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
            <div className="muted small" style={{ marginTop: 6 }}>{esrganModelNote(upscaleModel)}</div>
          </>
        )}

        <div className="section-title" style={{ marginTop: 14 }}>Output size</div>
        <label className="field">
          <span>Upscale</span>
          <select value={presetIdx} onChange={(e) => applyPreset(Number(e.target.value))}>
            {SIZE_PRESETS.map((p, i) => <option key={p.label} value={i}>{p.label}</option>)}
          </select>
        </label>
        <div className="muted small" style={{ marginTop: 6 }}>{sizePreset.note}</div>
        {scaleMode === "resolution" && (
          <div className="row2">
            <label className="field">
              <span>Short side</span>
              <input type="number" value={resolution} min={240} step={8} disabled={!customRes}
                onChange={(e) => setResolution(Number(e.target.value))} />
            </label>
            <label className="field">
              <span>Max edge</span>
              <input type="number" value={maxResolution} min={240} step={8} disabled={!customRes}
                onChange={(e) => setMaxResolution(Number(e.target.value))} />
            </label>
          </div>
        )}
        {scaleMode === "factor" && (
          <label className="field">
            <span>Scale factor</span>
            <input type="number" value={upscaleFactor} min={1} max={8} step={0.25} disabled={!customFactor}
              onChange={(e) => setUpscaleFactor(Number(e.target.value))} />
          </label>
        )}

        <div className="section-title" style={{ marginTop: 14 }}>Temporal</div>
        {engine === "seedvr2" && (
          <>
            <SliderNumber label="Batch frames" value={batchSize} min={1} max={81} step={4}
              onChange={(n) => setBatchSize(seedBatch(n))} />
            <SliderNumber label="Overlap" value={temporalOverlap} min={0} max={16} step={1}
              onChange={setTemporalOverlap} />
          </>
        )}
        <div className="row2" style={{ alignItems: "end" }}>
          <SliderNumber label="Framerate" value={targetFps} min={1} max={120} step={1}
            disabled={interpolation === "none"} format={(n) => `${n} fps`} onChange={setTargetFps} />
          <label className="field">
            <span>Interpolation</span>
            <select value={interpolation} onChange={(e) => setInterpolation(e.target.value as VideoUpscaleParams["interpolation"])}>
              <option value="rife_ncnn">RIFE Vulkan</option>
              <option value="ffmpeg_motion">FFmpeg motion</option>
              <option value="none">None</option>
            </select>
          </label>
        </div>
        {engine === "seedvr2" && (
          <label className="checkline">
            <input type="checkbox" checked={uniformBatch} onChange={(e) => setUniformBatch(e.target.checked)} />
            Uniform final batch
          </label>
        )}
        <label className="checkline">
          <input type="checkbox" checked={keepAudio} onChange={(e) => setKeepAudio(e.target.checked)} />
          Keep source audio
        </label>

        <div className="section-title" style={{ marginTop: 14 }}>Quality</div>
        {engine === "seedvr2" ? (
          <>
            <SliderNumber label="AI detail strength" value={aiDetailStrength} min={0} max={1} step={0.01}
              format={(n) => n.toFixed(2)} onChange={setAiDetailStrength} />
            <div className="muted small" style={{ marginTop: 4 }}>
              0.00 uses only the preservation base. 0.10-0.15 keeps mostly source structure. 1.00 is full SeedVR2.
            </div>
            <label className="field">
              <span>Color correction</span>
              <select value={colorCorrection} onChange={(e) => setColorCorrection(e.target.value as VideoUpscaleParams["color_correction"])}>
                {COLOR_MODES.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </label>
            <SliderNumber label="Chunk size" value={chunkSize} min={0} max={999} step={33}
              onChange={setChunkSize} />
            <SliderNumber label="Input noise" value={inputNoiseScale} min={0} max={1} step={0.001}
              format={(n) => n.toFixed(3)} onChange={setInputNoiseScale} />
            <SliderNumber label="Latent noise" value={latentNoiseScale} min={0} max={1} step={0.001}
              format={(n) => n.toFixed(3)} onChange={setLatentNoiseScale} />
            <label className="checkline">
              <input type="checkbox" checked={vaeEncodeTiled} onChange={(e) => setVaeEncodeTiled(e.target.checked)} />
              VAE encode tiling
            </label>
            <SliderNumber label="Encode tile size" value={vaeEncodeTileSize} min={256} max={2048} step={128}
              disabled={!vaeEncodeTiled} onChange={setVaeEncodeTileSize} />
            <SliderNumber label="Encode tile overlap" value={vaeEncodeTileOverlap} min={0} max={512} step={16}
              disabled={!vaeEncodeTiled} onChange={setVaeEncodeTileOverlap} />
            <label className="checkline">
              <input type="checkbox" checked={vaeDecodeTiled} onChange={(e) => setVaeDecodeTiled(e.target.checked)} />
              VAE decode tiling
            </label>
            <SliderNumber label="Decode tile size" value={vaeDecodeTileSize} min={256} max={2048} step={128}
              disabled={!vaeDecodeTiled} onChange={setVaeDecodeTileSize} />
            <SliderNumber label="Decode tile overlap" value={vaeDecodeTileOverlap} min={0} max={512} step={16}
              disabled={!vaeDecodeTiled} onChange={setVaeDecodeTileOverlap} />
            <label className="checkline">
              <input type="checkbox" checked={tenBit} onChange={(e) => setTenBit(e.target.checked)} />
              10-bit ffmpeg output
            </label>

            <div className="section-title" style={{ marginTop: 14 }}>Memory</div>
            <label className="field">
              <span>Attention</span>
              <select value={attentionMode} onChange={(e) => setAttentionMode(e.target.value as VideoUpscaleParams["attention_mode"])}>
                <option value="sdpa">SDPA</option>
                <option value="sageattn_2">SageAttention 2</option>
                <option value="sageattn_3">SageAttention 3</option>
                <option value="flash_attn_2">FlashAttention 2</option>
                <option value="flash_attn_3">FlashAttention 3</option>
              </select>
            </label>
            <SliderNumber label="Blocks to swap" value={blocksToSwap} min={0} max={36} step={1}
              onChange={setBlocksToSwap} />
            <label className="field">
              <span>Seed</span>
              <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
            </label>
            <label className="checkline">
              <input type="checkbox" checked={swapIo} disabled={blocksToSwap <= 0} onChange={(e) => setSwapIo(e.target.checked)} />
              Swap I/O components
            </label>
            <label className="checkline">
              <input type="checkbox" checked={compileDit} onChange={(e) => setCompileDit(e.target.checked)} />
              Compile DiT
            </label>
            <label className="checkline">
              <input type="checkbox" checked={compileVae} onChange={(e) => setCompileVae(e.target.checked)} />
              Compile VAE
            </label>
            <label className="field">
              <span>Compile mode</span>
              <select value={compileMode} disabled={!compileDit && !compileVae}
                onChange={(e) => setCompileMode(e.target.value as VideoUpscaleParams["compile_mode"])}>
                <option value="default">default</option>
                <option value="reduce-overhead">reduce-overhead</option>
                <option value="max-autotune">max-autotune</option>
                <option value="max-autotune-no-cudagraphs">max-autotune-no-cudagraphs</option>
              </select>
            </label>
          </>
        ) : (
          <>
            <SliderNumber label="Tile size" value={esrganTileSize} min={0} max={2048} step={128}
              onChange={setEsrganTileSize} />
            <SliderNumber label="Tile overlap" value={esrganTileOverlap} min={0} max={512} step={16}
              onChange={setEsrganTileOverlap} />
            <div className="muted small" style={{ marginTop: 6 }}>
              Tile size 0 uses full-frame processing. Use 768 or 1024 if 2K/4K runs hit memory limits.
            </div>
          </>
        )}

        <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
          {busy ? (
            <button className="primary" onClick={cancel}>Cancel</button>
          ) : (
            <button className="primary" onClick={run} disabled={!ready}>Upscale video</button>
          )}
        </div>
      </aside>
    </>
  );
}
