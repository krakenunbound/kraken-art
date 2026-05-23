import { useEffect, useMemo, useRef, useState } from "react";
import ImageViewer from "./ImageViewer";
import {
  cancelJob,
  deleteOutputs,
  openJobWS,
  outputFileUrl,
  saveLastGenerate,
  startGenerate,
  type GenerateParams,
  type LastGenerate,
  type LoraEntry,
  type ModelListing,
  type Settings,
} from "./api/sidecar";

const DIM_PRESETS: Array<[string, number, number]> = [
  ["1024 × 1024", 1024, 1024],
  ["1024 × 1536 (portrait)", 1024, 1536],
  ["1536 × 1024 (landscape)", 1536, 1024],
  ["768 × 768", 768, 768],
  ["512 × 512", 512, 512],
  ["Custom", 0, 0],
];

const SAMPLERS   = ["dpmpp_2m", "euler", "euler_a", "ddim", "unipc"];
const SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "beta", "simple"];

const DEFAULT_PROMPT   = "An abyssal kraken coiled around a sunken lighthouse, bioluminescent, cinematic";
const DEFAULT_NEGATIVE = "blurry, low quality, deformed";

// Filename-pattern detection. Given a model filename and the scanned model lists,
// return: detected arch, sensible cfg/steps, and best-guess VAE / text encoders.
// All fields are optional — only what we can infer with confidence is set.
type DetectedConfig = {
  arch?: string;
  cfg?: number;
  steps?: number;
  vae?: string;
  textEncoders?: string[];
};

function pickFirst(list: { name: string }[], ...patterns: string[]): string | undefined {
  for (const pat of patterns) {
    const p = pat.toLowerCase();
    const hit = list.find((m) => m.name.toLowerCase().includes(p));
    if (hit) return hit.name;
  }
  return undefined;
}

function defaultSamplingForArch(archId: string, modelName = ""): { cfg: number; steps: number } {
  const n = modelName.toLowerCase();
  if (n.includes("schnell")) return { cfg: 1.0, steps: 4 };
  if (n.includes("turbo") || n.includes("lightning") || n.includes("lcm") || n.includes("hyper")) {
    return { cfg: 1.0, steps: 8 };
  }
  switch (archId) {
    case "flux1":
    case "flux2":
      return { cfg: 1.0, steps: 28 }; // KSampler CFG off — same slot as SDXL, value 1
    case "z_image":
      return { cfg: 1.0, steps: 28 };
    case "illustrious":
    case "sdxl":
      return { cfg: 7.0, steps: 30 };
    case "sd15":
      return { cfg: 7.0, steps: 25 };
    default:
      return { cfg: 7.0, steps: 30 };
  }
}

function detectFromModelName(name: string, models: ModelListing | null): DetectedConfig {
  const n = name.toLowerCase();
  const vaes  = models?.categories.vae          ?? [];
  const tes   = models?.categories.text_encoders ?? [];
  const out: DetectedConfig = {};

  // ---- distilled / fast variants (any arch) ----
  if (n.includes("schnell")) {
    Object.assign(out, { arch: "flux1", cfg: 1.0, steps: 4 });
  } else if (n.includes("turbo") || n.includes("lightning") || n.includes("lcm") || n.includes("hyper")) {
    out.cfg = 1.0; out.steps = 8;
  }

  // ---- architecture ----
  if (!out.arch) {
    if (n.includes("flux-2") || n.includes("flux2") || n.includes("flux_2"))           out.arch = "flux2";
    else if (n.includes("flux"))                                                       out.arch = "flux1";
    else if (n.includes("qwen_image") || n.includes("qwen-image") || n.includes("qwenimage"))
                                                                                       out.arch = "qwen_image";
    else if (n.includes("z_image") || n.includes("zimage"))                            out.arch = "z_image";
    else if (n.includes("hunyuan"))                                                    out.arch = "hunyuan";
    else if (n.includes("wan"))                                                        out.arch = "wan";
    else if (n.includes("ltx"))                                                        out.arch = "ltx";
    else if (n.includes("illustrious") || n.includes("ilustreal"))                     out.arch = "illustrious";
    else if (n.includes("xl") || n.includes("juggernaut") || n.includes("epicrealism") || n.includes("sdxl"))
                                                                                       out.arch = "sdxl";
  }

  // ---- arch-specific defaults (CFG auto-tuned per architecture) ----
  if (out.arch) {
    const sampling = defaultSamplingForArch(out.arch, name);
    if (out.cfg   === undefined) out.cfg   = sampling.cfg;
    if (out.steps === undefined) out.steps = sampling.steps;
  }

  // ---- arch-specific component auto-pick ----
  switch (out.arch) {
    case "flux1": {
      const vae  = pickFirst(vaes, "fluxvaesft", "/ae.safetensors", "ae.safetensors");
      const clip = pickFirst(tes, "clip_l");
      const t5   = pickFirst(tes, "t5xxl_fp8", "t5xxl_fp16", "t5xxl");  // prefer FP8 to save RAM
      if (vae) out.vae = vae;
      if (clip || t5) out.textEncoders = [clip ?? "", t5 ?? ""];
      break;
    }
    case "flux2": {
      const vae  = pickFirst(vaes, "fluxvaesft", "ae.safetensors");
      const gemma = pickFirst(tes, "gemma");
      if (vae) out.vae = vae;
      if (gemma) out.textEncoders = [gemma];
      break;
    }
    case "qwen_image": {
      const vae = pickFirst(vaes, "qwen_image_vae", "qwen/ae");
      const te  = pickFirst(tes, "qwen_2.5_vl", "qwen_3_4b", "qwen");
      if (vae) out.vae = vae;
      if (te) out.textEncoders = [te];
      break;
    }
    case "z_image": {
      const vae = pickFirst(vaes, "zimageturbo_vae", "zimage_vae");
      const te  = pickFirst(tes, "zimageturbo_textencoder", "zimage");
      if (vae) out.vae = vae;
      if (te) out.textEncoders = [te];
      break;
    }
    case "wan": {
      const vae = pickFirst(vaes, "wan_2.1_vae", "wan");
      const te  = pickFirst(tes, "umt5_xxl", "umt5");
      if (vae) out.vae = vae;
      if (te) out.textEncoders = [te];
      break;
    }
  }

  return out;
}

// Profile drives which model selectors are visible per architecture.
type ArchProfile = {
  id: string;
  label: string;
  mode: "checkpoint" | "components";
  encoders: number;          // how many text encoder slots
  upscaleAllowed: boolean;
  supported: boolean;        // backend can actually run this today
};

const ARCH_PROFILES: ArchProfile[] = [
  { id: "sdxl",         label: "SDXL (all-in-one)",        mode: "checkpoint", encoders: 0, upscaleAllowed: true,  supported: true },
  { id: "illustrious",  label: "Illustrious (SDXL variant)", mode: "checkpoint", encoders: 0, upscaleAllowed: true,  supported: true },
  { id: "flux1",        label: "FLUX1 (components)",        mode: "components", encoders: 2, upscaleAllowed: true,  supported: true },
  { id: "flux2",        label: "FLUX2 (components) — Phase 3", mode: "components", encoders: 2, upscaleAllowed: true,  supported: false },
  { id: "qwen_image",   label: "Qwen-Image — Phase 3",      mode: "components", encoders: 1, upscaleAllowed: true,  supported: false },
  { id: "hunyuan",      label: "HunYuan — Phase 3",         mode: "components", encoders: 1, upscaleAllowed: true,  supported: false },
  { id: "wan",          label: "WAN (video) — Phase 4",     mode: "components", encoders: 1, upscaleAllowed: false, supported: false },
  { id: "ltx",          label: "LTX (video) — Phase 4",     mode: "components", encoders: 1, upscaleAllowed: false, supported: false },
];

type Preview = { idx: number; b64: string; seed: number; filename: string; rel_path: string };

export default function Generate({
  models,
  settings,
}: {
  models: ModelListing | null;
  settings?: Settings | null;
}) {
  const checkpoints   = models?.categories.checkpoints      ?? [];
  const diffusion     = models?.categories.diffusion_models ?? [];
  const vaes          = models?.categories.vae              ?? [];
  const textEncoders  = models?.categories.text_encoders    ?? [];
  const lorasAvail    = models?.categories.loras            ?? [];
  const embeddingsAvail = models?.categories.embeddings     ?? [];
  const upscalers     = models?.categories.upscale_models   ?? [];

  // ---- selections ----
  const [archId, setArchId] = useState<string>("sdxl");
  const profile = useMemo<ArchProfile>(
    () => ARCH_PROFILES.find((p) => p.id === archId) ?? ARCH_PROFILES[0],
    [archId]
  );

  const [checkpoint, setCheckpoint] = useState<string>("");
  const [diffusionModel, setDiffusionModel] = useState<string>("");
  const [vae, setVae] = useState<string>("");
  const [te, setTe] = useState<string[]>([]);  // up to profile.encoders entries

  const [loras, setLoras] = useState<LoraEntry[]>([]);
  const [loraToAdd, setLoraToAdd] = useState<string>("");

  const [embeddings, setEmbeddings] = useState<string[]>([]);
  const [embToAdd, setEmbToAdd] = useState<string>("");

  // ---- prompt + image params ----
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [preset, setPreset] = useState("1024 × 1024");
  const [w, setW] = useState(1024);
  const [h, setH] = useState(1024);
  const [steps, setSteps] = useState(30);
  const [cfg, setCfg] = useState(7.0);
  const [sampler, setSampler] = useState("dpmpp_2m");
  const [scheduler, setScheduler] = useState("normal");
  const [count, setCount] = useState(1);
  const [randomSeed, setRandomSeed] = useState(true);
  const [seed, setSeed] = useState<number>(0);

  // ---- upscale ----
  const [upEnabled, setUpEnabled] = useState(false);
  const [upMode, setUpMode] = useState<"esrgan" | "usdu" | "iterative">("esrgan");
  const [upModel, setUpModel] = useState<string>("");
  const [upFactor, setUpFactor] = useState(2.0);
  const [upDenoise, setUpDenoise] = useState(0.35);
  const [upTile, setUpTile] = useState(512);

  // Auto-pick the first sensible model when the list arrives.
  useEffect(() => {
    if (profile.mode === "checkpoint" && !checkpoint && checkpoints.length) setCheckpoint(checkpoints[0].name);
    if (profile.mode === "components" && !diffusionModel && diffusion.length) setDiffusionModel(diffusion[0].name);
  }, [profile.mode, checkpoints, diffusion, checkpoint, diffusionModel]);

  // ---------- Last-used persistence (remember what the user had selected) ----------
  const restoreDoneRef = useRef(false);
  const saveTimeoutRef = useRef<number | null>(null);

  function scheduleSave() {
    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    saveTimeoutRef.current = window.setTimeout(() => {
      const snapshot: LastGenerate = {
        archId,
        checkpoint: checkpoint || undefined,
        diffusionModel: diffusionModel || undefined,
        vae: vae || undefined,
        te: te.length ? [...te] : undefined,
        prompt: prompt || undefined,
        negative: negative || undefined,
        w,
        h,
        steps,
        cfg,
        sampler,
        scheduler,
      };
      // Fire and forget
      saveLastGenerate(snapshot).catch(() => {});
    }, 650); // debounce a bit so we don't spam on every keystroke
  }

  // Restore from lastGenerate the first time we have both models and settings
  useEffect(() => {
    if (restoreDoneRef.current) return;
    if (!models || !settings?.lastGenerate) return;

    const lg = settings.lastGenerate;
    let didRestore = false;

    // Helper to check if a name still exists in the current scan
    const existsIn = (cat: string, name?: string) =>
      !!name && (models.categories[cat] ?? []).some((m) => m.name === name);

    if (lg.archId && ARCH_PROFILES.some((p) => p.id === lg.archId)) {
      setArchId(lg.archId);
      didRestore = true;
    }

    if (lg.checkpoint && existsIn("checkpoints", lg.checkpoint)) {
      setCheckpoint(lg.checkpoint);
      didRestore = true;
    }
    if (lg.diffusionModel && existsIn("diffusion_models", lg.diffusionModel)) {
      setDiffusionModel(lg.diffusionModel);
      didRestore = true;
    }
    if (lg.vae && existsIn("vae", lg.vae)) {
      setVae(lg.vae);
      didRestore = true;
    }
    if (lg.te?.length) {
      // Only keep entries that still exist
      const valid = lg.te.filter((name: string) => existsIn("text_encoders", name));
      if (valid.length) {
        setTe(valid);
        didRestore = true;
      }
    }

    if (typeof lg.prompt === "string") { setPrompt(lg.prompt); didRestore = true; }
    if (typeof lg.negative === "string") { setNegative(lg.negative); didRestore = true; }
    if (typeof lg.w === "number" && lg.w > 0) { setW(lg.w); didRestore = true; }
    if (typeof lg.h === "number" && lg.h > 0) { setH(lg.h); didRestore = true; }
    if (typeof lg.steps === "number") { setSteps(lg.steps); didRestore = true; }
    if (typeof lg.cfg === "number") { setCfg(lg.cfg); didRestore = true; }
    if (lg.sampler) { setSampler(lg.sampler); didRestore = true; }
    if (lg.scheduler) { setScheduler(lg.scheduler); didRestore = true; }

    if (didRestore) {
      // mark that we restored so the next change effects don't immediately overwrite with old values
      restoreDoneRef.current = true;
    }
  }, [models, settings]); // run when both become available

  // Whenever key fields change *after* restore, schedule a save
  useEffect(() => {
    if (!restoreDoneRef.current) return; // don't save the initial auto-pick or the restore itself
    scheduleSave();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [archId, checkpoint, diffusionModel, vae, JSON.stringify(te), prompt, negative, w, h, steps, cfg, sampler, scheduler]);

  // Cleanup any pending save timer on unmount
  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    };
  }, []);

  // Auto-detect arch + tune CFG/steps + auto-fill VAE / text encoders when the
  // user changes the primary model file. Filename-pattern heuristics.
  const primaryModel = profile.mode === "checkpoint" ? checkpoint : diffusionModel;
  useEffect(() => {
    if (!primaryModel) return;
    const d = detectFromModelName(primaryModel, models);
    if (d.arch && d.arch !== archId) setArchId(d.arch);
    if (d.vae) setVae(d.vae);
    if (d.textEncoders) {
      setTe((cur) => {
        const next = [...cur];
        d.textEncoders!.forEach((name, i) => { if (name) next[i] = name; });
        return next;
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [primaryModel]);

  // Re-apply arch-appropriate CFG/steps when architecture or primary model changes.
  useEffect(() => {
    const { cfg: dCfg, steps: dSteps } = defaultSamplingForArch(archId, primaryModel);
    setCfg(dCfg);
    setSteps(dSteps);
  }, [archId, primaryModel]);

  useEffect(() => {
    // shrink/expand te[] to match profile.encoders
    setTe((cur) => {
      if (cur.length === profile.encoders) return cur;
      const next = cur.slice(0, profile.encoders);
      while (next.length < profile.encoders) next.push("");
      return next;
    });
  }, [profile.encoders]);

  useEffect(() => {
    if (upEnabled && !upModel && upscalers.length) setUpModel(upscalers[0].name);
  }, [upEnabled, upscalers, upModel]);

  const dimsLocked = useMemo(() => preset !== "Custom", [preset]);

  function applyPreset(name: string) {
    setPreset(name);
    const found = DIM_PRESETS.find((p) => p[0] === name);
    if (found && found[0] !== "Custom") { setW(found[1]); setH(found[2]); }
  }

  function addLora() {
    if (!loraToAdd) return;
    if (loras.some((l) => l.name === loraToAdd)) return;
    setLoras((cur) => [...cur, { name: loraToAdd, weight: 1.0 }]);
    setLoraToAdd("");
  }
  function setLoraWeight(name: string, weight: number) {
    setLoras((cur) => cur.map((l) => (l.name === name ? { ...l, weight } : l)));
  }
  function removeLora(name: string) {
    setLoras((cur) => cur.filter((l) => l.name !== name));
  }

  function addEmbedding() {
    if (!embToAdd) return;
    if (embeddings.includes(embToAdd)) return;
    setEmbeddings((cur) => [...cur, embToAdd]);
    setEmbToAdd("");
  }
  function removeEmbedding(name: string) {
    setEmbeddings((cur) => cur.filter((n) => n !== name));
  }

  // --- runtime ---
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [step, setStep] = useState({ step: 0, total: 0 });
  const [imgIdx, setImgIdx] = useState({ i: 0, total: 0 });
  const [previews, setPreviews] = useState<Preview[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // ---- selection + viewer ----
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [viewerIdx, setViewerIdx] = useState<number | null>(null);
  const [viewerUrl, setViewerUrl] = useState<string>("");
  const [deleting, setDeleting] = useState(false);

  // Resolve the sidecar URL for the active viewer image whenever its index changes.
  useEffect(() => {
    if (viewerIdx === null) { setViewerUrl(""); return; }
    const p = previews[viewerIdx];
    if (!p) return;
    let cancelled = false;
    outputFileUrl(p.rel_path).then((url) => { if (!cancelled) setViewerUrl(url); });
    return () => { cancelled = true; };
  }, [viewerIdx, previews]);

  function toggleSelect(filename: string, e: React.MouseEvent | React.ChangeEvent) {
    if ("stopPropagation" in e) e.stopPropagation();
    setSelected((cur) => {
      const next = new Set(cur);
      if (next.has(filename)) next.delete(filename); else next.add(filename);
      return next;
    });
  }

  function selectAll() {
    setSelected(new Set(previews.map((p) => p.filename)));
  }
  function clearSelection() { setSelected(new Set()); }

  function clearGallery() {
    if (previews.length === 0) return;
    if (!window.confirm(`Remove all ${previews.length} thumbnails from the gallery? (files on disk are kept)`)) return;
    setPreviews([]);
    setSelected(new Set());
  }

  async function deleteSelected() {
    if (selected.size === 0) return;
    if (!window.confirm(`Delete ${selected.size} image${selected.size === 1 ? "" : "s"}? This is permanent.`)) return;
    setDeleting(true);
    try {
      const paths = previews.filter((p) => selected.has(p.filename)).map((p) => p.rel_path);
      const r = await deleteOutputs(paths);
      setPreviews((cur) => cur.filter((p) => !r.deleted.includes(p.rel_path)));
      setSelected(new Set());
      if (r.errors.length) setErrorMsg(`some files couldn't be deleted: ${r.errors.map((e) => e.path).join(", ")}`);
    } catch (e: any) {
      setErrorMsg(`delete failed: ${e?.message ?? e}`);
    } finally {
      setDeleting(false);
    }
  }

  async function submit() {
    setErrorMsg(null);
    // Don't clear previews — let them accumulate across runs so the user can
    // compare. Use "Clear gallery" to wipe.
    setStep({ step: 0, total: steps });
    setImgIdx({ i: 0, total: count });
    setStatus("submitting");

    const params: GenerateParams = {
      arch: archId,
      checkpoint:      profile.mode === "checkpoint" ? (checkpoint || null) : null,
      diffusion_model: profile.mode === "components" ? (diffusionModel || null) : null,
      vae: vae || null,
      text_encoders: te.filter(Boolean),
      loras,
      embeddings,
      prompt:   prompt.trim()   || DEFAULT_PROMPT,
      negative: negative.trim() || DEFAULT_NEGATIVE,
      width: w,
      height: h,
      steps,
      cfg,
      sampler,
      scheduler,
      count,
      seed: randomSeed ? null : seed,
      upscale_enabled: upEnabled,
      upscale_mode: upMode,
      upscale_model: upEnabled ? (upModel || null) : null,
      upscale_factor: upFactor,
      upscale_denoise: upDenoise,
      upscale_tile_size: upTile,
    };

    try {
      const r = await startGenerate(params);
      setJobId(r.job_id);
      setStatus(r.status);
      const ws = await openJobWS(r.job_id);
      wsRef.current = ws;
      ws.onmessage = (e) => {
        const evt = JSON.parse(e.data);
        if (evt.type === "snapshot") setStatus(evt.status);
        if (evt.type === "status") {
          setStatus(evt.status);
          if (evt.error) setErrorMsg(evt.error);
        }
        if (evt.type === "progress") {
          setStep({ step: evt.step, total: evt.total_steps });
          setImgIdx({ i: evt.image_index, total: evt.total_images });
        }
        if (evt.type === "image") {
          setPreviews((p) => [...p, {
            idx: evt.image_index, b64: evt.preview_b64, seed: evt.seed,
            filename: evt.filename, rel_path: evt.rel_path ?? "",
          }]);
        }
      };
      const resetIfStuck = (msg: string) => {
        wsRef.current = null;
        setStatus((cur) => {
          if (cur === "queued" || cur === "running" || cur === "submitting") {
            setErrorMsg(msg);
            return "idle";
          }
          return cur;
        });
      };
      ws.onclose = () => resetIfStuck("Connection to sidecar lost — likely crashed mid-job (check Logs). Generate again to retry.");
      ws.onerror = () => resetIfStuck("WebSocket error — sidecar may have died. Check Logs.");
    } catch (e: any) {
      setStatus("idle");
      setErrorMsg(String(e?.message ?? e));
    }
  }

  async function cancel() {
    if (!jobId) return;
    try { await cancelJob(jobId); } catch { /* ignore */ }
  }

  const busy = status === "queued" || status === "running" || status === "submitting";
  const overallPct =
    imgIdx.total > 0
      ? Math.floor(((imgIdx.i * step.total + step.step) / (imgIdx.total * Math.max(step.total, 1))) * 100)
      : 0;

  const primaryModelMissing =
    (profile.mode === "checkpoint" && !checkpoint) ||
    (profile.mode === "components" && !diffusionModel);

  const loraOptions = lorasAvail.filter((l) => !loras.some((sel) => sel.name === l.name));
  const embOptions  = embeddingsAvail.filter((e) => !embeddings.includes(e.name));

  return (
    <>
      {/* Center pane */}
      <main className="pane center gen-center">
        <div className="prompt-stack">
          <label className="lbl">Prompt</label>
          <textarea rows={4} value={prompt} onChange={(e) => setPrompt(e.target.value)}
            placeholder={DEFAULT_PROMPT} />
          <label className="lbl">Negative</label>
          <textarea rows={2} value={negative} onChange={(e) => setNegative(e.target.value)}
            placeholder={DEFAULT_NEGATIVE} />
        </div>

        <div className="action-row">
          <button className="primary big" onClick={submit} disabled={busy || primaryModelMissing}>
            {busy ? "Generating…" : "Generate"}
          </button>
          {busy && <button onClick={cancel}>Cancel</button>}
          <div className="spacer" />
          {status !== "idle" && (
            <span className="muted small">
              {status} · {imgIdx.i + 1}/{Math.max(imgIdx.total, 1)} · step {step.step}/{step.total || "?"}
            </span>
          )}
        </div>

        {busy && (
          <div className="progress-bar">
            <div className="progress-bar-fill" style={{ width: `${overallPct}%` }} />
          </div>
        )}

        {errorMsg && <div className="err">{errorMsg}</div>}

        {previews.length > 0 && (
          <div className="thumb-toolbar">
            <span className="muted small">{previews.length} image{previews.length === 1 ? "" : "s"}</span>
            {selected.size > 0 && (
              <>
                <span className="muted small">· {selected.size} selected</span>
                <button onClick={clearSelection}>Deselect</button>
                <button onClick={deleteSelected} disabled={deleting} style={{ color: "var(--c-bad)", borderColor: "var(--c-bad)" }}>
                  {deleting ? "Deleting…" : `Delete ${selected.size}`}
                </button>
              </>
            )}
            <div className="spacer" />
            <button onClick={selectAll} className="muted small">Select all</button>
            <button onClick={clearGallery} className="muted small" title="Remove thumbnails from this view (files on disk are kept)">Clear gallery</button>
          </div>
        )}
        <div className="thumb-grid">
          {previews.map((p, i) => {
            const isSelected = selected.has(p.filename);
            return (
              <figure
                className={"thumb" + (isSelected ? " selected" : "")}
                key={p.rel_path || p.filename || `${p.idx}-${p.seed}`}
                onClick={() => p.rel_path && setViewerIdx(i)}
              >
                <label className="thumb-check" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={(e) => toggleSelect(p.filename, e)}
                  />
                </label>
                <img src={`data:image/jpeg;base64,${p.b64}`} alt={p.filename} draggable={false} />
                <figcaption className="muted small">{p.filename} · seed {p.seed}</figcaption>
              </figure>
            );
          })}
          {!previews.length && !busy && (
            <div className="placeholder small">Generated images appear here</div>
          )}
        </div>

        {viewerIdx !== null && viewerUrl && previews[viewerIdx] && (
          <ImageViewer
            url={viewerUrl}
            caption={`${previews[viewerIdx].filename} · seed ${previews[viewerIdx].seed}`}
            onClose={() => setViewerIdx(null)}
            onPrev={viewerIdx > 0 ? () => setViewerIdx(viewerIdx - 1) : undefined}
            onNext={viewerIdx < previews.length - 1 ? () => setViewerIdx(viewerIdx + 1) : undefined}
          />
        )}
      </main>

      {/* Right pane — scrollable */}
      <aside className="pane right-params">
        <div className="section-title">Architecture</div>
        <div className="field">
          <select value={archId} onChange={(e) => setArchId(e.target.value)}>
            {ARCH_PROFILES.map((p) => (
              <option key={p.id} value={p.id}>{p.label}{p.supported ? "" : "  ⏳"}</option>
            ))}
          </select>
          {!profile.supported && (
            <div className="muted small" style={{ marginTop: 4 }}>
              Backend will reject — pipeline implementation pending.
            </div>
          )}
        </div>

        <div className="section-title">Model</div>
        {profile.mode === "checkpoint" ? (
          <div className="field">
            <label>Checkpoint (all-in-one)</label>
            <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
              {checkpoints.length === 0 && <option value="">— none scanned —</option>}
              {checkpoints.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          </div>
        ) : (
          <div className="field">
            <label>Diffusion model</label>
            <select value={diffusionModel} onChange={(e) => setDiffusionModel(e.target.value)}>
              {diffusion.length === 0 && <option value="">— none scanned —</option>}
              {diffusion.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          </div>
        )}

        <div className="field">
          <label>VAE {profile.mode === "checkpoint" ? "(override, optional)" : ""}</label>
          <select value={vae} onChange={(e) => setVae(e.target.value)}>
            <option value="">{profile.mode === "checkpoint" ? "(bundled)" : "— pick one —"}</option>
            {vaes.map((v) => <option key={v.name} value={v.name}>{v.name}</option>)}
          </select>
        </div>

        {profile.encoders > 0 && (
          <>
            {Array.from({ length: profile.encoders }, (_, i) => (
              <div className="field" key={`te-${i}`}>
                <label>Text encoder {i + 1}</label>
                <select value={te[i] ?? ""}
                  onChange={(e) => setTe((cur) => { const x = [...cur]; x[i] = e.target.value; return x; })}>
                  <option value="">— pick one —</option>
                  {textEncoders.map((t) => <option key={t.name} value={t.name}>{t.name}</option>)}
                </select>
              </div>
            ))}
          </>
        )}

        <div className="section-title">LoRAs <span className="muted small">({loras.length})</span></div>
        <div className="lora-add">
          <select value={loraToAdd} onChange={(e) => setLoraToAdd(e.target.value)}>
            <option value="">{loraOptions.length ? "— add LoRA —" : "(none available)"}</option>
            {loraOptions.map((l) => <option key={l.name} value={l.name}>{l.name}</option>)}
          </select>
          <button onClick={addLora} disabled={!loraToAdd}>＋</button>
        </div>
        {loras.map((l) => (
          <div className="lora-row" key={l.name}>
            <div className="lora-name" title={l.name}>{l.name.split("/").pop()}</div>
            <input type="range" min={-1} max={2} step={0.05} value={l.weight}
              onChange={(e) => setLoraWeight(l.name, parseFloat(e.target.value))} />
            <input type="number" min={-1} max={2} step={0.05} value={l.weight}
              onChange={(e) => setLoraWeight(l.name, parseFloat(e.target.value) || 0)} className="weight-num" />
            <button onClick={() => removeLora(l.name)} className="x">✕</button>
          </div>
        ))}

        <div className="section-title">Embeddings <span className="muted small">({embeddings.length})</span></div>
        <div className="lora-add">
          <select value={embToAdd} onChange={(e) => setEmbToAdd(e.target.value)}>
            <option value="">{embOptions.length ? "— add embedding —" : "(none scanned)"}</option>
            {embOptions.map((l) => <option key={l.name} value={l.name}>{l.name}</option>)}
          </select>
          <button onClick={addEmbedding} disabled={!embToAdd}>＋</button>
        </div>
        {embeddings.map((n) => (
          <div className="lora-row" key={n}>
            <div className="lora-name">{n}</div>
            <button onClick={() => removeEmbedding(n)} className="x">✕</button>
          </div>
        ))}

        <div className="section-title">Dimensions</div>
        <div className="field">
          <select value={preset} onChange={(e) => applyPreset(e.target.value)}>
            {DIM_PRESETS.map(([n]) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div className="field-row">
          <div className="field">
            <label>Width</label>
            <input type="number" min={64} step={8} value={w}
              onChange={(e) => setW(parseInt(e.target.value) || 0)} disabled={dimsLocked} />
          </div>
          <div className="field">
            <label>Height</label>
            <input type="number" min={64} step={8} value={h}
              onChange={(e) => setH(parseInt(e.target.value) || 0)} disabled={dimsLocked} />
          </div>
        </div>

        <div className="section-title">Sampling</div>
        <div className="field-row">
          <div className="field">
            <label>Steps</label>
            <input type="number" min={1} max={150} value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value) || 30)} />
          </div>
          <div className="field">
            <label title={
              archId === "flux1" || archId === "flux2"
                ? "ComfyUI KSampler CFG — use 1.0 to disable (not SDXL-style guidance)"
                : "Classifier-free guidance scale"
            }>
              CFG{archId === "flux1" || archId === "flux2" ? " (1 = off)" : ""}
            </label>
            <input type="number" min={0} step={0.1} value={cfg}
              onChange={(e) => {
                const fallback = (archId === "flux1" || archId === "flux2" || archId === "z_image") ? 1.0 : 7.0;
                setCfg(parseFloat(e.target.value) || fallback);
              }} />
          </div>
        </div>
        <div className="field-row">
          <div className="field">
            <label>Sampler</label>
            <select value={sampler} onChange={(e) => setSampler(e.target.value)}>
              {SAMPLERS.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Scheduler</label>
            <select value={scheduler} onChange={(e) => setScheduler(e.target.value)}>
              {SCHEDULERS.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>

        <div className="section-title">Batch</div>
        <div className="field-row">
          <div className="field">
            <label>Count</label>
            <input type="number" min={1} max={64} value={count}
              onChange={(e) => setCount(parseInt(e.target.value) || 1)} />
          </div>
          <div className="field">
            <label>
              <input type="checkbox" checked={randomSeed}
                onChange={(e) => setRandomSeed(e.target.checked)} /> random seed
            </label>
          </div>
        </div>
        {!randomSeed && (
          <div className="field">
            <label>Seed</label>
            <input type="number" value={seed} onChange={(e) => setSeed(parseInt(e.target.value) || 0)} />
          </div>
        )}

        {profile.upscaleAllowed && (
          <>
            <div className="section-title">Upscale</div>
            <div className="field">
              <label>
                <input type="checkbox" checked={upEnabled}
                  onChange={(e) => setUpEnabled(e.target.checked)} /> enabled
              </label>
            </div>
            {upEnabled && (
              <>
                <div className="field">
                  <label>Mode</label>
                  <select value={upMode} onChange={(e) => setUpMode(e.target.value as any)}>
                    <option value="esrgan">ESRGAN (single-pass)</option>
                    <option value="usdu">Ultimate SD Upscale (tile + img2img)</option>
                    <option value="iterative">Iterative (progressive)</option>
                  </select>
                  <div className="muted small">USDU + Iterative are stubbed; ESRGAN lands first (task #10).</div>
                </div>
                <div className="field">
                  <label>Upscale model</label>
                  <select value={upModel} onChange={(e) => setUpModel(e.target.value)}>
                    {upscalers.length === 0 && <option value="">— none scanned —</option>}
                    {upscalers.map((u) => <option key={u.name} value={u.name}>{u.name}</option>)}
                  </select>
                </div>
                <div className="field-row">
                  <div className="field">
                    <label>Factor</label>
                    <input type="number" min={1.0} max={8.0} step={0.5} value={upFactor}
                      onChange={(e) => setUpFactor(parseFloat(e.target.value) || 2.0)} />
                  </div>
                  {upMode !== "esrgan" && (
                    <div className="field">
                      <label>Denoise</label>
                      <input type="number" min={0} max={1} step={0.05} value={upDenoise}
                        onChange={(e) => setUpDenoise(parseFloat(e.target.value) || 0)} />
                    </div>
                  )}
                </div>
                {upMode === "usdu" && (
                  <div className="field">
                    <label>Tile size</label>
                    <input type="number" min={128} max={2048} step={64} value={upTile}
                      onChange={(e) => setUpTile(parseInt(e.target.value) || 512)} />
                  </div>
                )}
              </>
            )}
          </>
        )}
      </aside>
    </>
  );
}
