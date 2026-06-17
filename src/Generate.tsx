import { useEffect, useMemo, useRef, useState } from "react";
import ImageViewer from "./ImageViewer";
import {
  cancelJob,
  deleteOutputs,
  getOutputSettings,
  getLatestImages,
  ideogramMagicPrompt,
  promptBuilderOptions,
  promptBuilderBuild,
  type PromptBuilderOptions,
  openJobWS,
  outputFileUrl,
  saveLastGenerate,
  startGenerate,
  type GenerateParams,
  type LastGenerate,
  type LoraEntry,
  type ModelEntry,
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

const SAMPLERS = [
  "dpmpp_2m",
  "dpmpp_2m_sde",
  "dpmpp_3m",
  "dpmpp_sde",
  "dpmpp_2s_a",
  "euler",
  "euler_a",
  "heun",
  "dpm_2",
  "dpm_2_a",
  "lms",
  "ddim",
  "unipc",
  "deis",
  "pndm",
  "lcm",
];
const SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "beta", "simple", "ddim_uniform", "trailing"];

const DEFAULT_PROMPT   = "An abyssal kraken coiled around a sunken lighthouse, bioluminescent, cinematic";
const DEFAULT_NEGATIVE = "blurry, low quality, deformed";

function loraModelWeight(lora: LoraEntry): number {
  return typeof lora.model_weight === "number"
    ? lora.model_weight
    : typeof lora.weight === "number"
      ? lora.weight
      : 1.0;
}

function normalizeLora(lora: LoraEntry): LoraEntry {
  return { name: lora.name, model_weight: loraModelWeight(lora), weight: loraModelWeight(lora) };
}

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
    case "ideogram4":
      return { cfg: 7.0, steps: 12 };
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
    if (n.includes("ideogram"))                                                        out.arch = "ideogram4";
    else if (n.includes("flux-2") || n.includes("flux2") || n.includes("flux_2"))      out.arch = "flux2";
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

function modelMatchesArch(model: ModelEntry, archId: string): boolean {
  if (model.detected_arch) return model.detected_arch === archId;

  const n = `${model.name} ${model.base_model ?? ""}`.toLowerCase();
  const has = (...parts: string[]) => parts.some((p) => n.includes(p));

  switch (archId) {
    case "illustrious":
      return has("illustrious", "ilust", "illu xl", "illustriousxl");
    case "sdxl":
      return has("sdxl", "xl", "juggernaut", "epicrealism", "perfectdeliberate", "pony")
        && !has("illustrious", "ilust", "flux", "qwen", "zimage", "z-image", "hunyuan", "wan", "ltx");
    case "flux1":
      return has("flux", "flux.1", "flux1") && !has("flux-2", "flux2", "flux_2");
    case "flux2":
      return has("flux-2", "flux2", "flux_2");
    case "ideogram4":
      return has("ideogram", "ideogram-4");
    case "qwen_image":
      return has("qwen_image", "qwen-image", "qwenimage", "qwen/");
    case "z_image":
      return has("z_image", "z-image", "zimage");
    case "hunyuan":
      return has("hunyuan");
    case "wan":
      return has("wan");
    case "ltx":
      return has("ltx");
    default:
      return false;
  }
}

function modelIsSelectable(model: ModelEntry): boolean {
  return !model.disabled_by_default && model.source_type !== "missing";
}

function modelOptionLabel(model: ModelEntry): string {
  if (model.disabled_by_default) return `${model.name} (guarded)`;
  if (model.source_type === "missing") return `${model.name} (missing)`;
  return model.name;
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
  { id: "ideogram4",    label: "Ideogram 4 (open weights)", mode: "components", encoders: 0, upscaleAllowed: false, supported: true },
  { id: "flux2",        label: "FLUX2 (components) — Phase 3", mode: "components", encoders: 2, upscaleAllowed: true,  supported: false },
  { id: "z_image",      label: "Z-Image (components)",      mode: "components", encoders: 1, upscaleAllowed: true,  supported: true },
  { id: "qwen_image",   label: "Qwen-Image — Phase 3",      mode: "components", encoders: 1, upscaleAllowed: true,  supported: false },
  { id: "hunyuan",      label: "HunYuan — Phase 3",         mode: "components", encoders: 1, upscaleAllowed: true,  supported: false },
  { id: "wan",          label: "WAN (video) — Phase 4",     mode: "components", encoders: 1, upscaleAllowed: false, supported: false },
  { id: "ltx",          label: "LTX (video) — Phase 4",     mode: "components", encoders: 1, upscaleAllowed: false, supported: false },
];

const GALLERY_LIMIT = 20;

type Preview = {
  idx: number;
  b64?: string;
  src?: string;
  seed: number;
  filename: string;
  rel_path: string;
  model_label?: string;
  batch_key?: string;
};
type ImageContextMenu = { x: number; y: number; preview: Preview };

function modelLabelForDisplay(modelName: string, archLabel: string): string {
  const label = (modelName || archLabel || "Unknown model").trim();
  return label.replace(/\.(safetensors|ckpt|pt|bin)$/i, "");
}

function batchKeyForOutput(relPath: string, filename: string): string {
  const folder = relPath.includes("/") ? relPath.slice(0, relPath.lastIndexOf("/")) : "";
  const stem = filename.replace(/\.[^.]+$/, "");
  const batchStem = stem.replace(/-\d+$/, "");
  return `${folder}/${batchStem}`;
}

export default function Generate({
  active,
  models,
  settings,
}: {
  active: boolean;
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
  const filteredCheckpoints = useMemo(
    () => checkpoints.filter((m) => modelMatchesArch(m, archId)),
    [checkpoints, archId]
  );
  const filteredDiffusion = useMemo(
    () => diffusion.filter((m) => modelMatchesArch(m, archId)),
    [diffusion, archId]
  );
  const selectableCheckpoints = useMemo(
    () => filteredCheckpoints.filter(modelIsSelectable),
    [filteredCheckpoints]
  );
  const selectableDiffusion = useMemo(
    () => filteredDiffusion.filter(modelIsSelectable),
    [filteredDiffusion]
  );
  const filteredLoras = useMemo(
    () => lorasAvail.filter((m) => modelMatchesArch(m, archId)),
    [lorasAvail, archId]
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
  const [ideogramMagic, setIdeogramMagic] = useState(true);
  const [ideogramMagicMode, setIdeogramMagicMode] = useState<"local" | "api" | "raw">("local");
  // Adaptive velocity-cache speed/quality knob (Ideogram only). high = ~2x faster
  // at near-identical quality on a 3090; measured 2026-06-17.
  const [ideogramSpeedMode, setIdeogramSpeedMode] = useState<"max" | "high" | "fast">("high");

  // ---- Prompt Builder (deterministic, renders to the current arch's dialect) ----
  const [pbOpen, setPbOpen] = useState(false);
  const [pbOptions, setPbOptions] = useState<PromptBuilderOptions | null>(null);
  const [pbStyle, setPbStyle] = useState("auto");
  const [pbLighting, setPbLighting] = useState("auto");
  const [pbCamera, setPbCamera] = useState("auto");
  const [pbMood, setPbMood] = useState("auto");
  const [pbBusy, setPbBusy] = useState(false);

  useEffect(() => {
    promptBuilderOptions().then(setPbOptions).catch(() => setPbOptions(null));
  }, []);

  async function buildPrompt() {
    setPbBusy(true);
    try {
      const r = await promptBuilderBuild({
        arch: archId,
        subject: prompt.trim() || DEFAULT_PROMPT,
        negative,
        style: pbStyle, lighting: pbLighting, camera: pbCamera, mood: pbMood,
        width: w, height: h,
      });
      setPrompt(r.prompt);
    } catch (e: any) {
      window.alert(`Prompt builder failed: ${e?.message ?? e}`);
    } finally {
      setPbBusy(false);
    }
  }
  const [magicBusy, setMagicBusy] = useState(false);
  const [preset, setPreset] = useState("1024 × 1024");
  const [w, setW] = useState(1024);
  const [h, setH] = useState(1024);
  const [steps, setSteps] = useState(30);
  const [cfg, setCfg] = useState(7.0);
  const [sampler, setSampler] = useState("dpmpp_2m");
  const [scheduler, setScheduler] = useState("normal");
  const [clipSkip, setClipSkip] = useState(0);
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
    if (profile.mode === "checkpoint") {
      const stillValid = !!checkpoint && selectableCheckpoints.some((m) => m.name === checkpoint);
      if (!stillValid) setCheckpoint(selectableCheckpoints[0]?.name ?? "");
      if (vae) setVae("");
      return;
    }

    const stillValid = !!diffusionModel && selectableDiffusion.some((m) => m.name === diffusionModel);
    if (!stillValid) setDiffusionModel(selectableDiffusion[0]?.name ?? "");
  }, [profile.mode, selectableCheckpoints, selectableDiffusion, checkpoint, diffusionModel]);

  // ---------- Last-used persistence (remember what the user had selected) ----------
  const restoreDoneRef = useRef(false);
  const restoredPrimaryModelRef = useRef<string | null>(null);
  const appliedRecommendationsForRef = useRef<string | null>(null);
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
        loras,
        embeddings,
        prompt: prompt || undefined,
        negative: negative || undefined,
        ideogramMagic,
        ideogramMagicMode,
        ideogramSpeedMode,
        preset,
        w,
        h,
        steps,
        cfg,
        sampler,
        scheduler,
        clipSkip,
        count,
        randomSeed,
        seed,
        upEnabled,
        upMode,
        upModel: upModel || undefined,
        upFactor,
        upDenoise,
        upTile,
      };
      // Fire and forget
      saveLastGenerate(snapshot).catch(() => {});
    }, 650); // debounce a bit so we don't spam on every keystroke
  }

  // Restore from lastGenerate the first time we have both models and settings
  useEffect(() => {
    if (restoreDoneRef.current) return;
    if (!models || settings === undefined || settings === null) return;

    const lg = settings.lastGenerate;
    if (!lg) {
      restoreDoneRef.current = true;
      return;
    }
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
      restoredPrimaryModelRef.current = lg.checkpoint;
      didRestore = true;
    }
    if (lg.diffusionModel && existsIn("diffusion_models", lg.diffusionModel)) {
      setDiffusionModel(lg.diffusionModel);
      restoredPrimaryModelRef.current = lg.diffusionModel;
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
    if (Array.isArray(lg.loras)) {
      const valid = lg.loras.filter((l: LoraEntry) => l?.name && existsIn("loras", l.name)).map(normalizeLora);
      setLoras(valid);
      didRestore = true;
    }
    if (Array.isArray(lg.embeddings)) {
      const valid = lg.embeddings.filter((name: string) => existsIn("embeddings", name));
      setEmbeddings(valid);
      didRestore = true;
    }

    if (typeof lg.prompt === "string") { setPrompt(lg.prompt); didRestore = true; }
    if (typeof lg.negative === "string") { setNegative(lg.negative); didRestore = true; }
    if (typeof lg.ideogramMagic === "boolean") { setIdeogramMagic(lg.ideogramMagic); didRestore = true; }
    if (lg.ideogramMagicMode === "local" || lg.ideogramMagicMode === "api" || lg.ideogramMagicMode === "raw") {
      setIdeogramMagicMode(lg.ideogramMagicMode);
      didRestore = true;
    }
    if (lg.ideogramSpeedMode === "max" || lg.ideogramSpeedMode === "high" || lg.ideogramSpeedMode === "fast") {
      setIdeogramSpeedMode(lg.ideogramSpeedMode);
      didRestore = true;
    }
    if (typeof lg.preset === "string" && DIM_PRESETS.some(([name]) => name === lg.preset)) {
      setPreset(lg.preset);
      didRestore = true;
    }
    if (typeof lg.w === "number" && lg.w > 0) { setW(lg.w); didRestore = true; }
    if (typeof lg.h === "number" && lg.h > 0) { setH(lg.h); didRestore = true; }
    if (typeof lg.steps === "number") { setSteps(lg.steps); didRestore = true; }
    if (typeof lg.cfg === "number") { setCfg(lg.cfg); didRestore = true; }
    if (lg.sampler && SAMPLERS.includes(lg.sampler)) { setSampler(lg.sampler); didRestore = true; }
    if (lg.scheduler && SCHEDULERS.includes(lg.scheduler)) { setScheduler(lg.scheduler); didRestore = true; }
    if (typeof lg.clipSkip === "number") { setClipSkip(lg.clipSkip); didRestore = true; }
    if (typeof lg.count === "number" && lg.count > 0) { setCount(lg.count); didRestore = true; }
    if (typeof lg.randomSeed === "boolean") { setRandomSeed(lg.randomSeed); didRestore = true; }
    if (typeof lg.seed === "number") { setSeed(lg.seed); didRestore = true; }
    if (typeof lg.upEnabled === "boolean") { setUpEnabled(lg.upEnabled); didRestore = true; }
    if (lg.upMode === "esrgan" || lg.upMode === "usdu" || lg.upMode === "iterative") {
      setUpMode(lg.upMode);
      didRestore = true;
    }
    if (lg.upModel && existsIn("upscale_models", lg.upModel)) { setUpModel(lg.upModel); didRestore = true; }
    if (typeof lg.upFactor === "number") { setUpFactor(lg.upFactor); didRestore = true; }
    if (typeof lg.upDenoise === "number") { setUpDenoise(lg.upDenoise); didRestore = true; }
    if (typeof lg.upTile === "number") { setUpTile(lg.upTile); didRestore = true; }

    // Mark hydration complete even if every saved model disappeared, so fresh
    // user choices in this session start persisting immediately.
    restoreDoneRef.current = true;
    void didRestore;
  }, [models, settings]); // run when both become available

  // Whenever key fields change *after* restore, schedule a save
  useEffect(() => {
    if (!restoreDoneRef.current) return; // don't save the initial auto-pick or the restore itself
    scheduleSave();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    archId, checkpoint, diffusionModel, vae, JSON.stringify(te),
    JSON.stringify(loras), JSON.stringify(embeddings),
    prompt, negative, ideogramMagic, ideogramMagicMode, ideogramSpeedMode, preset, w, h, steps, cfg, sampler, scheduler, clipSkip,
    count, randomSeed, seed,
    upEnabled, upMode, upModel, upFactor, upDenoise, upTile,
  ]);

  // Cleanup any pending save timer on unmount
  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    };
  }, []);

  // Auto-detect arch + tune CFG/steps + auto-fill VAE / text encoders when the
  // user changes the primary model file. Filename-pattern heuristics.
  const primaryModel = profile.mode === "checkpoint" ? checkpoint : diffusionModel;
  const primaryModelLabel = modelLabelForDisplay(primaryModel, profile.label);
  const selectedPrimaryModel = useMemo(() => {
    if (!primaryModel) return null;
    const list = profile.mode === "checkpoint" ? checkpoints : diffusion;
    return list.find((m) => m.name === primaryModel) ?? null;
  }, [primaryModel, profile.mode, checkpoints, diffusion]);
  useEffect(() => {
    if (!primaryModel) return;
    if (restoredPrimaryModelRef.current === primaryModel) return;
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
    if (restoredPrimaryModelRef.current === primaryModel) return;
    const { cfg: dCfg, steps: dSteps } = defaultSamplingForArch(archId, primaryModel);
    setCfg(dCfg);
    setSteps(dSteps);
  }, [archId, primaryModel]);

  useEffect(() => {
    const selectedName = profile.mode === "checkpoint" ? checkpoint : diffusionModel;
    if (!selectedName) return;
    if (appliedRecommendationsForRef.current === selectedName) return;
    const selected = profile.mode === "checkpoint"
      ? checkpoints.find((m) => m.name === selectedName)
      : diffusion.find((m) => m.name === selectedName);
    const rec = selected?.recommended_settings;
    if (!rec) return;
    appliedRecommendationsForRef.current = selectedName;

    if (typeof rec.cfg === "number") setCfg(rec.cfg);
    if (typeof rec.steps === "number") setSteps(rec.steps);
    if (rec.sampler && SAMPLERS.includes(rec.sampler)) setSampler(rec.sampler);
    if (rec.scheduler && SCHEDULERS.includes(rec.scheduler)) setScheduler(rec.scheduler);
    if (typeof rec.width === "number" && typeof rec.height === "number") {
      const presetName = DIM_PRESETS.find(([, pw, ph]) => pw === rec.width && ph === rec.height)?.[0] ?? "Custom";
      setPreset(presetName);
      setW(rec.width);
      setH(rec.height);
    }
  }, [profile.mode, checkpoint, diffusionModel, checkpoints, diffusion]);

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
    const allowed = new Set(filteredLoras.map((l) => l.name));
    setLoraToAdd((cur) => (cur && !allowed.has(cur) ? "" : cur));
    setLoras((cur) => {
      if (cur.every((l) => allowed.has(l.name))) return cur;
      return cur.filter((l) => allowed.has(l.name));
    });
  }, [filteredLoras]);

  useEffect(() => {
    if (upEnabled && !upModel && upscalers.length) setUpModel(upscalers[0].name);
  }, [upEnabled, upscalers, upModel]);

  const dimsLocked = useMemo(() => preset !== "Custom", [preset]);

  function applyPreset(name: string) {
    setPreset(name);
    const found = DIM_PRESETS.find((p) => p[0] === name);
    if (found && found[0] !== "Custom") { setW(found[1]); setH(found[2]); }
  }

  function selectArch(nextArch: string) {
    if (nextArch === archId) return;
    restoredPrimaryModelRef.current = null;
    appliedRecommendationsForRef.current = null;
    setArchId(nextArch);
    const nextProfile = ARCH_PROFILES.find((p) => p.id === nextArch) ?? ARCH_PROFILES[0];
    if (nextProfile.mode === "checkpoint") {
      setDiffusionModel("");
      setCheckpoint("");
      setVae("");
    } else {
      setCheckpoint("");
      setDiffusionModel("");
      if (nextArch === "ideogram4") setVae("");
    }
  }

  function addLora(name: string) {
    if (!name) return;
    if (loras.some((l) => l.name === name)) return;
    setLoras((cur) => [...cur, { name, weight: 1.0, model_weight: 1.0 }]);
    setLoraToAdd("");
  }
  function setLoraName(index: number, name: string) {
    if (!name) {
      setLoras((cur) => cur.filter((_, i) => i !== index));
      return;
    }
    setLoras((cur) => {
      if (cur.some((l, i) => i !== index && l.name === name)) return cur;
      return cur.map((l, i) => (i === index ? { ...normalizeLora(l), name } : l));
    });
  }
  function setLoraModelWeight(name: string, weight: number) {
    setLoras((cur) => cur.map((l) => (
      l.name === name ? { ...l, weight, model_weight: weight } : l
    )));
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
  const [progressMsg, setProgressMsg] = useState<string>("");
  const [step, setStep] = useState({ step: 0, total: 0 });
  const [imgIdx, setImgIdx] = useState({ i: 0, total: 0 });
  const [previews, setPreviews] = useState<Preview[]>([]);
  const [highlightBatchKey, setHighlightBatchKey] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // ---- selection + viewer ----
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [viewerIdx, setViewerIdx] = useState<number | null>(null);
  const [viewerUrl, setViewerUrl] = useState<string>("");
  const [deleting, setDeleting] = useState(false);
  const [imageMenu, setImageMenu] = useState<ImageContextMenu | null>(null);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    getLatestImages(GALLERY_LIMIT)
      .then(async (data) => {
        if (cancelled || previews.length > 0) return;
        const items = await Promise.all(data.items.map(async (item, idx) => ({
          idx,
          src: await outputFileUrl(item.rel_path),
          seed: Number(item.seed ?? 0),
          filename: item.filename,
          rel_path: item.rel_path,
          model_label: item.model_label,
          batch_key: batchKeyForOutput(item.rel_path, item.filename),
        } satisfies Preview)));
        if (!cancelled && items.length) {
          setPreviews(items);
          setHighlightBatchKey(items[0]?.batch_key ?? null);
        }
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [active, previews.length]);

  // Resolve the sidecar URL for the active viewer image whenever its index changes.
  useEffect(() => {
    if (viewerIdx === null) { setViewerUrl(""); return; }
    const p = previews[viewerIdx];
    if (!p) return;
    let cancelled = false;
    outputFileUrl(p.rel_path).then((url) => { if (!cancelled) setViewerUrl(url); });
    return () => { cancelled = true; };
  }, [viewerIdx, previews]);

  useEffect(() => {
    if (!imageMenu) return;
    const close = () => setImageMenu(null);
    window.addEventListener("click", close);
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", close);
    };
  }, [imageMenu]);

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
    setHighlightBatchKey(null);
    setSelected(new Set());
  }

  function applyGenerationSettings(p: GenerateParams) {
    restoredPrimaryModelRef.current = null;
    appliedRecommendationsForRef.current = null;
    const nextArch = p.arch || "sdxl";
    const nextProfile = ARCH_PROFILES.find((profile) => profile.id === nextArch);
    if (nextProfile) {
      setArchId(nextArch);
      if (nextProfile.mode === "checkpoint") setDiffusionModel("");
      if (nextProfile.mode === "components") setCheckpoint("");
    }
    if (p.checkpoint && checkpoints.some((m) => m.name === p.checkpoint)) setCheckpoint(p.checkpoint);
    if (p.diffusion_model && diffusion.some((m) => m.name === p.diffusion_model)) setDiffusionModel(p.diffusion_model);
    if (p.vae && vaes.some((m) => m.name === p.vae)) setVae(p.vae);
    if (Array.isArray(p.text_encoders)) {
      const valid = p.text_encoders.filter((name) => textEncoders.some((m) => m.name === name));
      setTe(valid);
    }
    if (Array.isArray(p.loras)) {
      const valid = p.loras.filter((l) => l?.name && lorasAvail.some((m) => m.name === l.name)).map(normalizeLora);
      setLoras(valid);
    }
    if (Array.isArray(p.embeddings)) {
      const valid = p.embeddings.filter((name) => embeddingsAvail.some((m) => m.name === name));
      setEmbeddings(valid);
    }
    setPrompt(p.prompt || "");
    setNegative(p.negative || "");
    if (typeof p.ideogram_magic === "boolean") setIdeogramMagic(p.ideogram_magic);
    if (p.ideogram_magic_mode === "local" || p.ideogram_magic_mode === "api" || p.ideogram_magic_mode === "raw") {
      setIdeogramMagicMode(p.ideogram_magic_mode);
    }
    if (typeof p.width === "number" && typeof p.height === "number") {
      const presetName = DIM_PRESETS.find(([, pw, ph]) => pw === p.width && ph === p.height)?.[0] ?? "Custom";
      setPreset(presetName);
      setW(p.width);
      setH(p.height);
    }
    if (typeof p.steps === "number") setSteps(p.steps);
    if (typeof p.cfg === "number") setCfg(p.cfg);
    if (p.sampler && SAMPLERS.includes(p.sampler)) setSampler(p.sampler);
    if (p.scheduler && SCHEDULERS.includes(p.scheduler)) setScheduler(p.scheduler);
    if (typeof p.clip_skip === "number") setClipSkip(p.clip_skip || 0);
    setCount(1);
    if (typeof p.seed === "number") {
      setRandomSeed(false);
      setSeed(p.seed);
    }
    setUpEnabled(!!p.upscale_enabled);
    if (p.upscale_mode === "esrgan" || p.upscale_mode === "usdu" || p.upscale_mode === "iterative") setUpMode(p.upscale_mode);
    if (p.upscale_model && upscalers.some((m) => m.name === p.upscale_model)) setUpModel(p.upscale_model);
    if (typeof p.upscale_factor === "number") setUpFactor(p.upscale_factor);
    if (typeof p.upscale_denoise === "number") setUpDenoise(p.upscale_denoise);
    if (typeof p.upscale_tile_size === "number") setUpTile(p.upscale_tile_size);
  }

  async function reuseImageSettings(preview: Preview) {
    setImageMenu(null);
    if (!preview.rel_path) return;
    try {
      const settings = await getOutputSettings(preview.rel_path);
      applyGenerationSettings(settings);
    } catch {
      setErrorMsg(`No reusable settings found for ${preview.filename}. Generate new images once after this update to create settings metadata.`);
    }
  }

  function openImageMenu(e: React.MouseEvent, preview: Preview) {
    e.preventDefault();
    e.stopPropagation();
    setImageMenu({ x: e.clientX, y: e.clientY, preview });
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

  async function applyIdeogramMagicPrompt() {
    if (archId !== "ideogram4") return;
    setErrorMsg(null);
    setMagicBusy(true);
    try {
      const mode = ideogramMagic ? ideogramMagicMode : "raw";
      const r = await ideogramMagicPrompt({
        prompt: prompt.trim() || DEFAULT_PROMPT,
        negative: negative.trim(),
        width: w,
        height: h,
        mode,
      });
      setPrompt(r.pretty_prompt || r.prompt);
      if (mode === "raw") setIdeogramMagic(false);
    } catch (e: any) {
      setErrorMsg(`Magic prompt failed: ${e?.message ?? e}`);
    } finally {
      setMagicBusy(false);
    }
  }

  async function appendResultOutputs(result: any) {
    const outputs = Array.isArray(result?.outputs) ? result.outputs : [];
    if (!outputs.length) return;
    const recovered = await Promise.all(outputs.map(async (out: any, idx: number) => {
      const relPath = String(out?.rel_path ?? "");
      const filename = String(out?.filename ?? relPath.split("/").pop() ?? `image-${idx}.png`);
      return {
        idx: Number(out?.image_index ?? idx),
        src: relPath ? await outputFileUrl(relPath) : undefined,
        seed: Number(out?.seed ?? 0),
        filename,
        rel_path: relPath,
        model_label: primaryModelLabel,
        batch_key: batchKeyForOutput(relPath, filename),
      } satisfies Preview;
    }));
    setHighlightBatchKey(recovered[0]?.batch_key ?? null);
    setPreviews((cur) => {
      const seen = new Set(cur.map((p) => p.rel_path || p.filename));
      const next = recovered.filter((p) => !seen.has(p.rel_path || p.filename));
      return next.length ? [...next, ...cur].slice(0, GALLERY_LIMIT) : cur.slice(0, GALLERY_LIMIT);
    });
  }

  async function submit() {
    setErrorMsg(null);
    // Don't clear previews — let them accumulate across runs so the user can
    // compare. Use "Clear gallery" to wipe.
    setStep({ step: 0, total: steps });
    setImgIdx({ i: 0, total: count });
    setStatus("submitting");
    setProgressMsg("Submitting");

    const params: GenerateParams = {
      arch: archId,
      checkpoint:      profile.mode === "checkpoint" ? (checkpoint || null) : null,
      diffusion_model: profile.mode === "components" ? (diffusionModel || null) : null,
      vae: profile.mode === "checkpoint" || archId === "ideogram4" ? null : (vae || null),
      text_encoders: archId === "ideogram4" ? [] : te.filter(Boolean),
      loras,
      embeddings,
      prompt:   prompt.trim()   || DEFAULT_PROMPT,
      negative: negative.trim() || DEFAULT_NEGATIVE,
      ideogram_magic: archId === "ideogram4" ? ideogramMagic : false,
      ideogram_magic_mode: archId === "ideogram4" ? (ideogramMagic ? ideogramMagicMode : "raw") : "raw",
      ideogram_speed_mode: archId === "ideogram4" ? ideogramSpeedMode : "high",
      width: w,
      height: h,
      steps,
      cfg,
      sampler,
      scheduler,
      clip_skip: clipSkip > 0 ? clipSkip : null,
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
        if (evt.type === "snapshot") {
          setStatus(evt.status);
          if (evt.progress?.message) setProgressMsg(evt.progress.message);
          if (evt.status === "succeeded" && evt.result) void appendResultOutputs(evt.result);
        }
        if (evt.type === "status") {
          setStatus(evt.status);
          if (evt.message) setProgressMsg(evt.message);
          if (evt.error) setErrorMsg(evt.error);
          if (evt.status === "succeeded" && evt.result) void appendResultOutputs(evt.result);
        }
        if (evt.type === "progress") {
          setStep({ step: evt.step, total: evt.total_steps });
          setImgIdx({ i: evt.image_index, total: evt.total_images });
          if (evt.message) setProgressMsg(evt.message);
        }
        if (evt.type === "image") {
          const filename = evt.filename;
          const relPath = evt.rel_path ?? "";
          const preview = {
            idx: evt.image_index, b64: evt.preview_b64, seed: evt.seed,
            filename, rel_path: relPath,
            model_label: primaryModelLabel,
            batch_key: batchKeyForOutput(relPath, filename),
          } satisfies Preview;
          const key = preview.rel_path || preview.filename;
          setHighlightBatchKey(preview.batch_key ?? null);
          setPreviews((cur) => [preview, ...cur.filter((p) => (p.rel_path || p.filename) !== key)].slice(0, GALLERY_LIMIT));
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

  const loraOptions = filteredLoras.filter((l) => !loras.some((sel) => sel.name === l.name));
  const embOptions  = embeddingsAvail.filter((e) => !embeddings.includes(e.name));
  const loraRowOptions = (index: number) =>
    filteredLoras.filter((opt) => opt.name === loras[index]?.name || !loras.some((sel, i) => i !== index && sel.name === opt.name));

  return (
    <>
        {/* Center pane */}
      <main className={"pane center gen-center" + (active ? "" : " tab-hidden")}>
        <div className="prompt-stack">
          <div className="prompt-label-row">
            <label className="lbl">Prompt</label>
            <button
              className="mini"
              onClick={() => setPrompt("")}
              disabled={!prompt}
              title="Clear prompt"
            >
              Clear prompt
            </button>
          </div>
          <textarea className="prompt-main" rows={12} value={prompt} onChange={(e) => setPrompt(e.target.value)}
            placeholder={DEFAULT_PROMPT} />

          {/* Deterministic prompt builder — one scene, rendered into the
              current model's native dialect (Ideogram JSON / FLUX prose / SDXL tags). */}
          <div className="prompt-builder">
            <button className="mini pb-toggle" onClick={() => setPbOpen((o) => !o)}>
              {pbOpen ? "▾" : "▸"} Prompt Builder
              <span className="muted small" style={{ marginLeft: 8 }}>
                {archId === "ideogram4" ? "→ Ideogram JSON"
                  : archId === "sdxl" || archId === "illustrious" ? "→ SDXL tags"
                  : "→ natural-language prose"}
              </span>
            </button>
            {pbOpen && pbOptions && (
              <div className="pb-body">
                <div className="pb-grid">
                  {([["Style", pbStyle, setPbStyle, pbOptions.style],
                     ["Lighting", pbLighting, setPbLighting, pbOptions.lighting],
                     ["Camera", pbCamera, setPbCamera, pbOptions.camera],
                     ["Mood", pbMood, setPbMood, pbOptions.mood]] as const).map(
                    ([label, val, set, opts]) => (
                      <label key={label} className="pb-field">
                        <span>{label}</span>
                        <select value={val} onChange={(e) => (set as (v: string) => void)(e.target.value)}>
                          {opts.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </select>
                      </label>
                    ))}
                </div>
                <div className="pb-actions">
                  <span className="muted small">
                    Type your subject in the prompt box above, pick options, then build.
                    Quote any in-image text, e.g. &quot;OPEN&quot;.
                  </span>
                  <button className="primary" onClick={buildPrompt} disabled={pbBusy}>
                    {pbBusy ? "Building…" : "Build prompt"}
                  </button>
                </div>
              </div>
            )}
          </div>

          {archId === "ideogram4" && (
            <div className="magic-row">
              <label className="checkline">
                <input
                  type="checkbox"
                  checked={ideogramMagic}
                  onChange={(e) => setIdeogramMagic(e.target.checked)}
                />
                Magic prompt
              </label>
              <select
                value={ideogramMagicMode}
                onChange={(e) => setIdeogramMagicMode(e.target.value as "local" | "api" | "raw")}
                disabled={!ideogramMagic}
              >
                <option value="local">Kraken local JSON</option>
                <option value="api">Ideogram API</option>
                <option value="raw">Raw prompt</option>
              </select>
              <button onClick={applyIdeogramMagicPrompt} disabled={magicBusy || !prompt.trim()}>
                {magicBusy ? "Working..." : "Apply Magic"}
              </button>
            </div>
          )}
          {archId === "ideogram4" && (!ideogramMagic || ideogramMagicMode === "raw") && !prompt.trim().startsWith("{") && (
            <div className="model-warning">
              Raw passthrough is only used for valid Ideogram JSON. Plain text is converted to Kraken local JSON before generation.
            </div>
          )}
          <label className="lbl">Negative</label>
          <textarea className="prompt-negative" rows={5} value={negative} onChange={(e) => setNegative(e.target.value)}
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
              {progressMsg || status} · {imgIdx.i + 1}/{Math.max(imgIdx.total, 1)} · step {step.step}/{step.total || "?"}
            </span>
          )}
        </div>

        {busy && (
          <div className={"progress-bar" + (overallPct <= 0 ? " indeterminate" : "")}>
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
            const isLastBatch = !!highlightBatchKey && p.batch_key === highlightBatchKey;
            return (
              <figure
                className={"thumb" + (isSelected ? " selected" : "") + (isLastBatch ? " last-batch" : "")}
                key={p.rel_path || p.filename || `${p.idx}-${p.seed}`}
                onClick={() => p.rel_path && setViewerIdx(i)}
                onContextMenu={(e) => openImageMenu(e, p)}
              >
                {isLastBatch && <span className="batch-badge">Last batch</span>}
                <label className="thumb-check" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={(e) => toggleSelect(p.filename, e)}
                  />
                </label>
                <img src={p.src ?? `data:image/jpeg;base64,${p.b64 ?? ""}`} alt={p.filename} draggable={false} />
                <figcaption className="muted small" title={p.filename}>
                  <span>{p.model_label || p.filename}</span>
                  <span>seed {p.seed || "?"}</span>
                </figcaption>
              </figure>
            );
          })}
          {!previews.length && !busy && (
            <div className="placeholder small">Generated images appear here</div>
          )}
        </div>

        {imageMenu && (
          <div
            className="image-context-menu"
            style={{ left: imageMenu.x, top: imageMenu.y }}
            onClick={(e) => e.stopPropagation()}
            onContextMenu={(e) => e.preventDefault()}
          >
            <button onClick={() => reuseImageSettings(imageMenu.preview)}>Reuse generation settings</button>
          </div>
        )}

        {viewerIdx !== null && viewerUrl && previews[viewerIdx] && (
          <ImageViewer
            url={viewerUrl}
            caption={`${previews[viewerIdx].model_label || previews[viewerIdx].filename} · seed ${previews[viewerIdx].seed || "?"}`}
            onClose={() => setViewerIdx(null)}
            onPrev={viewerIdx > 0 ? () => setViewerIdx(viewerIdx - 1) : undefined}
            onNext={viewerIdx < previews.length - 1 ? () => setViewerIdx(viewerIdx + 1) : undefined}
          />
        )}
      </main>

      {/* Right pane — scrollable */}
      <aside className={"pane right-params" + (active ? "" : " tab-hidden")}>
        <div className="section-title">Architecture</div>
        <div className="field">
          <select value={archId} onChange={(e) => selectArch(e.target.value)}>
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
            <select value={checkpoint} onChange={(e) => { appliedRecommendationsForRef.current = null; setCheckpoint(e.target.value); }}>
              {filteredCheckpoints.length === 0 && <option value="">— none scanned for {profile.label} —</option>}
              {filteredCheckpoints.map((c) => (
                <option key={c.name} value={c.name} disabled={!modelIsSelectable(c)}>
                  {modelOptionLabel(c)}
                </option>
              ))}
            </select>
          </div>
        ) : (
          <div className="field">
            <label>Diffusion model</label>
            <select value={diffusionModel} onChange={(e) => { appliedRecommendationsForRef.current = null; setDiffusionModel(e.target.value); }}>
              {filteredDiffusion.length === 0 && <option value="">— none scanned for {profile.label} —</option>}
              {filteredDiffusion.map((c) => (
                <option key={c.name} value={c.name} disabled={!modelIsSelectable(c)}>
                  {modelOptionLabel(c)}
                </option>
              ))}
            </select>
          </div>
        )}
        {selectedPrimaryModel?.warning && (
          <div className={selectedPrimaryModel.experimental ? "model-warning" : "model-note"}>
            {selectedPrimaryModel.warning}
          </div>
        )}

        <div className="field">
          <label>VAE {profile.mode === "checkpoint" || archId === "ideogram4" ? "(bundled)" : ""}</label>
          <select value={profile.mode === "checkpoint" || archId === "ideogram4" ? "" : vae} onChange={(e) => setVae(e.target.value)} disabled={profile.mode === "checkpoint" || archId === "ideogram4"}>
            <option value="">{profile.mode === "checkpoint" || archId === "ideogram4" ? "(bundled)" : "— pick one —"}</option>
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
        <div className="field">
          <label title="SDXL/Illustrious only. 0 uses the pipeline default.">CLIP skip</label>
          <input
            type="number"
            min={0}
            max={12}
            step={1}
            value={clipSkip}
            onChange={(e) => setClipSkip(Math.max(0, parseInt(e.target.value) || 0))}
            disabled={profile.mode !== "checkpoint"}
          />
        </div>
        <div className="muted small" style={{ marginBottom: 6 }}>0 = default. Set 1/2 only when the LoRA or model recommends it.</div>

        {loras.map((l, i) => (
          <div className="lora-row lora-slot" key={`${l.name}-${i}`}>
            <select value={l.name} onChange={(e) => setLoraName(i, e.target.value)} title={l.name}>
              <option value="">— remove LoRA —</option>
              {loraRowOptions(i).map((opt) => <option key={opt.name} value={opt.name}>{opt.name}</option>)}
            </select>
            <label className="lora-strength">
              <span>Model</span>
              <input
                type="number"
                min={-1}
                max={2}
                step={0.05}
                value={loraModelWeight(l)}
                onChange={(e) => setLoraModelWeight(l.name, parseFloat(e.target.value) || 0)}
              />
            </label>
            <button onClick={() => removeLora(l.name)} className="x">✕</button>
          </div>
        ))}

        <div className="lora-add">
          <select
            value={loraToAdd}
            onChange={(e) => addLora(e.target.value)}
          >
            <option value="">{loraOptions.length ? "— add another LoRA —" : `(none scanned for ${profile.label})`}</option>
            {loraOptions.map((l) => <option key={l.name} value={l.name}>{l.name}</option>)}
          </select>
        </div>

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
        {archId === "ideogram4" ? (
          <>
            <div className="preset-row" aria-label="Ideogram quality presets">
              <button
                className={steps <= 12 ? "active" : ""}
                onClick={() => setSteps(12)}
                title="Ideogram V4_TURBO_12. Fastest useful local profile."
              >
                Turbo 12
              </button>
              <button
                className={steps > 12 && steps <= 20 ? "active" : ""}
                onClick={() => setSteps(20)}
                title="Ideogram's V4_DEFAULT_20 preset — a balanced 20-step profile. (Named 'Default' by Ideogram; not the app's default selection.)"
              >
                Standard 20
              </button>
              <button
                className={steps > 20 ? "active" : ""}
                onClick={() => setSteps(48)}
                title="Ideogram V4_QUALITY_48. Best quality, slowest."
              >
                Quality 48
              </button>
            </div>
            <div className="field">
              <label title="Adaptive velocity-cache. High keeps near-identical quality at ~2x speed on a 3090.">
                Speed (cache)
              </label>
              <select
                value={ideogramSpeedMode}
                onChange={(e) => setIdeogramSpeedMode(e.target.value as "max" | "high" | "fast")}
                style={{ width: "100%" }}
              >
                <option value="max">Max — every step (~4.3 min)</option>
                <option value="high">High — recommended (~2.0 min)</option>
                <option value="fast">Fast — draft (~1.6 min)</option>
              </select>
            </div>
            <div className="field-row">
              <div className="field">
                <label>Steps</label>
                <input type="number" min={1} max={150} value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value) || 20)} />
              </div>
              <div className="field">
                <label title="Ideogram profiles use bundled guidance schedules.">Guidance</label>
                <input type="text" value="preset" disabled readOnly />
              </div>
            </div>
          </>
        ) : (
          <>
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
          </>
        )}

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
