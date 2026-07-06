import { useEffect, useMemo, useRef, useState } from "react";
import {
  captureLastVideoFrame,
  cancelJob,
  openJobWS,
  outputFileUrl,
  startGenerateVideo,
  type LoraEntry,
  type ModelListing,
  type VideoGenerateParams,
} from "./api/sidecar";

// WAN 2.2 i2v A14B resolution buckets. 480p and 720p, landscape + portrait.
const DIM_PRESETS: Array<[string, number, number]> = [
  ["832 × 480 (480p landscape)", 832, 480],
  ["480 × 832 (480p portrait)", 480, 832],
  ["1280 × 720 (720p landscape)", 1280, 720],
  ["720 × 1280 (720p portrait)", 720, 1280],
  ["Custom", 0, 0],
];

const DEFAULT_PROMPT =
  "the creature slowly turns its head toward the camera, tentacles drifting in the current, cinematic";
const DEFAULT_NEGATIVE =
  "static, still, blurry, low quality, deformed, jpeg artifacts, watermark";

type VideoMode = "i2v" | "t2v";

const MOTION_PRESETS: Array<{
  id: string;
  label: string;
  prompt: string;
  frames: number;
  fps: number;
  steps: number;
  cfg: number;
}> = [
  {
    id: "cinematic_orbit",
    label: "Cinematic orbit",
    prompt: "slow cinematic camera orbit, subject remains coherent, parallax depth, dramatic light movement, smooth motion",
    frames: 81,
    fps: 16,
    steps: 4,
    cfg: 1.0,
  },
  {
    id: "subtle_idle",
    label: "Subtle idle",
    prompt: "subtle natural motion, gentle breathing, small environmental movement, stable composition, no sudden camera move",
    frames: 81,
    fps: 16,
    steps: 4,
    cfg: 1.0,
  },
  {
    id: "product_turntable",
    label: "Turntable",
    prompt: "clean product turntable rotation, centered subject, smooth studio camera move, consistent lighting, sharp details",
    frames: 81,
    fps: 16,
    steps: 4,
    cfg: 1.0,
  },
  {
    id: "quality_full",
    label: "Full-quality sample",
    prompt: "high quality cinematic scene, coherent subject motion, detailed environment, smooth temporal consistency",
    frames: 81,
    fps: 16,
    steps: 28,
    cfg: 3.0,
  },
];

function isWanDiffusion(name: string): boolean {
  return name.toLowerCase().includes("wan");
}

function isWanMode(name: string, mode: VideoMode): boolean {
  const n = name.toLowerCase();
  if (mode === "t2v") return n.includes("t2v") || n.includes("text");
  return n.includes("i2v") || (!n.includes("t2v") && !n.includes("text"));
}

function pickFirst(list: { name: string }[], ...patterns: string[]): string | undefined {
  for (const pat of patterns) {
    const p = pat.toLowerCase();
    const hit = list.find((m) => m.name.toLowerCase().includes(p));
    if (hit) return hit.name;
  }
  return undefined;
}

function uniqueByName<T extends { name: string }>(items: T[]): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (seen.has(item.name)) return false;
    seen.add(item.name);
    return true;
  });
}

type VideoResult = {
  rel_path: string;
  filename: string;
  seed: number;
  preview_b64: string;
};

export default function Video({
  active,
  models,
}: {
  active: boolean;
  models: ModelListing | null;
}) {
  const diffusion = models?.categories.diffusion_models ?? [];
  const vaes = models?.categories.vae ?? [];
  const textEncoders = models?.categories.text_encoders ?? [];
  const lorasAvail = models?.categories.loras ?? [];

  const wanDiffusion = useMemo(
    () => diffusion.filter((m) => isWanDiffusion(m.name)),
    [diffusion]
  );

  const [mode, setMode] = useState<VideoMode>("i2v");
  const modeDiffusion = useMemo(
    () => wanDiffusion.filter((m) => isWanMode(m.name, mode)),
    [wanDiffusion, mode]
  );
  const loraPool = useMemo(() => {
    const modeKey = mode === "t2v" ? "t2v" : "i2v";
    const otherModeKey = mode === "t2v" ? "i2v" : "t2v";
    const modeLoras = lorasAvail.filter((l) => l.name.toLowerCase().includes(modeKey));
    const genericWanLoras = lorasAvail.filter((l) => {
      const n = l.name.toLowerCase();
      return n.includes("wan") && !n.includes(otherModeKey);
    });
    return uniqueByName([...modeLoras, ...genericWanLoras, ...lorasAvail]);
  }, [lorasAvail, mode]);

  // ---- model selections ----
  const [highExpert, setHighExpert] = useState<string>("");
  const [lowExpert, setLowExpert] = useState<string>("");
  const [vae, setVae] = useState<string>("");
  const [te, setTe] = useState<string>("");

  // ---- lightx2v 4-step distill LoRA ----
  const [useLightx, setUseLightx] = useState(true);
  const [highLora, setHighLora] = useState("");
  const [lowLora, setLowLora] = useState("");
  const [highLoraWeight, setHighLoraWeight] = useState(1.0);
  const [lowLoraWeight, setLowLoraWeight] = useState(1.0);

  // ---- input image (the i2v first frame) ----
  const [inputImage, setInputImage] = useState<string | null>(null); // data URL
  const [inputPreview, setInputPreview] = useState<string | null>(null);
  const [inputName, setInputName] = useState<string>("");
  const fileRef = useRef<HTMLInputElement | null>(null);

  // ---- prompt + params ----
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [preset, setPreset] = useState("832 × 480 (480p landscape)");
  const [w, setW] = useState(832);
  const [h, setH] = useState(480);
  const [numFrames, setNumFrames] = useState(81);
  const [fps, setFps] = useState(16);
  const [steps, setSteps] = useState(4);
  const [cfg, setCfg] = useState(1.0);
  const [randomSeed, setRandomSeed] = useState(true);
  const [seed, setSeed] = useState<number>(0);

  // Auto-pick sensible defaults when the model list arrives.
  useEffect(() => {
    if (!highExpert || !modeDiffusion.some((m) => m.name === highExpert)) {
      setHighExpert(pickFirst(modeDiffusion, "high_noise", "high") ?? modeDiffusion[0]?.name ?? "");
    }
    if (!lowExpert || !modeDiffusion.some((m) => m.name === lowExpert)) {
      setLowExpert(pickFirst(modeDiffusion, "low_noise", "low") ?? modeDiffusion[1]?.name ?? modeDiffusion[0]?.name ?? "");
    }
    if (!vae || !vaes.some((m) => m.name === vae)) {
      setVae(pickFirst(vaes, "wan_2.1_vae", "wan") ?? "");
    }
    if (!te || !textEncoders.some((m) => m.name === te)) {
      setTe(pickFirst(textEncoders, "umt5_xxl", "umt5") ?? "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models, mode, modeDiffusion]);

  useEffect(() => {
    const hi = pickFirst(loraPool, "highnoise_if", "highnoise", "high-noise", "high_noise", "lightx2v_4steps_lora_v1_high_noise", "lightx2v") ?? "";
    const lo = pickFirst(loraPool, "lownoise_if", "lownoise", "low-noise", "low_noise", "lightx2v_4steps_lora_v1_low_noise") ?? "";
    if (!highLora || !loraPool.some((m) => m.name === highLora)) setHighLora(hi);
    if (!lowLora || !loraPool.some((m) => m.name === lowLora)) setLowLora(lo);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loraPool]);

  const dimsLocked = useMemo(() => preset !== "Custom", [preset]);

  function applyPreset(name: string) {
    setPreset(name);
    const found = DIM_PRESETS.find((p) => p[0] === name);
    if (found && found[0] !== "Custom") { setW(found[1]); setH(found[2]); }
  }

  function applyMotionPreset(id: string) {
    const found = MOTION_PRESETS.find((p) => p.id === id);
    if (!found) return;
    setPrompt(found.prompt);
    setNumFrames(found.frames);
    setFps(found.fps);
    setSteps(found.steps);
    setCfg(found.cfg);
    if (found.steps <= 8) setUseLightx(true);
  }

  function onPickFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      const value = typeof reader.result === "string" ? reader.result : null;
      setInputImage(value);
      setInputPreview(value);
      setInputName(f.name);
    };
    reader.readAsDataURL(f);
  }

  function clearInput() {
    setInputImage(null);
    setInputPreview(null);
    setInputName("");
    if (fileRef.current) fileRef.current.value = "";
  }

  async function useLastFrame(r: VideoResult) {
    setErrorMsg(null);
    try {
      const captured = await captureLastVideoFrame(r.rel_path);
      const url = await outputFileUrl(captured.item.rel_path);
      setInputImage(captured.item.rel_path);
      setInputPreview(url);
      setInputName(captured.item.filename);
      setMode("i2v");
      setErrorMsg(null);
    } catch (e: any) {
      setErrorMsg(`last-frame capture failed: ${e?.message ?? e}`);
    }
  }

  // --- runtime ---
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [step, setStep] = useState({ step: 0, total: 0 });
  const [results, setResults] = useState<VideoResult[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Resolve sidecar URLs for each result's mp4 (outputFileUrl is async).
  const [videoUrls, setVideoUrls] = useState<Record<string, string>>({});
  useEffect(() => {
    let cancelled = false;
    results.forEach((r) => {
      if (videoUrls[r.rel_path]) return;
      outputFileUrl(r.rel_path).then((url) => {
        if (!cancelled) setVideoUrls((cur) => ({ ...cur, [r.rel_path]: url }));
      });
    });
    return () => { cancelled = true; };
  }, [results]);

  const busy = status === "queued" || status === "running" || status === "submitting";
  const pct = step.total > 0 ? Math.floor((step.step / step.total) * 100) : 0;

  const ready = !!highExpert && !!lowExpert && !!vae && !!te && (mode === "t2v" || !!inputImage);

  function buildLoras(): LoraEntry[] {
    if (!useLightx) return [];
    const out: LoraEntry[] = [];
    if (highLora) out.push({ name: highLora, weight: highLoraWeight, model_weight: highLoraWeight });
    if (lowLora && lowLora !== highLora) out.push({ name: lowLora, weight: lowLoraWeight, model_weight: lowLoraWeight });
    return out;
  }

  async function submit() {
    setErrorMsg(null);
    if (mode === "i2v" && !inputImage) { setErrorMsg("Pick an input image (the first frame) first."); return; }
    setStep({ step: 0, total: steps });
    setStatus("submitting");

    const params: VideoGenerateParams = {
      arch: "wan",
      mode,
      diffusion_model: highExpert || null,
      diffusion_model_2: lowExpert || null,
      vae: vae || null,
      text_encoders: te ? [te] : [],
      loras: buildLoras(),
      prompt: prompt.trim() || DEFAULT_PROMPT,
      negative: negative.trim() || DEFAULT_NEGATIVE,
      input_image: mode === "i2v" ? inputImage : null,
      width: w,
      height: h,
      num_frames: numFrames,
      fps,
      steps,
      cfg,
      seed: randomSeed ? null : seed,
    };

    try {
      const r = await startGenerateVideo(params);
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
        }
        if (evt.type === "video") {
          setResults((cur) => [
            { rel_path: evt.rel_path ?? "", filename: evt.filename, seed: evt.seed, preview_b64: evt.preview_b64 ?? "" },
            ...cur,
          ]);
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

  return (
    <>
      {/* Center pane */}
      <main className={"pane center gen-center" + (active ? "" : " tab-hidden")}>
        <div className="section-title">WAN video mode</div>
        <div className="field-row">
          <button className={mode === "i2v" ? "primary" : ""} onClick={() => setMode("i2v")} disabled={busy}>Image to video</button>
          <button className={mode === "t2v" ? "primary" : ""} onClick={() => setMode("t2v")} disabled={busy}>Text to video</button>
        </div>

        {mode === "i2v" && (
          <>
            <div className="section-title">Input frame</div>
            <div className="field">
              <input ref={fileRef} type="file" accept="image/*" onChange={onPickFile} />
              {inputPreview && (
                <div style={{ marginTop: 8, position: "relative", display: "inline-block" }}>
                  <img
                    src={inputPreview}
                    alt={inputName}
                    style={{ maxWidth: "100%", maxHeight: 240, borderRadius: 6, display: "block" }}
                  />
                  <div className="muted small" style={{ marginTop: 4 }}>
                    {inputName} <button onClick={clearInput} className="x" style={{ marginLeft: 8 }}>✕</button>
                  </div>
                </div>
              )}
              {!inputImage && (
                <div className="muted small" style={{ marginTop: 4 }}>
                  WAN i2v animates a still image — pick the first frame.
                </div>
              )}
            </div>
          </>
        )}

        <div className="field">
          <label className="lbl">Motion preset</label>
          <select defaultValue="" onChange={(e) => applyMotionPreset(e.target.value)}>
            <option value="">Custom motion</option>
            {MOTION_PRESETS.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>
        </div>

        <div className="prompt-stack">
          <label className="lbl">Prompt (motion)</label>
          <textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)}
            placeholder={DEFAULT_PROMPT} />
          <label className="lbl">Negative</label>
          <textarea rows={2} value={negative} onChange={(e) => setNegative(e.target.value)}
            placeholder={DEFAULT_NEGATIVE} />
        </div>

        <div className="action-row">
          <button className="primary big" onClick={submit} disabled={busy || !ready}>
            {busy ? "Generating…" : "Generate video"}
          </button>
          {busy && <button onClick={cancel}>Cancel</button>}
          <div className="spacer" />
          {status !== "idle" && (
            <span className="muted small">{status} · step {step.step}/{step.total || "?"}</span>
          )}
        </div>

        {busy && (
          <div className="progress-bar">
            <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
          </div>
        )}

        {!ready && !busy && (
          <div className="muted small">
            Need: high-noise expert, low-noise expert, WAN VAE, UMT5 encoder{mode === "i2v" ? ", and an input image." : "."}
          </div>
        )}

        {errorMsg && <div className="err">{errorMsg}</div>}

        <div className="thumb-grid" style={{ marginTop: 12 }}>
          {results.map((r) => (
            <figure className="thumb" key={r.rel_path || r.filename} style={{ width: "100%", maxWidth: 520 }}>
              {videoUrls[r.rel_path] ? (
                <video
                  src={videoUrls[r.rel_path]}
                  controls
                  loop
                  poster={r.preview_b64 ? `data:image/jpeg;base64,${r.preview_b64}` : undefined}
                  style={{ width: "100%", borderRadius: 6, display: "block" }}
                />
              ) : (
                <div className="placeholder small">loading…</div>
              )}
              <figcaption className="muted small">{r.filename} · seed {r.seed}</figcaption>
              <button onClick={() => useLastFrame(r)} disabled={busy || !r.rel_path}>
                Use last frame
              </button>
            </figure>
          ))}
          {!results.length && !busy && (
            <div className="placeholder small">Generated videos appear here</div>
          )}
        </div>
      </main>

      {/* Right pane */}
      <aside className={"pane right-params" + (active ? "" : " tab-hidden")}>
        <div className="section-title">WAN 2.2 {mode.toUpperCase()} experts</div>
        <div className="field">
          <label>High-noise expert</label>
          <select value={highExpert} onChange={(e) => setHighExpert(e.target.value)}>
            {modeDiffusion.length === 0 && <option value="">— no {mode.toUpperCase()} WAN models scanned —</option>}
            {modeDiffusion.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Low-noise expert</label>
          <select value={lowExpert} onChange={(e) => setLowExpert(e.target.value)}>
            {modeDiffusion.length === 0 && <option value="">— no {mode.toUpperCase()} WAN models scanned —</option>}
            {modeDiffusion.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
          </select>
        </div>
        <div className="field">
          <label>VAE</label>
          <select value={vae} onChange={(e) => setVae(e.target.value)}>
            <option value="">— pick the WAN VAE —</option>
            {vaes.map((v) => <option key={v.name} value={v.name}>{v.name}</option>)}
          </select>
        </div>
        <div className="field">
          <label>Text encoder (UMT5-XXL)</label>
          <select value={te} onChange={(e) => setTe(e.target.value)}>
            <option value="">— pick UMT5-XXL —</option>
            {textEncoders.map((t) => <option key={t.name} value={t.name}>{t.name}</option>)}
          </select>
        </div>

        <div className="section-title">Speed</div>
        <div className="field">
          <label>
            <input type="checkbox" checked={useLightx}
              onChange={(e) => {
                setUseLightx(e.target.checked);
                if (e.target.checked) { setSteps(4); setCfg(1.0); }
              }} /> WAN speed LoRAs
          </label>
          <div className="muted small" style={{ marginTop: 4 }}>
            Distilled / inference LoRAs for short-step WAN sampling. Uncheck for full 20–40 step sampling.
          </div>
        </div>
        {useLightx && (
          <>
            <div className="field">
              <label>High-noise LoRA</label>
              <select value={highLora} onChange={(e) => setHighLora(e.target.value)}>
                <option value="">— none —</option>
                {loraPool.map((l) => <option key={l.name} value={l.name}>{l.name}</option>)}
              </select>
            </div>
            <div className="field">
              <label>Low-noise LoRA</label>
              <select value={lowLora} onChange={(e) => setLowLora(e.target.value)}>
                <option value="">— none —</option>
                {loraPool.map((l) => <option key={l.name} value={l.name}>{l.name}</option>)}
              </select>
            </div>
            <div className="field">
              <label>High LoRA weight: {highLoraWeight.toFixed(2)}</label>
              <input type="range" min={0} max={1.5} step={0.05} value={highLoraWeight}
                onChange={(e) => setHighLoraWeight(parseFloat(e.target.value) || 0)} />
            </div>
            <div className="field">
              <label>Low LoRA weight: {lowLoraWeight.toFixed(2)}</label>
              <input type="range" min={0} max={1.5} step={0.05} value={lowLoraWeight}
                onChange={(e) => setLowLoraWeight(parseFloat(e.target.value) || 0)} />
            </div>
          </>
        )}

        <div className="section-title">Dimensions</div>
        <div className="field">
          <select value={preset} onChange={(e) => applyPreset(e.target.value)}>
            {DIM_PRESETS.map(([n]) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div className="field-row">
          <div className="field">
            <label>Width</label>
            <input type="number" min={64} step={16} value={w}
              onChange={(e) => setW(parseInt(e.target.value) || 0)} disabled={dimsLocked} />
          </div>
          <div className="field">
            <label>Height</label>
            <input type="number" min={64} step={16} value={h}
              onChange={(e) => setH(parseInt(e.target.value) || 0)} disabled={dimsLocked} />
          </div>
        </div>

        <div className="section-title">Frames</div>
        <div className="field-row">
          <div className="field">
            <label title="(frames − 1) is snapped to a multiple of 4">Frames</label>
            <input type="number" min={5} max={161} step={4} value={numFrames}
              onChange={(e) => setNumFrames(parseInt(e.target.value) || 81)} />
          </div>
          <div className="field">
            <label>FPS</label>
            <input type="number" min={1} max={60} value={fps}
              onChange={(e) => setFps(parseInt(e.target.value) || 16)} />
          </div>
        </div>
        <div className="muted small" style={{ marginBottom: 6 }}>
          {(((numFrames - 1) / Math.max(fps, 1)) + (1 / Math.max(fps, 1))).toFixed(1)}s clip at {fps} fps.
        </div>

        <div className="section-title">Sampling</div>
        <div className="field-row">
          <div className="field">
            <label>Steps</label>
            <input type="number" min={1} max={60} value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value) || 4)} />
          </div>
          <div className="field">
            <label title="WAN i2v uses guidance 1.0 with the lightx2v LoRA">CFG</label>
            <input type="number" min={0} step={0.5} value={cfg}
              onChange={(e) => setCfg(parseFloat(e.target.value) || 1.0)} />
          </div>
        </div>
        <div className="field">
          <label>
            <input type="checkbox" checked={randomSeed}
              onChange={(e) => setRandomSeed(e.target.checked)} /> random seed
          </label>
        </div>
        {!randomSeed && (
          <div className="field">
            <label>Seed</label>
            <input type="number" value={seed} onChange={(e) => setSeed(parseInt(e.target.value) || 0)} />
          </div>
        )}
      </aside>
    </>
  );
}
