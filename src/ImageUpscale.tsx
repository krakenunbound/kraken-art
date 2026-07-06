import { useEffect, useRef, useState } from "react";
import {
  cancelJob,
  getLatestImages,
  openJobWS,
  outputFileUrl,
  startUpscale,
  type ModelListing,
  type Settings,
  type UpscaleParams,
} from "./api/sidecar";

type Source = {
  rel_path: string;
  filename: string;
  src: string;
  model_label: string;
  width: number | null;
  height: number | null;
};

type Mode = "esrgan" | "usdu";
type SizeMode = "factor" | "resolution";

const RES_PRESETS: { label: string; w: number; h: number }[] = [
  { label: "1080p · 1920×1080", w: 1920, h: 1080 },
  { label: "1440p · 2560×1440", w: 2560, h: 1440 },
  { label: "4K · 3840×2160", w: 3840, h: 2160 },
  { label: "Square 2K · 2048×2048", w: 2048, h: 2048 },
  { label: "Square 4K · 4096×4096", w: 4096, h: 4096 },
  { label: "8K · 7680×4320", w: 7680, h: 4320 },
];

const SAMPLERS = ["dpmpp_2m", "euler_a", "euler", "dpmpp_sde", "heun", "unipc", "ddim"];
const SCHEDULERS = ["karras", "normal", "beta", "exponential", "sgm_uniform"];

const GALLERY_LIMIT = 60;

export default function ImageUpscale({
  active,
  models,
}: {
  active: boolean;
  models: ModelListing | null;
  settings: Settings | null;
}) {
  const [sources, setSources] = useState<Source[]>([]);
  const [selected, setSelected] = useState<Source | null>(null);

  const [mode, setMode] = useState<Mode>("usdu");
  const [upscaleModel, setUpscaleModel] = useState<string>("");
  const [sizeMode, setSizeMode] = useState<SizeMode>("resolution");
  const [factor, setFactor] = useState(2);
  const [presetIdx, setPresetIdx] = useState(2); // 4K
  const [customRes, setCustomRes] = useState(false);
  const [targetW, setTargetW] = useState(3840);
  const [targetH, setTargetH] = useState(2160);

  // USDU refine
  const [refineCkpt, setRefineCkpt] = useState<string>("");
  const [refineVae, setRefineVae] = useState<string>("");
  const [steps, setSteps] = useState(20);
  const [denoise, setDenoise] = useState(0.2);
  const [tileSize, setTileSize] = useState<number | "">("");
  const [cfg, setCfg] = useState(6);
  const [sampler, setSampler] = useState("dpmpp_2m");
  const [scheduler, setScheduler] = useState("karras");
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");

  // job state
  const [status, setStatus] = useState<string>("idle");
  const [step, setStep] = useState({ step: 0, total: 0 });
  const [progressMsg, setProgressMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [resultUrl, setResultUrl] = useState<string>("");
  const [resultInfo, setResultInfo] = useState<string>("");
  const [jobId, setJobId] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const upscalers = models?.categories.upscale_models ?? [];
  const checkpoints = models?.categories.checkpoints ?? [];
  const vaes = models?.categories.vae ?? [];

  // Default the upscale model + refine checkpoint once models arrive.
  useEffect(() => {
    if (!upscaleModel && upscalers.length) {
      const sharp = upscalers.find((m) => /ultrasharp|4x/i.test(m.name));
      setUpscaleModel((sharp ?? upscalers[0]).name);
    }
    if (!refineCkpt && checkpoints.length) {
      const xl = checkpoints.find((m) => /xl|epicrealism|juggernaut|illustrious/i.test(m.name));
      setRefineCkpt((xl ?? checkpoints[0]).name);
    }
  }, [models]);

  // Load gallery as the source picker whenever the tab activates.
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    getLatestImages(GALLERY_LIMIT)
      .then(async (data) => {
        if (cancelled) return;
        const items = await Promise.all(
          data.items.map(async (it) => ({
            rel_path: it.rel_path,
            filename: it.filename,
            src: await outputFileUrl(it.rel_path),
            model_label: it.model_label,
            width: it.width ?? null,
            height: it.height ?? null,
          } satisfies Source)),
        );
        if (!cancelled) {
          setSources(items);
          if (!selected && items.length) setSelected(items[0]);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [active]);

  const preset = RES_PRESETS[presetIdx];
  const effTargetW = sizeMode === "resolution" ? (customRes ? targetW : preset.w) : 0;
  const effTargetH = sizeMode === "resolution" ? (customRes ? targetH : preset.h) : 0;

  const busy = status === "queued" || status === "running" || status === "submitting";

  async function run() {
    if (!selected) {
      setErrorMsg("Pick a source image from the gallery first.");
      return;
    }
    setErrorMsg("");
    setResultUrl("");
    setResultInfo("");
    setStep({ step: 0, total: 0 });
    setProgressMsg("");
    setStatus("submitting");

    const params: UpscaleParams = {
      source_rel_path: selected.rel_path,
      mode,
      upscale_model: upscaleModel || null,
      size_mode: sizeMode,
      factor,
      target_w: sizeMode === "resolution" ? effTargetW : null,
      target_h: sizeMode === "resolution" ? effTargetH : null,
      refine_arch: "sdxl",
      refine_checkpoint: refineCkpt || null,
      refine_vae: refineVae || null,
      steps,
      denoise,
      tile_size: tileSize === "" ? null : Number(tileSize),
      cfg,
      sampler,
      scheduler,
      prompt,
      negative,
      seed: 0,
    };

    try {
      const r = await startUpscale(params);
      setJobId(r.job_id);
      setStatus(r.status);
      const ws = await openJobWS(r.job_id);
      wsRef.current = ws;
      ws.onmessage = async (e) => {
        const evt = JSON.parse(e.data);
        if (evt.type === "snapshot" || evt.type === "status") {
          setStatus(evt.status);
          if (evt.error) setErrorMsg(evt.error);
          if (evt.status === "succeeded" && evt.result) {
            const out = evt.result.output;
            if (out?.rel_path) {
              setResultUrl(await outputFileUrl(out.rel_path));
              setResultInfo(
                `${out.width}×${out.height} · ${evt.result.mode.toUpperCase()}` +
                  (evt.result.tiles ? ` · ${evt.result.tiles} tiles` : "") +
                  ` · ${evt.result.elapsed_sec}s`,
              );
            }
          }
        }
        if (evt.type === "progress") {
          setStep({ step: evt.step, total: evt.total_steps });
          if (evt.message) setProgressMsg(evt.message);
        }
        if (evt.type === "image" && evt.preview_b64) {
          setResultUrl(`data:image/jpeg;base64,${evt.preview_b64}`);
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
      ws.onclose = () => resetIfStuck("Connection to sidecar lost — check Logs.");
      ws.onerror = () => resetIfStuck("WebSocket error — sidecar may have died. Check Logs.");
    } catch (e: any) {
      setStatus("idle");
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

  const pct = step.total > 0 ? Math.floor((step.step / step.total) * 100) : 0;

  return (
    <>
      <main className={"pane center gen-center" + (active ? "" : " tab-hidden")}>
        <div className="section-title">Source</div>
        <div className="card" style={{ maxHeight: 260, overflowY: "auto" }}>
          {sources.length === 0 && <div className="muted">No images yet — generate something first.</div>}
          <div className="up-grid">
            {sources.map((s) => (
              <button
                key={s.rel_path}
                className={"up-thumb" + (selected?.rel_path === s.rel_path ? " sel" : "")}
                onClick={() => setSelected(s)}
                title={`${s.filename}${s.width ? ` · ${s.width}×${s.height}` : ""}`}
              >
                <img src={s.src} alt="" />
              </button>
            ))}
          </div>
        </div>

        <div className="section-title" style={{ marginTop: 14 }}>
          {resultUrl ? "Result" : "Preview"}
        </div>
        <div className="card up-preview">
          {resultUrl ? (
            <img src={resultUrl} alt="result" />
          ) : selected ? (
            <img src={selected.src} alt="source" />
          ) : (
            <div className="muted">Select a source image.</div>
          )}
          {resultInfo && <div className="muted small" style={{ marginTop: 8 }}>{resultInfo}</div>}
          {selected && !resultUrl && (
            <div className="muted small" style={{ marginTop: 8 }}>
              source: {selected.width ? `${selected.width}×${selected.height}` : selected.filename}
              {sizeMode === "resolution" && effTargetW ? `  →  ${effTargetW}×${effTargetH}` : ""}
              {sizeMode === "factor" ? `  →  ${factor}×` : ""}
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
            <div className="muted small" style={{ marginTop: 6 }}>
              {progressMsg || status}
            </div>
          </div>
        )}
        {errorMsg && (
          <div className="card error-card" style={{ marginTop: 12 }}>
            {errorMsg}
          </div>
        )}
      </main>

      <aside className={"pane right-params" + (active ? "" : " tab-hidden")}>
        <div className="section-title">Upscale mode</div>
        <div className="seg">
          <button className={mode === "esrgan" ? "active" : ""} onClick={() => setMode("esrgan")}>
            ESRGAN
          </button>
          <button className={mode === "usdu" ? "active" : ""} onClick={() => setMode("usdu")}>
            USDU
          </button>
        </div>
        <div className="muted small" style={{ marginTop: 4 }}>
          {mode === "esrgan"
            ? "Fast single-pass upscaler. No diffusion."
            : "Tile + img2img refine through a model. Slow, adds detail."}
        </div>

        <div className="section-title" style={{ marginTop: 14 }}>Upscaler model</div>
        <select value={upscaleModel} onChange={(e) => setUpscaleModel(e.target.value)}>
          {upscalers.length === 0 && <option value="">(none found)</option>}
          {upscalers.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}
            </option>
          ))}
        </select>

        <div className="section-title" style={{ marginTop: 14 }}>Output size</div>
        <div className="seg">
          <button className={sizeMode === "factor" ? "active" : ""} onClick={() => setSizeMode("factor")}>
            Factor
          </button>
          <button className={sizeMode === "resolution" ? "active" : ""} onClick={() => setSizeMode("resolution")}>
            Target res
          </button>
        </div>

        {sizeMode === "factor" ? (
          <label className="field" style={{ marginTop: 8 }}>
            <span>Scale ×{factor}</span>
            <input
              type="range"
              min={1}
              max={4}
              step={0.5}
              value={factor}
              onChange={(e) => setFactor(Number(e.target.value))}
            />
          </label>
        ) : (
          <>
            <label className="field" style={{ marginTop: 8 }}>
              <span>Preset</span>
              <select
                value={customRes ? "custom" : String(presetIdx)}
                onChange={(e) => {
                  if (e.target.value === "custom") {
                    setCustomRes(true);
                  } else {
                    setCustomRes(false);
                    setPresetIdx(Number(e.target.value));
                  }
                }}
              >
                {RES_PRESETS.map((p, i) => (
                  <option key={p.label} value={String(i)}>
                    {p.label}
                  </option>
                ))}
                <option value="custom">Custom…</option>
              </select>
            </label>
            {customRes && (
              <div className="row2" style={{ marginTop: 6 }}>
                <label className="field">
                  <span>Width</span>
                  <input type="number" value={targetW} min={64} step={8} onChange={(e) => setTargetW(Number(e.target.value))} />
                </label>
                <label className="field">
                  <span>Height</span>
                  <input type="number" value={targetH} min={64} step={8} onChange={(e) => setTargetH(Number(e.target.value))} />
                </label>
              </div>
            )}
          </>
        )}

        {mode === "usdu" && (
          <>
            <div className="section-title" style={{ marginTop: 14 }}>Refine model (SDXL)</div>
            <select value={refineCkpt} onChange={(e) => setRefineCkpt(e.target.value)}>
              {checkpoints.length === 0 && <option value="">(no checkpoints)</option>}
              {checkpoints.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                </option>
              ))}
            </select>
            <label className="field" style={{ marginTop: 6 }}>
              <span>VAE (optional)</span>
              <select value={refineVae} onChange={(e) => setRefineVae(e.target.value)}>
                <option value="">(checkpoint default)</option>
                {vaes.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name}
                  </option>
                ))}
              </select>
            </label>

            <div className="row2" style={{ marginTop: 8 }}>
              <label className="field">
                <span>Steps</span>
                <input type="number" value={steps} min={1} max={60} onChange={(e) => setSteps(Number(e.target.value))} />
              </label>
              <label className="field">
                <span>Denoise {denoise.toFixed(2)}</span>
                <input
                  type="range"
                  min={0.05}
                  max={0.6}
                  step={0.01}
                  value={denoise}
                  onChange={(e) => setDenoise(Number(e.target.value))}
                />
              </label>
            </div>

            <div className="row2" style={{ marginTop: 6 }}>
              <label className="field">
                <span>Tile (auto if blank)</span>
                <input
                  type="number"
                  value={tileSize}
                  min={256}
                  step={64}
                  placeholder="auto"
                  onChange={(e) => setTileSize(e.target.value === "" ? "" : Number(e.target.value))}
                />
              </label>
              <label className="field">
                <span>CFG {cfg}</span>
                <input type="range" min={1} max={12} step={0.5} value={cfg} onChange={(e) => setCfg(Number(e.target.value))} />
              </label>
            </div>

            <div className="row2" style={{ marginTop: 6 }}>
              <label className="field">
                <span>Sampler</span>
                <select value={sampler} onChange={(e) => setSampler(e.target.value)}>
                  {SAMPLERS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Scheduler</span>
                <select value={scheduler} onChange={(e) => setScheduler(e.target.value)}>
                  {SCHEDULERS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <label className="field" style={{ marginTop: 8 }}>
              <span>Refine prompt (optional)</span>
              <textarea rows={2} value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="leave blank to refine detail only" />
            </label>
            <label className="field" style={{ marginTop: 6 }}>
              <span>Negative (optional)</span>
              <textarea rows={2} value={negative} onChange={(e) => setNegative(e.target.value)} />
            </label>
          </>
        )}

        <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
          {busy ? (
            <button className="primary" onClick={cancel}>
              Cancel
            </button>
          ) : (
            <button className="primary" onClick={run} disabled={!selected || (mode === "usdu" && !refineCkpt)}>
              Upscale
            </button>
          )}
        </div>
      </aside>
    </>
  );
}
