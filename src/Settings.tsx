import { useState } from "react";
import { patchSettings, type Settings as TSettings } from "./api/sidecar";

export default function Settings({ settings, onClose, onChanged }: {
  settings: TSettings;
  onClose: () => void;
  onChanged: (s: TSettings) => void;
}) {
  const [token, setToken] = useState("");
  const [nsfw, setNsfw] = useState(settings.civitai.nsfw_visible);
  const [fastMode, setFastMode] = useState<"auto" | "on" | "off">(settings.performance.flux_fast_inference);
  const [bufferGb, setBufferGb] = useState(settings.performance.flux_fast_inference_buffer_gb);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  async function save() {
    setSaving(true); setMsg(null);
    try {
      const next = await patchSettings({
        civitai: {
          ...(token ? { api_token: token } : {}),
          nsfw_visible: nsfw,
        },
        performance: {
          flux_fast_inference: fastMode,
          flux_fast_inference_buffer_gb: Math.max(0.25, bufferGb),
        },
      });
      onChanged(next);
      setMsg("Saved. New settings apply on the next FLUX gen.");
      setToken("");
    } catch (e: any) {
      setMsg(`error: ${e?.message ?? e}`);
    } finally {
      setSaving(false);
    }
  }

  async function clearToken() {
    setSaving(true); setMsg(null);
    try {
      const next = await patchSettings({ civitai: { api_token: "" } });
      onChanged(next);
      setMsg("Token cleared.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 520 }}>
        <div className="modal-head">
          <div className="modal-title">Settings</div>
          <button onClick={onClose} className="x">✕</button>
        </div>

        <div className="section-title">Civitai</div>

        <div className="field">
          <label>API token (optional)</label>
          <input type="password" value={token} onChange={(e) => setToken(e.target.value)}
            placeholder={settings.civitai.api_token_set ? "•••••••• (set — paste to replace)" : "paste token to enable auth"} />
          <div className="muted small">
            Optional. Higher rate limits, access to gated models, ability to see your own profile.
            Get one at <a href="https://civitai.com/user/account" target="_blank" rel="noreferrer">civitai.com/user/account</a> → API Keys.
          </div>
        </div>

        <div className="field">
          <label className="row-flex">
            <input type="checkbox" checked={nsfw} onChange={(e) => setNsfw(e.target.checked)} />
            Show NSFW results in the Library tab
          </label>
        </div>

        <div className="section-title">FLUX performance</div>

        <div className="field">
          <label>Offload strategy</label>
          <select value={fastMode} onChange={(e) => setFastMode(e.target.value as "auto" | "on" | "off")}>
            <option value="auto">Auto — keep on GPU if it fits, else stream (recommended)</option>
            <option value="on">Force fully resident — fastest, may OOM on tight VRAM</option>
            <option value="off">Force streaming — slowest, always safe</option>
          </select>
          <div className="muted small">
            <code>Auto</code> compares model size against free VRAM. If the model fits with the headroom
            below, it stays fully on GPU. Otherwise the Forge-style per-Linear streaming offload runs:
            transformer blocks that fit stay resident, the rest stream from pinned host memory on a
            dedicated CUDA stream.
          </div>
        </div>

        <div className="field">
          <label>Activation headroom: <strong>{bufferGb.toFixed(2)} GB</strong></label>
          <input type="range" min={0.5} max={4} step={0.25} value={bufferGb}
                 onChange={(e) => setBufferGb(parseFloat(e.target.value))} />
          <div className="muted small">
            VRAM kept free for per-step activations + allocator fragmentation. <strong>2.0 GB</strong>
            is the default and safe for FLUX-dev at 1024². Lower it (1.0–1.5 GB) to let larger
            transformers stay fully resident — faster, but you risk OOM mid-step on big resolutions
            or batch counts. Raise it (3 GB+) if you see OOM errors at &gt; 1024×1024.
          </div>
        </div>

        <div className="action-row" style={{ marginTop: 12 }}>
          <button className="primary" onClick={save} disabled={saving}>Save</button>
          {settings.civitai.api_token_set && (
            <button onClick={clearToken} disabled={saving}>Clear token</button>
          )}
          <div className="spacer" />
          {msg && <span className="muted small">{msg}</span>}
        </div>
      </div>
    </div>
  );
}
