import { useState } from "react";
import {
  audioFileUrl,
  audioGenerate,
  cancelAudioJob,
  getAudioJob,
  type AudioHealth,
  type ModelListing,
} from "./api/sidecar";

interface MusicProps {
  models: ModelListing | null;
  sidecar: string; // "up" | "down" | "checking"
}

export default function Music({ models, sidecar }: MusicProps) {
  // Form state
  const [prompt, setPrompt] = useState("cinematic space odyssey, vast choirs, pulsing synths, emotional climax");
  const [lyrics, setLyrics] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [bpm, setBpm] = useState(128);
  const [duration, setDuration] = useState(75);
  const [temperature, setTemperature] = useState(0.9);
  const [generateCover, setGenerateCover] = useState(true);
  const [coverPrompt, setCoverPrompt] = useState("");

  // Job / progress state
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState("");
  const [result, setResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  // ACE service health (local to this tab)
  const [aceHealth, setAceHealth] = useState<AudioHealth | null>(null);
  const [aceChecking, setAceChecking] = useState(false);

  async function checkAce() {
    setAceChecking(true);
    try {
      const resp = await (await import("./api/sidecar")).audioHealth();
      setAceHealth(resp);
    } catch (e: any) {
      setAceHealth({ ok: false, detail: String(e?.message ?? e) });
    } finally {
      setAceChecking(false);
    }
  }

  async function submit() {
    if (sidecar !== "up") return;
    setBusy(true);
    setJobId(null);
    setJobStatus(null);
    setProgress(0);
    setProgressMsg("Submitting to orchestrator...");
    setResult(null);

    try {
      const payload = {
        prompt,
        lyrics,
        ace_model: selectedModel || null,
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

      // Poll until done
      const poll = async () => {
        try {
          const snap = await getAudioJob(jid);
          setJobStatus(snap.status);

          if (snap.progress?.message) setProgressMsg(snap.progress.message);

          if (snap.result) {
            setResult(snap.result);
            setProgress(1);
            setProgressMsg("Complete — audio ready for playback");
            setBusy(false);
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

  const audioModels = models?.categories?.audio ?? [];

  return (
    <>
      {/* Center pane — creative flow (prompt, lyrics, big button, progress, results) */}
      <main className="pane center gen-center" style={{ padding: 20, overflow: "auto" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
          <span style={{ fontSize: 28 }}>♪</span>
          <div>
            <div style={{ fontSize: 20, fontWeight: 600 }}>Music — ACE-Step</div>
            <div style={{ color: "var(--muted)", fontSize: 12 }}>
              Using your existing Kraken_Audio installation (no model duplication)
            </div>
          </div>
        </div>

        <div className="prompt-stack">
          <label className="lbl">Prompt / Caption</label>
          <textarea
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="epic orchestral journey through a dying star, cinematic, emotional"
          />

          <label className="lbl" style={{ marginTop: 8 }}>Lyrics (optional)</label>
          <textarea
            rows={6}
            value={lyrics}
            onChange={(e) => setLyrics(e.target.value)}
            placeholder="[Verse 1]\nWe sailed the edge of night...\n[Chorus]\n..."
            style={{ fontFamily: "monospace", fontSize: 13 }}
          />
        </div>

        <div className="action-row" style={{ marginTop: 12 }}>
          <button
            className="primary big"
            onClick={submit}
            disabled={busy || sidecar !== "up"}
          >
            {busy ? "Working..." : "♪ Generate with ACE-Step"}
          </button>
          {jobId && <button onClick={cancel}>Cancel</button>}
          <div className="spacer" />
          <div style={{ fontSize: 12, color: "var(--muted)" }}>
            Cover (if checked) is generated with Kraken Art's internal FLUX/SDXL first.
          </div>
        </div>

        {/* Live job progress */}
        {jobId && jobStatus && (
          <div style={{ marginTop: 16, background: "#0b0f16", padding: 14, borderRadius: 6 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 13 }}>
              <div><b>Job:</b> {jobId.slice(0, 8)}… &nbsp; <b>Status:</b> {jobStatus}</div>
              <div>{progressMsg}</div>
            </div>
            <div style={{ height: 6, background: "#1f2937", borderRadius: 3, overflow: "hidden" }}>
              <div style={{ width: `${Math.min(100, progress * 100)}%`, height: "100%", background: "#22c55e", transition: "width 200ms" }} />
            </div>

            {result && (
              <div style={{ marginTop: 12, color: "var(--c-good)", fontSize: 13 }}>
                Done!
                {(result.audio_paths || []).length > 0 ? (
                  (result.audio_paths as string[]).map((p: string, idx: number) => (
                    <div key={idx} style={{ marginTop: 8 }}>
                      <audio controls src={audioFileUrl(p)} style={{ width: "100%", maxWidth: 520 }} />
                      <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{p}</div>
                    </div>
                  ))
                ) : (
                  <pre style={{ fontSize: 11, marginTop: 8, whiteSpace: "pre-wrap" }}>{JSON.stringify(result, null, 2)}</pre>
                )}
                {result.cover_path && (
                  <div style={{ marginTop: 8, fontSize: 12 }}>
                    Album cover saved by Kraken Art: <code>{result.cover_path}</code>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Right pane — model selectors & settings (exactly like Generate.tsx) */}
      <aside className="pane right-params" style={{ padding: 14, overflowY: "auto" }}>
        <div className="section-title">ACE Engine</div>

        <div className="field">
          <button onClick={checkAce} disabled={aceChecking || sidecar !== "up"} style={{ fontSize: 12, width: "100%" }}>
            {aceChecking ? "Checking..." : "Check ACE service (port 8001)"}
          </button>
          {!aceHealth ? (
            <div className="muted small" style={{ marginTop: 6 }}>Click above to verify the finished Kraken_Audio engine is running.</div>
          ) : aceHealth.ok ? (
            <div style={{ color: "var(--c-good)", fontSize: 12, marginTop: 6 }}>ACE service healthy ✓</div>
          ) : (
            <div style={{ color: "var(--c-bad)", fontSize: 12, marginTop: 6 }}>ACE not reachable — start it with your normal Kraken_Audio launcher first.</div>
          )}
        </div>

        <div className="section-title" style={{ marginTop: 16 }}>Model</div>
        <div className="field">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{ width: "100%" }}
          >
            <option value="">(auto — let ACE pick)</option>
            {audioModels.map((m, i) => (
              <option key={i} value={m.abs_path}>
                {m.filename} — {(m.size_bytes / 1e9).toFixed(1)} GB
              </option>
            ))}
          </select>
          <div className="muted small" style={{ marginTop: 4 }}>
            {audioModels.length} audio model(s) found via Refresh Models
          </div>
        </div>

        <div className="section-title" style={{ marginTop: 16 }}>Album Cover</div>
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
              placeholder="Cover prompt (optional)"
              value={coverPrompt}
              onChange={(e) => setCoverPrompt(e.target.value)}
            />
          )}
          <div className="muted small" style={{ marginTop: 4 }}>
            This redirects what the old Kraken_Audio used to call ComfyUI for.
          </div>
        </div>

        <div className="section-title" style={{ marginTop: 16 }}>Parameters</div>
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
            <input type="number" step="0.05" value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value) || 0.85)} />
          </div>
        </div>

        <div style={{ marginTop: 20, fontSize: 11, color: "var(--muted)" }}>
          The actual heavy synthesis runs in your existing finished ACE-Step process (the one on port 8001). Kraken Art only orchestrates + generates the cover.
        </div>
      </aside>
    </>
  );
}
