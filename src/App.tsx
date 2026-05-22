import { Fragment, useEffect, useState } from "react";
import "./App.css";
import logoUrl from "./assets/kraken-logo.png";
import Generate from "./Generate";
import Library from "./Library";
import Logs from "./Logs";
import SettingsModal from "./Settings";
import { clearMemory, getDeps, getGpu, getLogs, getModels, getSettings, health, refreshModels, type DepsStatus, type GpuInfo, type Health, type ModelListing, type Settings } from "./api/sidecar";
import { listen } from "@tauri-apps/api/event";

type Tab = "image" | "library";

type Status = "checking" | "up" | "down";

export default function App() {
  const [sidecar, setSidecar] = useState<Status>("checking");
  const [hp, setHp] = useState<Health | null>(null);
  const [gpu, setGpu] = useState<GpuInfo | null>(null);
  const [deps, setDeps] = useState<DepsStatus | null>(null);
  const [models, setModels] = useState<ModelListing | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [logsOpen, setLogsOpen] = useState(false);
  const [logBadge, setLogBadge] = useState(0); // count of new ERROR/WARN since last view
  const [lastSeenLogId, setLastSeenLogId] = useState(-1);
  const [clearing, setClearing] = useState(false);
  const [clearMsg, setClearMsg] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("image");
  const [settings, setSettings] = useState<Settings | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  async function bootstrap() {
    try {
      const h = await health();
      setHp(h);
      setSidecar("up");
    } catch {
      setSidecar("down");
      return;
    }
    await Promise.allSettled([
      getGpu().then(setGpu),
      getDeps().then(setDeps),
      getModels().then(setModels),
      getSettings().then(setSettings),
    ]);
  }

  useEffect(() => {
    let cancelled = false;
    let attempts = 0;
    async function tryConnect() {
      while (!cancelled && attempts < 30) {
        try {
          await health();
          await bootstrap();
          return;
        } catch {
          attempts++;
          await new Promise((r) => setTimeout(r, 500));
        }
      }
      if (!cancelled) setSidecar("down");
    }
    tryConnect();
    return () => {
      cancelled = true;
    };
  }, []);

  // Sidecar watchdog notifications. The Tauri host emits `sidecar-restarted`
  // after it successfully respawns a crashed Python process (see lib.rs).
  // We re-bootstrap so the UI picks up the fresh /health, /api/gpu etc.
  // A short banner tells the user what happened so they don't think the gen
  // they were running just disappeared into the void.
  const [restartedBanner, setRestartedBanner] = useState<string | null>(null);
  useEffect(() => {
    const unlisten = listen("sidecar-restarted", () => {
      setSidecar("checking");
      setRestartedBanner("Sidecar restarted after a crash. Any in-flight job was cancelled — generate again to retry.");
      bootstrap();
      // Auto-dismiss the banner after 15 s.
      setTimeout(() => setRestartedBanner(null), 15000);
    });
    return () => { unlisten.then((f) => f()); };
  }, []);

  // Background poll: count new ERROR/WARN events; auto-open drawer on first error.
  useEffect(() => {
    if (sidecar !== "up") return;
    let cancelled = false;
    let lastId = lastSeenLogId;
    async function tick() {
      try {
        const r = await getLogs(lastId >= 0 ? lastId : undefined, 200);
        if (cancelled) return;
        const errors = r.items.filter((it) => it.level === "ERROR" || it.level === "CRITICAL");
        const warns  = r.items.filter((it) => it.level === "WARNING");
        if (errors.length > 0 && !logsOpen) setLogsOpen(true);
        if (!logsOpen) setLogBadge((b) => b + errors.length + warns.length);
        if (r.last_id > lastId) lastId = r.last_id;
      } catch { /* ignore */ }
    }
    tick();
    const h = setInterval(tick, 2500);
    return () => { cancelled = true; clearInterval(h); };
  }, [sidecar, logsOpen]);

  function toggleLogs() {
    setLogsOpen((o) => !o);
    setLogBadge(0);
  }

  // Periodically refresh GPU info so vram free reflects load/unload.
  useEffect(() => {
    if (sidecar !== "up") return;
    const h = setInterval(() => { getGpu().then(setGpu).catch(() => {}); }, 5000);
    return () => clearInterval(h);
  }, [sidecar]);

  async function doClearMemory() {
    setClearing(true);
    setClearMsg(null);
    try {
      const r = await clearMemory();
      const freed = r.freed_mb ?? 0;
      setClearMsg(freed > 0 ? `freed ${(freed/1024).toFixed(2)} GB` : "VRAM cache cleared");
      getGpu().then(setGpu).catch(() => {});
      setTimeout(() => setClearMsg(null), 4000);
    } catch (e: any) {
      setClearMsg(`error: ${e?.message ?? e}`);
    } finally {
      setClearing(false);
    }
  }

  async function doRefresh() {
    setRefreshing(true);
    try {
      await refreshModels();
      const fresh = await getModels();
      setModels(fresh);
    } finally {
      setRefreshing(false);
    }
  }

  const sidecarPill =
    sidecar === "up" ? <span className="pill ok"><span className="dot" />sidecar online</span> :
    sidecar === "down" ? <span className="pill bad"><span className="dot" />sidecar offline</span> :
    <span className="pill"><span className="dot" />connecting…</span>;

  const gpuPill = !gpu ? null :
    gpu.cuda_available
      ? <span className="pill ok"><span className="dot" />{gpu.name?.replace("NVIDIA GeForce ", "") ?? "GPU"}</span>
      : <span className="pill bad"><span className="dot" />no CUDA</span>;

  const depsPill = !deps ? null :
    deps.all_ok
      ? <span className="pill ok"><span className="dot" />deps OK</span>
      : <span className="pill warn"><span className="dot" />{deps.missing} missing</span>;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><img className="brand-logo" src={logoUrl} alt="" />Kraken Art</div>
        <nav className="tab-strip">
          <button className={"tab " + (tab === "image" ? "active" : "")} onClick={() => setTab("image")}>Image</button>
          <button className={"tab " + (tab === "library" ? "active" : "")} onClick={() => setTab("library")}>Library</button>
        </nav>
        {sidecarPill}
        {gpuPill}
        {depsPill}
        <div className="spacer" />
        {clearMsg && <span className="muted small" style={{ marginRight: 6 }}>{clearMsg}</span>}
        <button onClick={() => setSettingsOpen(true)} disabled={sidecar !== "up" || !settings} title="Settings (Civitai token, NSFW)">⚙</button>
        <button onClick={doClearMemory} disabled={clearing || sidecar !== "up"} title="Unload pipelines and clear CUDA cache">
          {clearing ? "Clearing…" : "Clear VRAM"}
        </button>
        <button onClick={toggleLogs} className={logBadge > 0 ? "btn-with-badge" : ""}>
          Logs{logBadge > 0 ? <span className="badge">{logBadge}</span> : null}
        </button>
        <button onClick={doRefresh} disabled={refreshing || sidecar !== "up"}>
          {refreshing ? "Scanning…" : "Refresh models"}
        </button>
      </header>

      {restartedBanner && (
        <div className="restart-banner" role="alert">
          {restartedBanner}
          <button className="banner-close" onClick={() => setRestartedBanner(null)} aria-label="dismiss">×</button>
        </div>
      )}

      <div className="main">
        {/* Left pane — system + model counts */}
        <aside className="pane left-system">
          <div className="section-title">GPU</div>
          <div className="card">
            {!gpu && <div className="muted">probing…</div>}
            {gpu && (
              <div className="kv">
                <div className="k">device</div><div className="v">{gpu.name ?? "—"}</div>
                <div className="k">vram</div>  <div className="v">{gpu.vram_total_mb ? `${(gpu.vram_total_mb/1024).toFixed(1)} GB` : "—"}</div>
                <div className="k">free</div>  <div className="v" style={{ color: gpu.vram_free_mb && gpu.vram_free_mb < 4096 ? "var(--c-bad)" : undefined }}>{gpu.vram_free_mb ? `${(gpu.vram_free_mb/1024).toFixed(1)} GB` : "—"}</div>
                <div className="k">cuda</div>  <div className="v">{gpu.cuda_runtime ?? "—"}</div>
                <div className="k">torch</div> <div className="v">{gpu.torch_version ?? "—"}</div>
                <div className="k">driver</div><div className="v">{gpu.driver ?? "—"}</div>
              </div>
            )}
          </div>

          <div className="section-title">Dependencies</div>
          <div className="card">
            {!deps && <div className="muted">probing…</div>}
            {deps && (
              <div className="kv">
                {deps.packages.map((p) => (
                  <PkgRow key={p.name} pkg={p} />
                ))}
              </div>
            )}
          </div>

          <div className="section-title">Models</div>
          <div className="card">
            {!models && <div className="muted">scanning…</div>}
            {models && (
              <div className="count-grid">
                {Object.entries(models.counts).map(([k, n]) => (
                  <Fragment key={k}>
                    <div className="k">{k}</div>
                    <div className={"v " + (n === 0 ? "zero" : "")}>{n}</div>
                  </Fragment>
                ))}
              </div>
            )}
            {models && <div className="muted" style={{ marginTop: 10, fontSize: 11 }}>{models.root}</div>}
          </div>
        </aside>

        {/* Image tab renders BOTH center + right pane as siblings of the left aside.
            Library tab renders a single center pane (no right). */}
        {tab === "image" && <Generate models={models} />}
        {tab === "library" && <Library settings={settings} onModelsChanged={doRefresh} />}
      </div>

      <Logs open={logsOpen} onClose={() => { setLogsOpen(false); setLogBadge(0); }} />
      {settingsOpen && settings && (
        <SettingsModal settings={settings} onClose={() => setSettingsOpen(false)} onChanged={setSettings} />
      )}
    </div>
  );
}

function PkgRow({ pkg }: { pkg: import("./api/sidecar").DepPackage }) {
  return (
    <>
      <div className="k">{pkg.name}</div>
      <div className="v" style={{ color: pkg.ok ? "var(--c-good)" : "var(--c-bad)" }}>
        {pkg.installed ?? "missing"}
      </div>
    </>
  );
}
