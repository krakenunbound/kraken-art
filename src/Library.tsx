import { useCallback, useEffect, useState } from "react";
import {
  civitaiDownload, civitaiSearch, openJobWS,
  type CivitaiModel, type CivitaiSearchParams, type CivitaiVersion, type CivitaiFile, type Settings,
} from "./api/sidecar";

const TYPES = ["", "Checkpoint", "LORA", "TextualInversion", "VAE", "Controlnet", "Upscaler"];
const BASE_MODELS = ["", "SDXL 1.0", "SDXL Turbo", "Pony", "Illustrious", "SD 1.5", "Flux.1 D", "Flux.1 S", "SD 3", "SD 3.5"];
const SORTS = ["Most Downloaded", "Highest Rated", "Most Liked", "Newest"];

type DownloadState = { jobId: string; pct: number; speedMbps: number; status: string; dest?: string; error?: string };

export default function Library({ settings, onModelsChanged }: { settings: Settings | null; onModelsChanged: () => void }) {
  const [query, setQuery] = useState("");
  const [type, setType] = useState("LORA");
  const [baseModel, setBaseModel] = useState("");
  const [sort, setSort] = useState("Most Downloaded");
  const [showNsfw, setShowNsfw] = useState(!!settings?.civitai.nsfw_visible);
  const [page, setPage] = useState(1);

  const [items, setItems] = useState<CivitaiModel[]>([]);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [detail, setDetail] = useState<CivitaiModel | null>(null);

  // download state by versionId → DownloadState (also lets us show progress on cards)
  const [downloads, setDownloads] = useState<Record<number, DownloadState>>({});

  const runSearch = useCallback(async (resetPage: boolean) => {
    const p = resetPage ? 1 : page;
    setLoading(true); setError(null);
    try {
      const params: CivitaiSearchParams = {
        query: query.trim() || undefined,
        types: type || undefined,
        baseModels: baseModel || undefined,
        nsfw: showNsfw ? undefined : false,
        limit: 24, page: p, sort,
      };
      const r = await civitaiSearch(params);
      setItems(r.items);
      setTotalPages(r.metadata?.totalPages ?? 1);
      if (resetPage) setPage(1);
    } catch (e: any) {
      setError(String(e?.message ?? e));
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [query, type, baseModel, sort, showNsfw, page]);

  // initial load + reload on page change
  useEffect(() => { runSearch(false); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [page]);
  // first mount
  useEffect(() => { runSearch(true); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, []);

  function onSubmitFilters(e: React.FormEvent) { e.preventDefault(); runSearch(true); }

  async function startDownload(version: CivitaiVersion, file?: CivitaiFile) {
    try {
      const r = await civitaiDownload(version.id, file?.id);
      const initial: DownloadState = { jobId: r.job_id, pct: 0, speedMbps: 0, status: "queued", dest: r.dest };
      setDownloads((d) => ({ ...d, [version.id]: initial }));
      const ws = await openJobWS(r.job_id);
      ws.onmessage = (e) => {
        const ev = JSON.parse(e.data);
        setDownloads((d) => {
          const cur = d[version.id]; if (!cur) return d;
          if (ev.type === "status") {
            const next = { ...cur, status: ev.status, error: ev.error || cur.error };
            if (ev.status === "succeeded") { next.pct = 100; onModelsChanged(); }
            return { ...d, [version.id]: next };
          }
          if (ev.type === "progress") {
            return { ...d, [version.id]: { ...cur, pct: ev.step ?? 0, speedMbps: ev.speed_mbps ?? 0, status: "running" } };
          }
          return d;
        });
        if (ev.type === "status" && (ev.status === "succeeded" || ev.status === "failed" || ev.status === "cancelled")) {
          ws.close();
        }
      };
      ws.onerror = () => {
        setDownloads((d) => ({ ...d, [version.id]: { ...(d[version.id] || initial), status: "failed", error: "WS error" } }));
      };
    } catch (e: any) {
      setDownloads((d) => ({ ...d, [version.id]: { jobId: "", pct: 0, speedMbps: 0, status: "failed", error: String(e?.message ?? e) } }));
    }
  }

  const filtersRow = (
    <form className="lib-filters" onSubmit={onSubmitFilters}>
      <input className="lib-search" placeholder="Search Civitai…" value={query} onChange={(e) => setQuery(e.target.value)} />
      <select value={type} onChange={(e) => setType(e.target.value)} title="Type">
        {TYPES.map((t) => <option key={t || "_"} value={t}>{t || "Any type"}</option>)}
      </select>
      <select value={baseModel} onChange={(e) => setBaseModel(e.target.value)} title="Base model">
        {BASE_MODELS.map((b) => <option key={b || "_"} value={b}>{b || "Any base"}</option>)}
      </select>
      <select value={sort} onChange={(e) => setSort(e.target.value)} title="Sort">
        {SORTS.map((s) => <option key={s} value={s}>{s}</option>)}
      </select>
      <label className="row-flex" title="Allow NSFW results">
        <input type="checkbox" checked={showNsfw} onChange={(e) => setShowNsfw(e.target.checked)} /> NSFW
      </label>
      <button type="submit" className="primary" disabled={loading}>{loading ? "Searching…" : "Search"}</button>
    </form>
  );

  return (
    <main className="pane center lib-root">
      {filtersRow}
      {error && <div className="err">{error}</div>}

      <div className="lib-grid">
        {items.map((m) => {
          const v0 = m.modelVersions?.[0];
          const img = v0?.images?.find((i) => !i.nsfw || showNsfw)?.url ?? v0?.images?.[0]?.url;
          const dl = v0 ? downloads[v0.id] : undefined;
          return (
            <div className="lib-card" key={m.id} onClick={() => setDetail(m)}>
              <div className="lib-thumb">
                {img ? <img src={img} alt="" loading="lazy" /> : <div className="placeholder small">no preview</div>}
                {m.nsfw && <span className="lib-nsfw">NSFW</span>}
                <span className="lib-type">{m.type}</span>
              </div>
              <div className="lib-meta">
                <div className="lib-name" title={m.name}>{m.name}</div>
                <div className="muted small">
                  by {m.creator?.username ?? "?"} · {v0?.baseModel ?? "?"} · ⬇ {m.stats?.downloadCount ?? 0}
                </div>
                {dl && (
                  <div className="lib-progress">
                    <div className="progress-bar"><div className="progress-bar-fill" style={{ width: `${dl.pct}%` }} /></div>
                    <div className="muted small">{dl.status} · {dl.pct}% · {dl.speedMbps.toFixed(1)} MB/s</div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
        {!loading && items.length === 0 && <div className="placeholder">no results</div>}
      </div>

      <div className="lib-pagination">
        <button onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page <= 1 || loading}>← prev</button>
        <span className="muted small">page {page} / {totalPages}</span>
        <button onClick={() => setPage((p) => Math.min(totalPages, p + 1))} disabled={page >= totalPages || loading}>next →</button>
      </div>

      {detail && (
        <DetailsModal
          model={detail}
          downloads={downloads}
          onClose={() => setDetail(null)}
          onDownload={(version, file) => startDownload(version, file)}
          showNsfw={showNsfw}
        />
      )}
    </main>
  );
}

function DetailsModal({
  model, downloads, onClose, onDownload, showNsfw,
}: {
  model: CivitaiModel;
  downloads: Record<number, DownloadState>;
  onClose: () => void;
  onDownload: (v: CivitaiVersion, f?: CivitaiFile) => void;
  showNsfw: boolean;
}) {
  const versions = model.modelVersions ?? [];
  const [vIdx, setVIdx] = useState(0);
  const version = versions[vIdx];
  const [fIdx, setFIdx] = useState(0);
  const file = version?.files?.[fIdx];

  if (!version) {
    return <div className="modal-overlay" onClick={onClose}><div className="modal" onClick={(e) => e.stopPropagation()}>No versions.<button onClick={onClose}>Close</button></div></div>;
  }
  const dl = downloads[version.id];
  const preview = version.images?.find((i) => !i.nsfw || showNsfw)?.url ?? version.images?.[0]?.url;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal lib-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-title">{model.name}</div>
            <div className="muted small">by {model.creator?.username ?? "?"} · {model.type}</div>
          </div>
          <button onClick={onClose} className="x">✕</button>
        </div>

        <div className="lib-modal-body">
          <div className="lib-modal-left">
            {preview ? <img src={preview} alt="" /> : <div className="placeholder">no preview</div>}
            {version.trainedWords && version.trainedWords.length > 0 && (
              <>
                <div className="section-title">Trained words</div>
                <div className="trained-words">
                  {version.trainedWords.map((w) => <span className="tw-chip" key={w}>{w}</span>)}
                </div>
              </>
            )}
          </div>

          <div className="lib-modal-right">
            <div className="section-title">Version</div>
            <select value={vIdx} onChange={(e) => { setVIdx(parseInt(e.target.value)); setFIdx(0); }}>
              {versions.map((v, i) => <option key={v.id} value={i}>{v.name} — {v.baseModel}</option>)}
            </select>

            <div className="section-title">File</div>
            <select value={fIdx} onChange={(e) => setFIdx(parseInt(e.target.value))}>
              {(version.files ?? []).map((f, i) => (
                <option key={f.id} value={i}>{f.name} — {((f.sizeKB ?? 0) / 1024).toFixed(1)} MB{f.primary ? " (primary)" : ""}</option>
              ))}
            </select>

            <div className="section-title">Description</div>
            <div className="lib-desc" dangerouslySetInnerHTML={{ __html: model.description ?? "<em>no description</em>" }} />

            <div className="action-row" style={{ marginTop: 12 }}>
              <button className="primary big"
                onClick={() => onDownload(version, file)}
                disabled={dl?.status === "running" || dl?.status === "queued"}>
                {dl?.status === "succeeded" ? "Downloaded ✓"
                  : dl?.status === "running" ? `Downloading… ${dl.pct}%`
                  : dl?.status === "queued" ? "Queued…"
                  : `Download ${file ? `(${((file.sizeKB ?? 0) / 1024).toFixed(1)} MB)` : ""}`}
              </button>
            </div>
            {dl && (
              <div className="lib-progress" style={{ marginTop: 8 }}>
                <div className="progress-bar"><div className="progress-bar-fill" style={{ width: `${dl.pct}%` }} /></div>
                <div className="muted small">
                  {dl.status === "succeeded" && dl.dest ? `Saved to ${dl.dest}` : `${dl.status} · ${dl.pct}% · ${dl.speedMbps.toFixed(1)} MB/s`}
                </div>
                {dl.error && <div className="err">{dl.error}</div>}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
