import { useCallback, useEffect, useRef, useState } from "react";
import { clearLogs, getLogs, type LogEntry } from "./api/sidecar";

type Props = { open: boolean; onClose: () => void };

const LEVEL_CLASS: Record<string, string> = {
  ERROR:    "lvl-error",
  CRITICAL: "lvl-error",
  WARNING:  "lvl-warn",
  INFO:     "lvl-info",
  DEBUG:    "lvl-debug",
};

const HEIGHT_KEY = "kraken.logs.height";
const DEFAULT_H  = 380;
const MIN_H      = 120;

function tsStr(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toTimeString().slice(0, 8) + "." + String(d.getMilliseconds()).padStart(3, "0");
}

export default function Logs({ open, onClose }: Props) {
  const [items, setItems] = useState<LogEntry[]>([]);
  const [lastId, setLastId] = useState<number>(-1);
  const [paused, setPaused] = useState(false);
  const [copied, setCopied] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  // ---- resizable height ----
  const [height, setHeight] = useState<number>(() => {
    const raw = localStorage.getItem(HEIGHT_KEY);
    const n = raw ? parseInt(raw, 10) : NaN;
    return Number.isFinite(n) && n > 0 ? n : DEFAULT_H;
  });
  useEffect(() => { localStorage.setItem(HEIGHT_KEY, String(height)); }, [height]);

  const dragRef = useRef({ startY: 0, startH: 0, dragging: false });

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!dragRef.current.dragging) return;
    const delta = dragRef.current.startY - e.clientY; // drag up grows
    const maxH = Math.max(MIN_H, Math.floor(window.innerHeight * 0.85));
    const next = Math.max(MIN_H, Math.min(maxH, dragRef.current.startH + delta));
    setHeight(next);
  }, []);

  const onMouseUp = useCallback(() => {
    dragRef.current.dragging = false;
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseup", onMouseUp);
  }, [onMouseMove]);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    dragRef.current = { startY: e.clientY, startH: height, dragging: true };
    document.body.style.userSelect = "none";
    document.body.style.cursor = "ns-resize";
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    e.preventDefault();
  }, [height, onMouseMove, onMouseUp]);

  // Double-click handle: collapse to min OR restore to default.
  const onHandleDouble = useCallback(() => {
    setHeight((h) => (h > MIN_H + 10 ? MIN_H : DEFAULT_H));
  }, []);

  // ---- log fetching ----
  async function refresh(incremental: boolean) {
    try {
      const since = incremental && lastId >= 0 ? lastId : undefined;
      const r = await getLogs(since, 500);
      if (r.items.length === 0 && since !== undefined) return;
      setItems((prev) => {
        const merged = since !== undefined ? [...prev, ...r.items] : r.items;
        return merged.slice(-2000);
      });
      if (r.last_id > lastId) setLastId(r.last_id);
    } catch { /* ignore */ }
  }

  useEffect(() => {
    if (!open || paused) return;
    refresh(true);
    const h = setInterval(() => refresh(true), 1500);
    return () => clearInterval(h);
  }, [open, paused, lastId]);

  useEffect(() => {
    if (open) { setLastId(-1); refresh(false); }
  }, [open]);

  useEffect(() => {
    if (!paused && listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [items, paused]);

  async function copyAll() {
    const text = items.map((it) => `[${tsStr(it.ts)}] ${it.level.padEnd(7)} ${it.logger}: ${it.message}`).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch { /* ignore */ }
  }

  async function doClear() {
    try {
      await clearLogs();
      setItems([]);
      setLastId(-1);
    } catch { /* ignore */ }
  }

  if (!open) return null;

  return (
    <div className="logs-drawer" style={{ height }}>
      <div className="logs-resize" onMouseDown={onMouseDown} onDoubleClick={onHandleDouble} title="Drag to resize · double-click to toggle min/default">
        <span className="logs-resize-grip" />
      </div>
      <div className="logs-bar">
        <div className="logs-title">SIDECAR LOG <span className="muted small">({items.length})</span></div>
        <div className="spacer" />
        <button onClick={copyAll}>{copied ? "Copied ✓" : "Copy all"}</button>
        <button onClick={() => setPaused((p) => !p)}>{paused ? "Resume" : "Pause"}</button>
        <button onClick={doClear}>Clear</button>
        <button onClick={onClose}>✕</button>
      </div>
      <div className="logs-list" ref={listRef}>
        {items.length === 0 && <div className="muted small" style={{ padding: 8 }}>no entries</div>}
        {items.map((it) => (
          <div key={it.id} className={"logs-row " + (LEVEL_CLASS[it.level] || "")}>
            <span className="logs-ts">{tsStr(it.ts)}</span>
            <span className="logs-lvl">{it.level}</span>
            <span className="logs-name">{it.logger}</span>
            <pre className="logs-msg">{it.message}</pre>
          </div>
        ))}
      </div>
    </div>
  );
}
