import { useCallback, useEffect, useRef, useState } from "react";

type Props = {
  url: string;
  caption?: string;
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
};

export default function ImageViewer({ url, caption, onClose, onPrev, onNext }: Props) {
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const dragRef = useRef({ down: false, sx: 0, sy: 0, px: 0, py: 0 });
  const stageRef = useRef<HTMLDivElement>(null);

  // ESC closes; arrow keys nav.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowLeft" && onPrev) onPrev();
      else if (e.key === "ArrowRight" && onNext) onNext();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose, onPrev, onNext]);

  // Block page scroll while the viewer is open.
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, []);

  // Reset zoom/pan when the URL changes (different image).
  useEffect(() => { setScale(1); setPan({ x: 0, y: 0 }); }, [url]);

  // Native wheel listener with passive:false so preventDefault works.
  // React's synthetic onWheel is passive on modern browsers (won't stop scroll).
  useEffect(() => {
    const node = stageRef.current;
    if (!node) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = node.getBoundingClientRect();
      const cx = e.clientX - rect.left - rect.width / 2;
      const cy = e.clientY - rect.top - rect.height / 2;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      setScale((s) => {
        const next = Math.max(0.1, Math.min(20, s * factor));
        // Pivot zoom at the cursor: shift pan so the point under the cursor
        // stays under the cursor after the scale change.
        setPan((p) => ({
          x: cx - (cx - p.x) * (next / s),
          y: cy - (cy - p.y) * (next / s),
        }));
        return next;
      });
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel as EventListener);
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    dragRef.current = { down: true, sx: e.clientX, sy: e.clientY, px: pan.x, py: pan.y };
  }, [pan]);
  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current.down) return;
    const dx = e.clientX - dragRef.current.sx;
    const dy = e.clientY - dragRef.current.sy;
    setPan({ x: dragRef.current.px + dx, y: dragRef.current.py + dy });
  }, []);
  const stopDrag = useCallback(() => { dragRef.current.down = false; }, []);

  function reset() { setScale(1); setPan({ x: 0, y: 0 }); }

  return (
    <div className="viewer-overlay" onClick={onClose}>
      <div className="viewer-toolbar" onClick={(e) => e.stopPropagation()}>
        <button onClick={() => setScale((s) => Math.min(20, s * 1.25))} title="Zoom in">+</button>
        <button onClick={() => setScale((s) => Math.max(0.1, s / 1.25))} title="Zoom out">−</button>
        <button onClick={reset} title="Reset to 100% (double-click image also works)">1:1</button>
        <span className="muted small">{(scale * 100).toFixed(0)}%</span>
        {caption && <span className="muted small viewer-caption">· {caption}</span>}
        <div className="spacer" />
        {onPrev && <button onClick={onPrev} title="Previous (←)">‹</button>}
        {onNext && <button onClick={onNext} title="Next (→)">›</button>}
        <button onClick={onClose} className="x" title="Close (Esc)">✕</button>
      </div>
      <div
        ref={stageRef}
        className="viewer-stage"
        onClick={(e) => e.stopPropagation()}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={stopDrag}
        onMouseLeave={stopDrag}
        onDoubleClick={reset}
        style={{ cursor: dragRef.current.down ? "grabbing" : (scale > 1 ? "grab" : "default") }}
      >
        <img
          src={url}
          alt={caption ?? ""}
          draggable={false}
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`,
            transformOrigin: "center center",
          }}
        />
      </div>
    </div>
  );
}
