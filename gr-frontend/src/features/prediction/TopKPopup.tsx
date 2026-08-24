import { useEffect, useRef, useState } from "react";
import type { GesturePrediction } from "@/types/api";
import { createPortal } from "react-dom";

interface TopKPopupContext {
  topk: GesturePrediction[];
  anchorRef: React.RefObject<HTMLButtonElement>;
  onClose: () => void;
}

// A on-click popup with prediction's top-3 confidences.
export function TopKPopup({ topk, anchorRef, onClose }: TopKPopupContext) {
  const ref = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ top: 0, left: 0 });

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node) && !anchorRef.current?.contains(e.target)) onClose();
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [onClose, anchorRef]);

  useEffect(() => {
    const updatePosition = () => {
      const anchor = anchorRef.current;
      if (!anchor) return;
      const rect = anchor.getBoundingClientRect();

      setPosition({ top: rect.bottom + 6, left: rect.left });
    };
    updatePosition();
    window.addEventListener("scroll", updatePosition, true);
    window.addEventListener("resize", updatePosition);

    return () => {
      window.removeEventListener("scroll", updatePosition, true);
      window.removeEventListener("resize", updatePosition);
    };
  }, [anchorRef]);

  return createPortal(
    <div ref={ref} className="topk-popup" style={{ position: "fixed", top: position.top, left: position.left}}>
      <span className="topk-popup-title">Top 3</span>
      <ul>
        {topk.map((p, i) => (
          <li key={p.class_index}>
            <span className="topk-rank">{i + 1}</span>
            <span className="topk-label">{p.label}</span>
            <span className="topk-bar-track">
              <span className="topk-bar-fill" style={{ width: `${p.confidence * 100}%` }} />
            </span>
            <span className="topk-confidence">{(p.confidence * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>,
    document.body
  );
}
