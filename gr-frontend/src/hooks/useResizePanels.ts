import { useCallback, useEffect, useRef, useState } from "react";

const MIN_PERCENT = 50;
const MAX_PERCENT = 80;
const DEFAULT_PERCENT = 60;

interface UseResizePanelsResult {
  containerRef: React.RefObject<HTMLDivElement>;
  leftWidthPct: number;
  isDragging: boolean;
  onDragStart: (e: React.PointerEvent) => void;
}

// Tracks the left (main) panel's width as a percentage of the container
export function useResizePanels(): UseResizePanelsResult {
  const containerRef = useRef<HTMLDivElement>(null); // container
  const [mainWidthPct, setMainWidthPct] = useState(DEFAULT_PERCENT); // percentage
  const [isDragging, setIsDragging] = useState(false); // dragging state

  const clamp = (v: number) => Math.min(MAX_PERCENT, Math.max(MIN_PERCENT, v));

  const onPointerMove = useCallback((e: PointerEvent) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const pct = ((e.clientX - rect.left) / rect.width) * 100;
    setMainWidthPct(clamp(pct));
  }, []);

  const onPointerUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (!isDragging) return;
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
    };
  }, [isDragging, onPointerMove, onPointerUp]);

  const onDragStart = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  return { containerRef, mainWidthPct, isDragging, onDragStart };
}
