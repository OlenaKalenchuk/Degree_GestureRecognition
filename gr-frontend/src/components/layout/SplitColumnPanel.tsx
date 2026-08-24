import type { ReactNode } from "react";
import { useResizePanels } from "../../hooks/useResizePanels";

interface SplitPanelContent {
  left: ReactNode;
  right: ReactNode;
}

export function SplitPanel({ left, right }: SplitPanelContent) {
  const { containerRef, mainWidthPct, isDragging, onDragStart } = useResizePanels();

  return (
    <div ref={containerRef} className={`split-panel${isDragging ? " split-panel--dragging" : ""}`}>
      <div className="split-panel-left" style={{ width: `${mainWidthPct}%` }}>
        {left}
      </div>
      <div className="split-panel-divider" onPointerDown={onDragStart} />
      <div className="split-panel-right" style={{ width: `${100 - mainWidthPct}%` }}>
        {right}
      </div>
    </div>
  );
}
