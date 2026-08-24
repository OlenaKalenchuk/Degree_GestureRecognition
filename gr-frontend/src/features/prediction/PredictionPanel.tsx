import { useEffect, useRef } from "react";
import { PredictionItem } from "./PredictionItem";
import type { PredictionEntry } from "@/utils/predictionHelpers";

interface PredictionPanelContext {
  entries: PredictionEntry[];
}

// A history panel with recognized gestures.
// Holds PredictionItem-s in it
export function PredictionPanel({ entries }: PredictionPanelContext) {
  const listRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [entries.length]);

  return (
    <div className="prediction-panel">
      <h2 className="prediction-panel-title">Predictions</h2>
      <div ref={listRef} className="prediction-list">
        {entries.length === 0 && <p className="prediction-list-empty">No predictions yet.</p>}
        {entries.map((entry) => (
          <PredictionItem key={entry.id} entry={entry} />
        ))}
      </div>
    </div>
  );
}
