import { useRef, useState } from "react";
import { Spinner } from "@components/basic/Spinner";
import { TopKPopup } from "./TopKPopup";
import type { PredictionEntry } from "@/utils/predictionHelpers";

interface PredictionItemContext {
  entry: PredictionEntry;
}

// A single prediction response. Shows dots if in progress.
export function PredictionItem({ entry }: PredictionItemContext) {
  const [showTopK, setShowTopK] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null); // "parent" button

  if (entry.pending) {
    return (
      <div className="prediction-item prediction-item--pending">
        <Spinner />
      </div>
    );
  }

  const confidencePct = (entry.top1.confidence * 100).toFixed(1);

  return (
    <div className="prediction-item-wrap">
      <button
        type="button"
        className={`prediction-item${entry.isUnknown ? " prediction-item--unknown" : ""}`}
        title={`Confidence: ${confidencePct}%`} // hover confidence
        ref={buttonRef}
        onClick={() => setShowTopK((v) => !v)}
      >
        {entry.isUnknown ? "unknown" : entry.top1.label}
      </button>
      {showTopK && (
        <TopKPopup 
          topk={entry.topk} 
          anchorRef={buttonRef}
          onClose={() => setShowTopK(false)} 
        />
      )}
    </div>
  );
}
