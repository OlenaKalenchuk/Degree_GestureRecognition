import { useRef, useState } from "react";
import { Spinner } from "@components/basic/Spinner";
import { TopKPopup } from "./TopKPopup";
import { isUnknownPrediction, printConfidence, type PredictionEntry, type SequenceEntry, type WindowEntry } from "@/utils/predictionHelpers";

interface PredictionItemContext {
  entry: PredictionEntry;
}

// A single prediction response. Shows dots if in progress.
export function PredictionItem({ entry }: PredictionItemContext) {
  switch (entry.kind) {
    case "pending":
      return (
        <div className="prediction-item prediction-item--pending">
          <Spinner />
        </div>
      );

    case "window":
      return <WindowItem entry={entry} />;
    case "sequence":
      return <SequenceItem entry={entry} />;
  }
}
// Top-k window prediction
function WindowItem({ entry }: { entry: WindowEntry }) {
  const [showTopK, setShowTopK] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null); // "parent" button
  const {top1, top3} = entry.data;
  const isUnknown = isUnknownPrediction(top1);
  const confidencePct = printConfidence(top1)

  return (
    <div className="prediction-item-wrap">
      <button
        type="button"
        className={`prediction-item${isUnknown ? " prediction-item--unknown" : ""}`}
        title={`Confidence: ${confidencePct}%`} // hover confidence
        ref={buttonRef}
        onClick={() => setShowTopK((v) => !v)}
      >
        {isUnknown ? "unknown" : top1.label}
      </button>
      {showTopK && (
        <TopKPopup 
          topk={top3} 
          anchorRef={buttonRef}
          onClose={() => setShowTopK(false)} 
        />
      )}
    </div>
  );
}

// CTC Sequence prediction
function SequenceItem({ entry }: { entry: SequenceEntry }) {
  const [showTopK, setShowTopK] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null); // "parent" button
  const {text, tokens} = entry.data;

  return (
    <div className="prediction-item-wrap">
      <button
        type="button"
        className={`prediction-item`}
        ref={buttonRef}
        onClick={() => setShowTopK((v) => !v)}
      >
        {text}
      </button>
      {showTopK && (
        <TopKPopup 
          topk={tokens} 
          anchorRef={buttonRef}
          onClose={() => setShowTopK(false)} 
          title="Sequence"
        />
      )}
    </div>
  );
}
