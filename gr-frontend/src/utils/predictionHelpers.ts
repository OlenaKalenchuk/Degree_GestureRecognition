import type { GesturePrediction, PredictResponse, SequencePrediction, WindowPrediction } from "@/types/api";

export const UNKNOWN_CONFIDENCE_THRESHOLD = 0.1; // confidence, at which the label will be marked as "unknown"

// entry union wrapper for different tasks
export type PredictionEntry = // do not confuse with PredictionItem, which uses this
  | {kind: "window"; id: string; data: WindowPrediction } // Multi-class
  | {kind: "sequence"; id: string; data: SequencePrediction } // CTC
  | {kind: "pending"; id: string }; // "..."
export type WindowEntry = Extract<PredictionEntry, { kind: "window" }>;
export type SequenceEntry = Extract<PredictionEntry, { kind: "sequence" }>;

let entryCounter = 0;

// generate entry id as 36-char string
export function nextEntryId(): string {
  entryCounter += 1;
  return typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `entry-${entryCounter}`;
}

export function isUnknownPrediction(top1: GesturePrediction): boolean {
  return top1.confidence < UNKNOWN_CONFIDENCE_THRESHOLD;
}

export function printConfidence(prediction: GesturePrediction): string {
  const confidence = (prediction.confidence * 100).toFixed(1);
  return `${confidence}`
}


// "..." pending entry
const isPending = (e: PredictionEntry) => e.kind === "pending";

export function makePendingEntry(): PredictionEntry {
  return { kind: "pending", id: nextEntryId() };
}
export function appendPending(entries: PredictionEntry[]): PredictionEntry[] {
  return entries.some(isPending) ? entries : [...entries, makePendingEntry()];
}
export function removePending(entries: PredictionEntry[]): PredictionEntry[] {
  return entries.some(isPending) ? entries.filter((e) => !isPending(e)) : entries;
}

// Convert Window/Sequence Prediction into PredictionEntry
export function responseToEntry(response: PredictResponse): PredictionEntry[] {
  switch (response.task) {
    case "classification":
      return response.windows.map((data) => ({ kind: "window", id: nextEntryId(), data }));
    case "ctc":
      return response.sequences
        .filter((s) => s.text.length > 0 || s.tokens.length > 0) // nothing was decoded -> don't show an empty chip
        .map((data) => ({ kind: "sequence", id: nextEntryId(), data }));
  }
}