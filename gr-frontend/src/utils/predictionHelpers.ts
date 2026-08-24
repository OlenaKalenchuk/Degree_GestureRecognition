import type { GesturePrediction, WindowPrediction } from "@/types/api";

export const UNKNOWN_CONFIDENCE_THRESHOLD = 0.1; // confidence, at which the label will be marked as "unknown"

export interface PredictionEntry { // do not confuse with PredictionItem, which uses this
  id: string;
  top1: GesturePrediction;
  topk: GesturePrediction[];
  isUnknown: boolean;
  startTime?: number | null;
  endTime?: number | null;
  pending?: boolean; // "..."
}

let entryCounter = 0;
// generate entry id as 36-char string
export function nextEntryId(): string {
  entryCounter += 1;
  return typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `entry-${entryCounter}`;
}

export function isUnknownPrediction(top1: GesturePrediction): boolean {
  return top1.confidence < UNKNOWN_CONFIDENCE_THRESHOLD;
}

// Convert WindowPrediction and GesturePrediction into PredictionEntry
export function windowToEntry(window: WindowPrediction): PredictionEntry {
  return {
    id: nextEntryId(),
    top1: window.top1,
    topk: window.top3,
    isUnknown: isUnknownPrediction(window.top1),
    startTime: window.start_time,
    endTime: window.end_time,
  };
}

export function predictionToEntry(top1: GesturePrediction, topk: GesturePrediction[]): PredictionEntry {
  return {
    id: nextEntryId(),
    top1,
    topk,
    isUnknown: isUnknownPrediction(top1),
  };
}