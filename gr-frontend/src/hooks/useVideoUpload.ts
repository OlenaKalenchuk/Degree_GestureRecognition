import { useCallback, useEffect, useRef, useState } from "react";
import { predictFile } from "@features/prediction/predict";
import { nextEntryId, windowToEntry, type PredictionEntry } from "@/utils/predictionHelpers";
import type { LandmarkFrame } from "@/types/api";

interface UseVideoUploadResult {
  videoUrl: string | null;
  fileName: string | null;
  entries: PredictionEntry[];
  landmarks: LandmarkFrame[] | null;
  isPredicting: boolean;
  error: string | null;
  selectFile: (file: File) => void;
  reset: () => void;
}

// Handles per-video prediction: POST /predict/file
export function useVideoUpload(): UseVideoUploadResult {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [entries, setEntries] = useState<PredictionEntry[]>([]);
  const [landmarks, setLandmarks] = useState<LandmarkFrame[] | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const reset = useCallback(() => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = null;
    setVideoUrl(null);
    setFileName(null);
    setEntries([]);
    setLandmarks(null);
    setError(null);
  }, []);

  const selectFile = useCallback((file: File) => {
    reset();
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setVideoUrl(url);
    setFileName(file.name);

    // the "..."
    const pendingId = nextEntryId();
    setIsPredicting(true);
    setEntries([{ id: pendingId, top1: { label: "", class_index: -1, confidence: 0 }, topk: [], isUnknown: false, pending: true }]);

    predictFile(file)
      .then((response) => {
        setEntries(response.windows.map(windowToEntry));
        setLandmarks(response.landmarks ?? null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Prediction failed.");
        setEntries([]);
      })
      .finally(() => setIsPredicting(false));
  }, [reset]);

  // Revoke the object URL on unmount.
  useEffect(() => () => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
  }, []);

  return { videoUrl, fileName, entries, landmarks, isPredicting, error, selectFile, reset };
}
