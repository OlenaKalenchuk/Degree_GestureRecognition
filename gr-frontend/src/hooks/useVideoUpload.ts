import { useCallback, useEffect, useRef, useState } from "react";
import { predictFile } from "@features/prediction/predict";
import { makePendingEntry, responseToEntry, type PredictionEntry } from "@/utils/predictionHelpers";
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
  const requestIdRef = useRef(0);

  const reset = useCallback(() => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = null;
    requestIdRef.current += 1;
    setVideoUrl(null);
    setFileName(null);
    setEntries([]);
    setLandmarks(null);
    setError(null);
    setIsPredicting(false);
  }, []);

  const selectFile = useCallback((file: File) => {
    reset();
    const requestId = requestIdRef.current;
    const isOutdated = () => requestId !== requestIdRef.current; // if current is not the newest request

    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setVideoUrl(url);
    setFileName(file.name);

    // the "..."
    setEntries([makePendingEntry()]);
    setIsPredicting(true);

    predictFile(file)
      .then((response) => {
        if (isOutdated()) return;
        setEntries(responseToEntry(response));
        setLandmarks(response.landmarks ?? null);
      })
      .catch((err) => {
        if (isOutdated()) return;
        setError(err instanceof Error ? err.message : "Prediction failed.");
        setEntries([]);
      })
      .finally(() => {
        if (!isOutdated()) setIsPredicting(false);
      });
  }, [reset]);

  // Revoke the object URL on unmount.
  useEffect(() => () => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
  }, []);

  return { videoUrl, fileName, entries, landmarks, isPredicting, error, selectFile, reset };
}
