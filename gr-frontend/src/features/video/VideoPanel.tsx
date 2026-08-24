import { useEffect, useRef, useState } from "react";
import { VideoSourceToggle, type VideoSource } from "./VideoSourceToggle";
import { VideoDisplay } from "./VideoDisplay";
import { VideoControls } from "./VideoControls";
import { useVideoUpload } from "@/hooks/useVideoUpload";
import { useCameraSession } from "@/hooks/useCameraSession";
import { useAppContext } from "@/core/AppContext";
import type { PredictionEntry } from "@utils/predictionHelpers";

interface VideoPanelContext {
  onEntriesChange: (entries: PredictionEntry[]) => void;
}

export function VideoPanel({ onEntriesChange }: VideoPanelContext) {
  const [source, setSource] = useState<VideoSource>("upload");
  const [overlayEnabled, setOverlayEnabled] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { activeModelId } = useAppContext();

  const upload = useVideoUpload();
  const camera = useCameraSession();

  // Stop the camera stream/upload in the background
  const handleSourceChange = (next: VideoSource) => {
    if (source === "camera" && next !== "camera") camera.stop();
    if (source === "upload" && next !== "upload") upload.reset();
    setSource(next);
  };

  // Tells PredictionPanel if predictions are active or not
  useEffect(() => {
    onEntriesChange(source === "camera" ? camera.entries : upload.entries);
  }, [source, camera.entries, upload.entries, onEntriesChange]);


  const activeError = source === "camera" ? camera.error : upload.error;
  return (
    <div className="video-panel">
      <div className="video-panel-toolbar">
        <VideoSourceToggle source={source} onChange={handleSourceChange} />
        {source === "upload" && (
          <>
            <button
              type="button"
              className="btn btn--primary"
              disabled={!activeModelId || upload.isPredicting}
              onClick={() => fileInputRef.current?.click()}
              >
              {upload.isPredicting ? "Processing…" : "Choose video"}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              hidden
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) upload.selectFile(file);
                e.target.value = "";
              }}
              />
            {upload.fileName && <span className="video-panel-filename">{upload.fileName}</span>}
          </>
        )}
        {source === "camera" && (
          <button
            type="button"
            className={`btn ${camera.isActive ? "btn--danger" : "btn--primary"}`}
            disabled={!activeModelId || camera.isConnecting}
            onClick={() => (camera.isActive ? camera.stop() : camera.start())}
            >
            {camera.isConnecting ? "Connecting…" : camera.isActive ? "Stop camera" : "Start camera"}
          </button>
        )}
      </div>

      {!activeModelId && <p className="video-panel-hint">Select a model above for prediction.</p>}
      {activeError && <p className="video-panel-error">{activeError}</p>}

      <VideoDisplay
        mode={source === "camera" ? (camera.isActive ? "camera" : "empty") : upload.videoUrl ? "upload" : "empty"}
        cameraVideoRef={camera.videoRef}
        cameraActive={camera.isActive}
        uploadVideoUrl={upload.videoUrl}
        landmarks={source === "camera" ? camera.landmarks : upload.landmarks?.[0] ?? null}
        overlayEnabled={overlayEnabled}
        showRecognitionBadge={source === "camera" && camera.isActive && !camera.recognitionEnabled}
        />

      <VideoControls
        source={source}
        recognitionEnabled={camera.recognitionEnabled}
        onRecognitionChange={camera.toggleRecognition}
        overlayEnabled={overlayEnabled}
        onOverlayChange={setOverlayEnabled}
        overlayAvailable={source === "camera" || !!upload.landmarks}
        />
    </div>
  );
}
