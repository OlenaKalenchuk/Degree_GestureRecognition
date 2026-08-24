import { Switch } from "@components/basic/Switch";
import type { VideoSource } from "./VideoSourceToggle";

interface VideoControlsContext {
  source: VideoSource;
  recognitionEnabled: boolean;
  onRecognitionChange: (enabled: boolean) => void;
  overlayEnabled: boolean;
  onOverlayChange: (enabled: boolean) => void;
  overlayAvailable: boolean; // TODO: false for video uploads for now
}

export function VideoControls({
  source,
  recognitionEnabled,
  onRecognitionChange,
  overlayEnabled,
  onOverlayChange,
  overlayAvailable,
}: VideoControlsContext) {
  return (
    <div className="video-controls">
      {source === "camera" && (
        <Switch label="Recognition" checked={recognitionEnabled} onChange={onRecognitionChange} />
      )}
      <Switch
        label="Landmark overlay"
        checked={overlayEnabled}
        onChange={onOverlayChange}
        disabled={!overlayAvailable}
        title={overlayAvailable ? undefined : "Uploaded-video landmarks aren't returned by the API yet"}
      />
    </div>
  );
}
