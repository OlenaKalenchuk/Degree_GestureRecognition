import { useMemo, useRef } from "react";
import { LandmarkCanvas } from "./LandmarkCanvas";
import type { LandmarkFrame } from "@/types/api";

interface VideoDisplayContext {
  mode: "camera" | "upload" | "empty";
  cameraVideoRef?: React.RefObject<HTMLVideoElement>;
  cameraActive?: boolean // it exists without the cam, but when the cam tab is selected
  uploadVideoUrl?: string | null;
  landmarks: LandmarkFrame | null;
  overlayEnabled: boolean;
  showRecognitionBadge: boolean; // "No recognition"
  emptyLabel?: string;
}

// Hooks camera to the local render 
function mergeRefs<T>(a: React.RefObject<T>, b: React.MutableRefObject<T | null>) {
  return (node: T | null) => {
    (a as React.MutableRefObject<T | null>).current = node;
    b.current = node;
  };
}

export function VideoDisplay({
  mode,
  cameraVideoRef,
  cameraActive = false,
  uploadVideoUrl,
  landmarks,
  overlayEnabled,
  showRecognitionBadge,
  emptyLabel = "Upload a video or start the camera",
}: VideoDisplayContext) {
  const cameraLocalRef = useRef<HTMLVideoElement>(null);
  const uploadLocalRef = useRef<HTMLVideoElement>(null);
  const mergedCameraRef = useMemo(
    () => (cameraVideoRef ? mergeRefs(cameraVideoRef, cameraLocalRef) : cameraLocalRef),
    [cameraVideoRef],
  );

  return (
    <div className={`video-stage${mode === "camera" ? " video-stage--mirrored" : ""}`}>
      <video
        ref={mergedCameraRef}
        className="video-stage-el"
        style={{ display: mode === "camera" ? "block" : "none" }}
        autoPlay
        playsInline
        muted
      />
      {mode === "camera" && !cameraActive && (
        <div className="video-stage-empty video-stage-empty--overlay">Press "Start camera" to begin</div>
      )}
      {mode === "upload" && uploadVideoUrl && (
        <video ref={uploadLocalRef} className="video-stage-el" src={uploadVideoUrl} controls />
      )}
      {mode === "empty" && <div className="video-stage-empty">{emptyLabel}</div>}
    
      {mode !== "empty" && (
        <LandmarkCanvas landmarks={landmarks} targetRef={mode === "camera" ? cameraLocalRef : uploadLocalRef} visible={overlayEnabled} />
      )}

      {showRecognitionBadge && <div className="no-recognition-badge">&gt; No recognition</div>}
    </div>
  );
}
