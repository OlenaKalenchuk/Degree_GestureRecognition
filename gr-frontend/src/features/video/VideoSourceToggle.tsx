export type VideoSource = "upload" | "camera";

interface VideoSourceToggleContext { // don't like the long name
  source: VideoSource;
  onChange: (source: VideoSource) => void;
  disabled?: boolean;
}

export function VideoSourceToggle({ source, onChange, disabled }: VideoSourceToggleContext) {
  return (
    <div className="source-toggle" role="tablist" aria-label="Video source">
      <button
        type="button"
        role="tab"
        aria-selected={source === "upload"}
        className={`source-toggle-btn${source === "upload" ? " source-toggle-btn--active" : ""}`}
        onClick={() => onChange("upload")}
        disabled={disabled}
        >
        Upload video
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={source === "camera"}
        className={`source-toggle-btn${source === "camera" ? " source-toggle-btn--active" : ""}`}
        onClick={() => onChange("camera")}
        disabled={disabled}
        >
        Camera
      </button>
    </div>
  );
}
