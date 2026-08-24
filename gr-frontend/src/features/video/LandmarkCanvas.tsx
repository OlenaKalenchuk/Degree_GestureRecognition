import { useEffect, useRef } from "react";
import { computeContentRect, drawLandmarkFrame } from "@/utils/landmarkConnections";
import type { LandmarkFrame } from "@/types/api";


interface LandmarkCanvasContext {
  landmarks: LandmarkFrame | null;
  targetRef: React.RefObject<HTMLVideoElement>; // main video element, used as a render reference
  visible: boolean;
}

// Landmark overlay on top of the video element.
// May be a bit off, depending on the video resolution and aspect ratio
export function LandmarkCanvas({ landmarks, targetRef, visible }: LandmarkCanvasContext) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Canvas pixels <-> Video rendered size
  useEffect(() => {
    const target = targetRef.current;
    const canvas = canvasRef.current;
    if (!target || !canvas) return;

    const resize = () => {
      canvas.width = target.clientWidth;
      canvas.height = target.clientHeight;
    };
    resize();

    const observer = new ResizeObserver(resize);
    observer.observe(target);
    return () => observer.disconnect();
  }, [targetRef]);

  // Redraw when a landmark frame arrives
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = targetRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    if (!visible || !landmarks) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const contentRect = computeContentRect(
      canvas.width,
      canvas.height,
      video?.videoWidth ?? 0,
      video?.videoHeight ?? 0,
    );
    drawLandmarkFrame(ctx, landmarks, canvas.width, canvas.height, contentRect);
  }, [landmarks, visible, targetRef]);

  return <canvas ref={canvasRef} className="landmark-canvas" />;
}

