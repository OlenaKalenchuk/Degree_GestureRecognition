import type { LandmarkFrame } from "@/types/api";

// MediaPipe 21-point hand
const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], // thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // index
  [5, 9], [9, 10], [10, 11], [11, 12], // middle
  [9, 13], [13, 14], [14, 15], [15, 16], // ring
  [13, 17], [17, 18], [18, 19], [19, 20], // pinky
  [0, 17], // palm base
];

// Mediapipe pose: [11, 12, 13, 14, 15, 16] shoulders, elbows, wrists
const POSE_CONNECTIONS: [number, number][] = [
  [0, 1], // shoulders
  [0, 2], [2, 4], // left arm
  [1, 3], [3, 5], // right arm
];

interface DrawOptions {
  color?: string;
  pointRadius?: number;
  lineWidth?: number;
}

// for video-canvas mapping
export interface ContentRect {
  offsetX: number;
  offsetY: number;
  width: number;
  height: number;
}

// Draws x, y [0,1] coordinates and connecting edges.
function drawLandmarkSet(
  ctx: CanvasRenderingContext2D,
  points: number[][],
  connections: [number, number][],
  rect: ContentRect,
  { color = "#4fa3d9", pointRadius = 3, lineWidth = 2 }: DrawOptions,
) {
  const isValid = (p: number[] | undefined) => !!p && !(p[0] === 0 && p[1] === 0);
  const toX = (nx: number) => rect.offsetX + nx * rect.width;
  const toY = (ny: number) => rect.offsetY + ny * rect.height;

  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lineWidth;

  for (const [a, b] of connections) {
    const pa = points[a];
    const pb = points[b];
    if (!isValid(pa) || !isValid(pb)) continue;

    ctx.beginPath();
    ctx.moveTo(toX(pa[0]), toY(pa[1]));
    ctx.lineTo(toX(pb[0]), toY(pb[1]));
    ctx.stroke();
  }

  for (const p of points) {
    if (!isValid(p)) continue;
    ctx.beginPath();
    ctx.arc(toX(p[0]), toY(p[1]), pointRadius, 0, Math.PI * 2);
    ctx.fill();
  }
}

// Draws new landmarks on a cleaned frame
export function drawLandmarkFrame(
  ctx: CanvasRenderingContext2D,
  frame: LandmarkFrame,
  canvasWidth: number,
  canvasHeight: number,
  contentRect: ContentRect,
) {
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  // draw pose and hands
  drawLandmarkSet(ctx, frame.pose, POSE_CONNECTIONS, contentRect, { color: "#8a97a6", pointRadius: 4 });
  drawLandmarkSet(ctx, frame.left_hand, HAND_CONNECTIONS, contentRect, { color: "#4fa3d9" });
  drawLandmarkSet(ctx, frame.right_hand, HAND_CONNECTIONS, contentRect, { color: "#2f6690" });
}

// computes the displayed area within video's resolution
export function computeContentRect(
  containerWidth: number,
  containerHeight: number,
  videoWidth: number,
  videoHeight: number,
): ContentRect {
  if (!videoWidth || !videoHeight || !containerWidth || !containerHeight) {
    return { offsetX: 0, offsetY: 0, width: containerWidth, height: containerHeight };
  }
  const containerRatio = containerWidth / containerHeight;
  const videoRatio = videoWidth / videoHeight;

  // Video is wider
  if (videoRatio > containerRatio) {
    const width = containerWidth;
    const height = width / videoRatio;
    return { offsetX: 0, offsetY: (containerHeight - height) / 2, width, height };
  }
  // Video is longer
  const height = containerHeight;
  const width = height * videoRatio;
  return { offsetX: (containerWidth - width) / 2, offsetY: 0, width, height };
}