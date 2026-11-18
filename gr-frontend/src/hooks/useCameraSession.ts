import { useCallback, useEffect, useRef, useState } from "react";
import { getWsBaseUrl } from "@features/api/client";
import { appendPending, removePending, responseToEntry, type PredictionEntry } from "@/utils/predictionHelpers";
import type { HandsPresent, LandmarkFrame, WebSockResponse } from "@/types/api";

// Frames-per-second sent to the server.
const CAPTURE_FPS = 12;
const JPEG_QUALITY = 0.7;

interface UseCameraSessionResult {
  videoRef: React.RefObject<HTMLVideoElement>;
  isActive: boolean;
  isConnecting: boolean;
  recognitionEnabled: boolean;
  handsPresent: HandsPresent;
  landmarks: LandmarkFrame | null;
  entries: PredictionEntry[];
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
  toggleRecognition: (enabled: boolean) => void;
}

export function useCameraSession(): UseCameraSessionResult {
  // page element references
  const videoRef = useRef<HTMLVideoElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureIntervalRef = useRef<number | null>(null);
  const hasPendingIconRef = useRef(false);

  // toggle states
  const [isActive, setIsActive] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [recognitionEnabled, setRecognitionEnabled] = useState(true);
  const recognitionEnabledRef = useRef(true); // per-connection fix
  const [handsPresent, setHandsPresent] = useState<HandsPresent>({ left: false, right: false });
  const [landmarks, setLandmarks] = useState<LandmarkFrame | null>(null);
  const [entries, setEntries] = useState<PredictionEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  const captureAndSendFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    const ws = wsRef.current;
    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN || video.videoWidth === 0) {
      if (Date.now() % 1000 < (1000 / CAPTURE_FPS)) {
        console.debug("CameraSession: skipping frame capture", {
          hasVideo: !!video,
          hasCanvas: !!canvas,
          wsReadyState: ws?.readyState,
          videoWidth: video?.videoWidth,
        });
      }
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // TODO: change to canvas.toBlob() for high fps?
    const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    const base64 = dataUrl.split(",")[1];
    ws.send(JSON.stringify({ frame: base64 }));
  }, []);


  const stop = useCallback(() => {
    if (captureIntervalRef.current !== null) {
      window.clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    wsRef.current?.close();
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setIsActive(false);
    setIsConnecting(false);
    setLandmarks(null);
    setHandsPresent({ left: false, right: false });
    setEntries(removePending); 
  }, []);

  
  const start = useCallback(async () => {
    setError(null);
    setIsConnecting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;

      if (!videoRef.current) {
        console.error("CameraSession: camera <video> element isnt ready yet, but got requested as videoRef.current.");
        throw new Error("Camera preview element isn't ready yet.");
      }


      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        console.info(`CameraSession: video metadata loaded (${videoRef.current?.videoWidth}x${videoRef.current?.videoHeight})`);
      };

      // Landmark requests are always enabled. The "show overlay" switch just disables the overlay, but not the requests.
      const ws = new WebSocket(`${getWsBaseUrl()}/predict/camera?enable_landmarks=true`);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnecting(false);
        setIsActive(true);
        captureCanvasRef.current = document.createElement("canvas");
        captureIntervalRef.current = window.setInterval(captureAndSendFrame, 1000 / CAPTURE_FPS);
      };

      ws.onmessage = (event) => {
        const msg: WebSockResponse = JSON.parse(event.data);
        if (msg.error) {
          setError(msg.error);
          return;
        }
        if (msg.hands_present) setHandsPresent(msg.hands_present);
        else return;
        if (msg.landmarks) setLandmarks(msg.landmarks);

        const anyHandPresent = msg.hands_present.left || msg.hands_present.right;

        if (msg.prediction) {
          const msgEntries = responseToEntry(msg.prediction);
          setEntries((prev) => [...removePending(prev), ...msgEntries]);

        } else if (anyHandPresent && recognitionEnabledRef.current) {
          // Show a pending icon when recognition is on and working
          setEntries(appendPending);
        
        } else if (!anyHandPresent && hasPendingIconRef.current) {
          // Remove a pending icon
          setEntries(removePending);
        }
      };

      ws.onerror = () => setError("WebSocket connection error.");
      ws.onclose = () => setIsActive(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not access the camera.");
      setIsConnecting(false);
      stop();
    }
  }, [stop, captureAndSendFrame]);

  const toggleRecognition = useCallback((enabled: boolean) => {
    setRecognitionEnabled(enabled);
    recognitionEnabledRef.current = enabled;
    if (!enabled) setEntries(removePending);
    wsRef.current?.send(JSON.stringify({ recognition: enabled }));
  }, []);

  useEffect(() => () => stop(), [stop]);

  return {
    videoRef,
    isActive,
    isConnecting,
    recognitionEnabled,
    handsPresent,
    landmarks,
    entries,
    error,
    start,
    stop,
    toggleRecognition
  };
}
