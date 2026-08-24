import io
from api.config import FRAME_NORM_MEAN, FRAME_NORM_STD, FRAME_SIZE, FRAME_TRANSFORM, MAX_CLIP_FRAMES
import numpy as np
import cv2
import torch

from api.mediapipe_detection.LandmarkExtractor import LandmarkExtractor
from api.schemas import WindowPrediction
from api.video_processing.landmark_preprocess import pack_landmarks


# FRAME READING ---

def read_frame_bytes(video_path: str) -> tuple[list, float]:
    """
    Decode every frame of a video supplied as raw bytes, return FPS.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames, fps


def _sample_frames(frames: list[np.ndarray], sample_count: int = 32) -> list[np.ndarray]:
    """
    For managing video frames. Sample frames uniformly (each `n` frame).
    Returns a list of frame paths.
    """
    frames_total = len(frames)
    if frames_total == 0:
        raise ValueError("No frames to sample from.")
    # check for less frames just in case
    if frames_total < sample_count:
        # pad last frame
        indices = list(range(frames_total)) + [frames_total - 1] * (sample_count - frames_total)
    else:
        indices = np.linspace(0, frames_total - 1, sample_count, dtype=int)
        
    return [frames[i] for i in indices]



# FRAME CONVERTING & WINDOWS ---

def preprocess_frame(frame_bgr: np.ndarray):
    """Resize, convert RGB, normalize frame bytes. Returns: (H, W, C)."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError( f"Preprocessing expects a (H, W, 3) BGR frame, got: {frame_bgr.shape}.")
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return FRAME_TRANSFORM(frame)


def extract_windows(
    frames: list[np.ndarray],
    extractor: LandmarkExtractor,
    window_size: int,
    stride: int,
    min_frames: int
    ):
    """
    Split a video into overlapping clips using a sliding window.
    Window is made based on `min_frames`, `window_size` and the given stride. Stride may be overlapping (stride == n-frames == no overlap)
    """
    if window_size <= 0:
        raise ValueError("Window_size must be greater than 0.")
    if stride <= 0 or stride > window_size:
        raise ValueError(f"Stride must be greater than 0 and less than window_size: {stride}.")

    windows: list[dict] = []
    n_frames = len(frames)

    if n_frames < min_frames:
        return windows

    extractor.reset_stream()

    # Extract landmarks
    processed_frames: list[torch.Tensor] = []
    processed_landmarks: list[np.ndarray] = []
    for frame in frames:
        landmarks, _ = extractor.detect_frame_landmarks(frame)
        processed_frames.append(preprocess_frame(frame))
        processed_landmarks.append(pack_landmarks(landmarks))

    
    start = 0
    # Make frame-windows
    while start < n_frames:
        end = min(start + window_size, n_frames)
        length = end - start

        # Handle trailing fragment
        if length >= min_frames:
            windows.append({
                "frames": processed_frames[start:end],
                "landmarks": processed_landmarks[start:end],
                "frame_indices": list(range(start, end)),
            })
        if end >= n_frames:
            break

        start += stride

    return windows


def merge_window_preds(
    windows: list[WindowPrediction],
    min_confidence: float = 0.0,
    ) -> list[WindowPrediction]:
    """
    Merge repeated, consecutive predictions from overlapping windows. Can be set min confidence to filter out "garbage" windows.
    """
    merged: list[WindowPrediction] = []

    for window in windows:
        confidence = window.top1.confidence
        if confidence < min_confidence:
            continue
        if not merged:
            merged.append(window)
            continue

        previous = merged[-1]
        if window.top1.class_index == previous.top1.class_index:
            if window.top1.confidence > previous.top1.confidence:
                merged[-1] = window
        else:
            merged.append(window)
    return merged