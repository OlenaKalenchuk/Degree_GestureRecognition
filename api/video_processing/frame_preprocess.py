from collections import deque
import io
from pathlib import Path
from typing import Iterable, Iterator, Optional
from api.config import FRAME_NORM_MEAN, FRAME_NORM_STD, FRAME_SIZE, FRAME_TRANSFORM, MAX_CLIP_FRAMES
import numpy as np
import cv2
import torch

from api.mediapipe_detection.LandmarkExtractor import LandmarkExtractor
from api.schemas import WindowPrediction
from api.video_processing.landmark_preprocess import pack_landmarks


# FRAME READING ---

def fetch_fps(path: str | Path) -> tuple[float, int]:
    """Fetch a video's fps using cv2 capture."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, n_frames
    finally:
        cap.release()

def downscale_frame(frame: np.ndarray, max_size: Optional[int]) -> np.ndarray:
    if not max_size:
        return frame
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return frame
    else:
        return cv2.resize(frame, (int(w*scale), int(h*scale)))

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

def read_frames(video_path: str | Path, max_size: Optional[int] = None) -> Iterator[tuple[int, np.ndarray]]:
    """
    Decode each single frame of a video and iterate over them. Can downscale a video using side's `max_size` if needed.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("ERROR: Could not open the video file.")
    try:
        i = 0
        while cap.grab():
            ret, frame = cap.retrieve() # cuz grab, works similar to while True + cap.read()
            if ret:
                yield i, downscale_frame(frame, max_size)
            i += 1
    finally:
        cap.release()


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

def _buffer_to_window(buffer: deque, n: int) -> dict:
    """Make a window dict from last `n` frames in the buffer queue."""
    frames, landmarks, indices = [], [], []
    for i, f, lm in list(buffer)[-n:]:
        indices.append(i)
        frames.append(f)
        landmarks.append(lm)

    return {"frames": frames, "landmarks": landmarks, "frame_idx": indices,}


def extract_windows(
    frames: Iterable[tuple[int, np.ndarray]],
    extractor: LandmarkExtractor,
    window_size: int,
    stride: int,
    min_frames: int
    ):
    """
    Split a video into overlapping clips using a sliding window. Works per-frame.
    Calls iterate_window_split for the window processing.
    Window is made based on `min_frames`, `window_size` and the given stride. Stride may be overlapping (stride == n-frames == no overlap)
    """
    extractor.reset_stream()

    # Extract landmarks & prep frame
    def extract_frame_data() -> Iterator[tuple[int, torch.Tensor, np.ndarray]]:
        for i, frame in frames:
            landmarks, _ = extractor.detect_frame_landmarks(frame)
            yield i, preprocess_frame(frame), pack_landmarks(landmarks)

    return iterate_window_split(extract_frame_data(), window_size, stride, min_frames)

def iterate_window_split(
    data: Iterable[tuple[int, torch.Tensor, np.ndarray]],
    window_size: int,
    stride: int,
    min_frames: int
    ):
    """
    Split the processed frame and landmark `data` into overlapping mini-clips using a sliding window.
    Window is made based on `min_frames`, `window_size` and the given stride. Stride may be overlapping (stride == n-frames == no overlap)
    """
    if window_size <= 0:
        raise ValueError("Window_size must be greater than 0.")
    if stride <= 0 or stride > window_size:
        raise ValueError(f"Stride must be greater than 0 and less than window_size: {stride}.")

    return _split_to_windows(data, window_size, stride, min_frames)

def _split_to_windows(data, window_size: int, stride: int, min_frames: int) -> Iterator[dict]:
    # Iterate the windows.
    # When a window is formed, it gets sent, and the old frame gets pushed out with a new one.
    frame_queue: deque = deque(maxlen=window_size)
    frames_read, total_offset, keep_offset, trailing = 0, 0, 0, None

    # Iterate over (index, frame, landmarks) data samples
    for sample in data:
        frame_queue.append(sample)
        frames_read += 1

        if frames_read >= (total_offset + window_size): # full window
            window = _buffer_to_window(frame_queue, window_size)
            window['offset'] = total_offset

            if trailing is not None: # there is a pending window
                seam =(trailing["offset"] + window_size + total_offset) // 2
                trailing["keep"] = (keep_offset, seam)
                yield trailing
                keep_offset = seam
            trailing= window
            total_offset += stride

    # Handle trailing fragment
    if trailing is None:
        trail = frames_read
    else:
        trail = frames_read - total_offset

    has_frames = trailing is None or trail > (window_size - stride) # handles frames exactly in window size
    if trail >= min_frames and has_frames:
        window = _buffer_to_window(frame_queue, trail)
        window["offset"] = total_offset

        if trailing is not None:
            seam = (trailing["offset"] + window_size + total_offset) // 2
            trailing["keep"] = (keep_offset, seam)
            yield trailing
            keep_offset = seam

        window["keep"] = (keep_offset, frames_read)
        yield window

    elif trailing is not None:
        trailing["keep"] = (keep_offset, frames_read)
        yield trailing

    # if trailing:
    #     trail = frames_read + window_size - stride if frames_read else 0


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