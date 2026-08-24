import tempfile
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

from api.mediapipe_detection.LandmarkExtractor import LandmarkExtractor
from api.prediction.model_utils import model_manager
from api.schemas import GesturePrediction, PredictResponse, WindowPrediction
from api.video_processing.video_buffers import GestureBuffer, resample_video
from api.video_processing.frame_preprocess import extract_windows, merge_window_preds, read_frame_bytes
from api.config import LIVE_MAX_CLIP_FRAMES, LIVE_MIN_CLIP_FRAMES, LIVE_MISSING_TOLERANCE, MAX_CLIP_FRAMES

def _logits_to_top_k(logits: torch.Tensor, k: int = 3) -> list[GesturePrediction]:
    """
    Convert raw logits (1-D tensor of length NUM_CLASSES) to a ranked list of GesturePrediction objects.
    """
    classes = {int(v): k for k, v in model_manager.class_map.items()}
    num_classes = len(classes)

    if logits.shape[-1] != num_classes:
        raise RuntimeError(
            f"ERROR: Model output has {logits.shape[-1]} logits but classes file defines {num_classes} classes."
            f"Make sure the right class list and model weights are being used."
        )
    
    probs = F.softmax(logits, dim=-1)
    top_probs, top_idxs = torch.topk(probs, k=min(k, num_classes))
    return [
        GesturePrediction(
            label=classes[int(idx.item())],
            class_index=int(idx.item()),
            confidence=round(prob.item(), 4)
        )
        for prob, idx in zip(top_probs, top_idxs)
    ]


def run_predict_video(tensors: dict) -> list[GesturePrediction]:
    """
    Run a single video tensor through the active model.
    Returns top-3 GesturePrediction objects, ranked by confidence.
    Do not confuse with 'predict_from_video' - this works with actual tensors.

    `tensors` should have: 'landmarks' (1, FRAMES, LANDMARK_N, 3), 'frames' (1, FRAMES, 3, H, W), 'pad_mask' (1, T).
    """
    entry = model_manager.active_model
    if entry is None:
        raise RuntimeError("No active model loaded. POST /models/load first.")


    logits = entry.model.predict(tensors['landmarks'], tensors['frames'], tensors['pad_mask'].to(torch.bool))  # (1, num_classes)
    logits = logits.squeeze(0)  # (num_classes,)

    return _logits_to_top_k(logits, k=3)


def predict_from_video(
    video_bytes: bytes,
    extractor: LandmarkExtractor,
    max_frames: int = MAX_CLIP_FRAMES,
    window_size: int = LIVE_MAX_CLIP_FRAMES,
    stride: Optional[int] = None,
    min_window_frames: int = LIVE_MIN_CLIP_FRAMES,
    min_confidence: float = 0.0,
    ) -> PredictResponse:
    """
    Run inference over a single mp4 video and assemble a PredictResponse.
    Video is split into several cut clips ("windows") that run into prediction to ensure nothing is missed.
    May be slow because of video decoding.
    """
    if stride is None:
        stride = window_size // 2


    # Save a temporary video mp4 file for detection
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        frames, fps = read_frame_bytes(tmp.name) 
        if not frames:
            raise ValueError("ERROR: Could not decode any frames from the video.")
        
    clip_fragments = extract_windows(
        frames=frames,
        extractor=extractor,
        window_size=window_size,
        stride=stride,
        min_frames=min_window_frames,
    )

    if not clip_fragments:
        # Safeguard against "no person present" situations.
        return PredictResponse(active_model=model_manager.active_id, n_frames=len(frames), windows=[])

    windows: list[WindowPrediction] = []

    for i, clip in enumerate(clip_fragments):
        clip_tensors = resample_video(clip, max_frames)
        top3 = run_predict_video(clip_tensors)
        start_idx, end_idx = clip['frame_indices'][0], clip['frame_indices'][-1]

        windows.append(
            WindowPrediction(
                window_index=i,
                start_frame=start_idx,
                end_frame=end_idx,
                start_time=round(start_idx / fps, 3),
                end_time=round((end_idx + 1) / fps, 3),
                top1=top3[0],
                top3=top3
            )
        )
    # Merge them to unique ones and reindex
    merged_windows = merge_window_preds(windows, min_confidence=min_confidence)
    for i, window in enumerate(merged_windows):
        window.window_index = i

    return PredictResponse(active_model=model_manager.active_id, n_frames=len(frames), windows=merged_windows)