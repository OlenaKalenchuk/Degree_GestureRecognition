import os
from pathlib import Path
import tempfile
from typing import Iterable, Optional
import numpy as np
import torch
import torch.nn.functional as F

from api.mediapipe_detection.LandmarkExtractor import LandmarkExtractor
from api.prediction.model_utils import ModelEntry, model_manager
from api.prediction.pred_decoders import CTCGreedyDecoder, TaskType
from api.schemas import GesturePrediction, PredictResponse, SequencePrediction, WindowPrediction
from api.video_processing.video_buffers import GestureBuffer, video_to_tensors
from api.video_processing.frame_preprocess import extract_windows, fetch_fps, merge_window_preds, read_frame_bytes, read_frames
from api.config import FILE_MAX_RESOLUTION, MAX_CLIP_FRAMES, WINDOW_MAX_FRAMES, WINDOW_MIN_FRAMES, WINDOW_OVERLAP

def _check_required_model(task: Optional[TaskType] = None) -> ModelEntry:
    """Check active model entry for the right task type (classification, CTC)."""
    entry = model_manager.active_model
    if entry is None:
        raise RuntimeError("No active model loaded. POST /models/load first.")
    if task is not None and entry.task != task:
        raise RuntimeError(f"Active model '{entry.model_id}' is a {entry.task.value} model, but the operation requires {task.value}.")
    return entry

# >> Run Model Prediction
def run_predict_class(tensors: dict) -> list[GesturePrediction]: #run_predict_video
    """
    Run a single video tensor through the active CLASSIFICATION model.
    Returns top-3 GesturePrediction objects, ranked by confidence.
    Do not confuse with 'predict_from_video' - this works with actual tensors.

    `tensors` should have: 'landmarks' (1, FRAMES, LANDMARK_N, 3), 'frames' (1, FRAMES, 3, H, W), 'pad_mask' (1, T).
    """
    entry = _check_required_model(TaskType.CLASSIFICATION)
    print(f"LANDMARKS {tensors['landmarks'].shape}  FRAMES {tensors['frames'].shape}  PADDING {tensors['pad_mask'].shape}")
    logits = entry.model.predict(tensors['landmarks'], tensors['frames'], tensors['pad_mask'].to(torch.bool))  # (1, num_classes)
    logits = logits.squeeze(0) # (num_classes,)

    return entry.decoder.decode(logits, k=3) # to top-k

def run_predict_sequence(tensors: dict, frame_offset: int = 0) -> list[GesturePrediction]:
    """
    Run a single video tensor through the active CTC model.
    Returns decoded GesturePrediction objects, ranked by confidence.
    Token frames are relative to the video's start + `frame_offset` 

    `tensors` should have: 'landmarks' (1, FRAMES, LANDMARK_N, 3), 'frames' (1, FRAMES, 3, H, W), 'pad_mask' (1, T).
    """
    entry = _check_required_model(TaskType.CTC)
    print(f"LANDMARKS {tensors['landmarks'].shape}  FRAMES {tensors['frames'].shape}  PADDING {tensors['pad_mask'].shape}")
    logits = entry.model.predict(tensors['landmarks'], tensors['frames'], tensors['pad_mask'].to(torch.bool))  # (1, T', C)
    logits = logits.squeeze(0) # (num_classes,)

    return entry.decoder.decode(logits, n_frames=tensors['frames'].shape[1], offset=frame_offset) # to top-k

# >> Prediction Result Helpers
#def predict_token_sequence(tensor_set: dict, window_size: int = CTC_CHUNK_FRAMES, overlap: int = CTC_CHUNK_OVERLAP)-> list[GesturePrediction]:
def predict_token_sequence(windows: Iterable[dict], window_size: int = WINDOW_MAX_FRAMES)-> list[GesturePrediction]:
    """
    Form a list of CTC predictions for a single video.
    If the length of the video is over the model limit, the video should be split into multiple overlapping windows.
    The start/end frames of the original video are added to the return.
    Each window in `windows` should have: 'landmarks', 'frames', 'frame_idx' (1, T), 'offset' first frame position, 'keep' as a frame tag.
    """
    tokens: list[GesturePrediction] = [] # like window preds
    for window in windows:
        tensors = video_to_tensors(window, window_size)
        predicted_tokens = run_predict_sequence(tensors, frame_offset=window['offset']) # in the window
        # Keep a token that goes into this window, determined by a token's frame centre
        lo, hi = window["keep"]
        window_tokens = [t for t in predicted_tokens if lo <= (t.start_frame + t.end_frame) / 2 < hi]

        # Map processed-video positions to original frame indices.
        last = len(window["frame_idx"]) - 1

        for token in window_tokens:
            start = max(0, min(token.start_frame - window["offset"], last))
            end = max(0, min(token.end_frame - window["offset"], last))

            tokens.append(
                token.model_copy(update={
                "start_frame": window["frame_idx"][start],
                "end_frame": window["frame_idx"][end],
            }))

    return sorted(tokens, key=lambda t: t.start_frame)


# >> Video Inference types

def predict_from_path(
    video_path: Path,
    extractor: LandmarkExtractor,
    max_frames: int = MAX_CLIP_FRAMES,
    window_size: int = WINDOW_MAX_FRAMES,
    stride: Optional[int] = None,
    min_window_frames: int = WINDOW_MIN_FRAMES,
    min_confidence: float = 0.0,
    max_side: Optional[int] = FILE_MAX_RESOLUTION
    ) -> PredictResponse:
    """
    Run inference over a single UPLOADED mp4 video and assemble a PredictResponse.
    Video is split into several cut clips ("windows" or segments) that run into prediction to ensure nothing is missed.
    May be slow because of video decoding.
    """
    entry = _check_required_model()
    if stride is None:
        stride = window_size - WINDOW_OVERLAP

    fps, n_frames = fetch_fps(video_path)
    frames = read_frames(video_path, max_side)
    video_samples = extract_windows(frames, extractor, window_size, stride, min_window_frames) # TODO: NEEDS COUNT

    windows: list[WindowPrediction] = []
    sequences: list[SequencePrediction] = []

    # Two tasks - either CTC or Classification. The prep approach depends on the task.
    if entry.task == TaskType.CTC:
        tokens = predict_token_sequence(video_samples)
        if tokens: # none = blanks
            sequences.append(SequencePrediction(segment_index=len(sequences), text=CTCGreedyDecoder.to_text(tokens), tokens=tokens)) # span is omitted because its a full video
    elif entry.task == TaskType.CLASSIFICATION:
        for video in video_samples:
            start, end = video['frame_idx'][0], video['frame_idx'][-1]
            span = dict(
                start_frame=start,
                end_frame=end,
                start_time=round(start / fps, 3),
                end_time=round((end + 1) / fps, 3),
            )
            top3 = run_predict_class(video_to_tensors(video, max_frames, resample=True))
            windows.append(WindowPrediction(window_index=len(windows), top1=top3[0], top3=top3, **span))

        # Merge windows to unique ones and reindex
        windows = merge_window_preds(windows, min_confidence=min_confidence)
        for i, window in enumerate(windows):
            window.window_index = i

    return PredictResponse(active_model=model_manager.active_id, n_frames=n_frames, windows=windows, sequences=sequences)


def predict_from_video(
    video_bytes: bytes,
    extractor: LandmarkExtractor,
    **kwargs
    ) -> PredictResponse:
    """Predict from an upload"""
    # Save a temporary video mp4 file for detection
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        #tmp.flush()
    try:
        return predict_from_path(Path(tmp.name), extractor, **kwargs)
    finally:
        Path(tmp.name).unlink(missing_ok=True)

def predict_from_live(
    video: dict, 
    frame_limit: int = MAX_CLIP_FRAMES
    ) -> dict:
    """Predict from the websocket"""
    entry = _check_required_model()
    if entry.task == TaskType.CTC:
        tensors = video_to_tensors(video, WINDOW_MAX_FRAMES)
        tokens = run_predict_sequence(tensors)
        # tokens = _map_token_frames(tokens, video["frame_idx"])
        return {"task": entry.task.value, "text": CTCGreedyDecoder.to_text(tokens), "tokens": [t.model_dump() for t in tokens]}
    else:
        top3 = run_predict_class(video_to_tensors(video, frame_limit, True))
        return {"task": entry.task.value, "top1": top3[0].model_dump(), "top3": [p.model_dump() for p in top3]}


# def _map_token_frames(
#     tokens: list[GesturePrediction],
#     frame_idx: list[int]
# ) -> list[GesturePrediction]:

#     last = len(frame_idx) - 1

#     return [
#         token.model_copy(update={
#             "start_frame": frame_idx[max(0, min(token.start_frame, last))],
#             "end_frame": frame_idx[max(0, min(token.end_frame, last))]
#         })
#         for token in tokens
#     ]