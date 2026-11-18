import numpy as np

from api.mediapipe_detection.LandmarkExtractor import LandmarkExtractor


def find_shared_index(landmark_count: int, frame_count: int, target_len: int) -> np.ndarray:
    """
    Finds one shared set of `target_len` indices used for landmarks and frames.
    Returns: (target_len,) array.
    """
    if landmark_count != frame_count:
        print(f"WARNING: landmark_count={landmark_count} is different from frame_count={frame_count}.")
    n = min(landmark_count, frame_count)
    if n <= 0:
        raise ValueError(f"No usable frames: landmark_count={landmark_count}, frame_count={frame_count}.")
    idx = np.linspace(0, n - 1, num=target_len)
    return np.round(idx).astype(np.int64)


def pack_landmarks(landmarks: dict) -> np.ndarray:
    """
    Concats landmarks of pose + left_hand + right_hand into a single (LANDMARK_N, 3) array per frame.
    Hands are normalized around the wrist, Pose coordinates are left non-normalized.
    """
    left = LandmarkExtractor.normalize_hand(landmarks['left_hand'][None, ...])[0]
    right = LandmarkExtractor.normalize_hand(landmarks['right_hand'][None, ...])[0]
    packed = np.concatenate([landmarks['pose'], left, right], axis=0) if "pose" in landmarks else np.concatenate([left, right], axis=0)
    return packed