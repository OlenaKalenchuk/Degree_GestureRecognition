# Camera input
import numpy as np
import torch
from typing import Optional

from api.video_processing.frame_preprocess import preprocess_frame
from api.video_processing.landmark_preprocess import find_shared_index, pack_landmarks


class GestureBuffer:
    """
    Buffer for video recording input. Once the buffer has MAX_FRAMES frames it emits ready (frame, landmarks) pairs.
    These data are gated on hand presence - start when a hand becomes visible, end when its absent for  `hand_tolerance` frames.
    Buffers shorter than `min_frames` are discarded as noise.
    When the video ends with the frame cap, the last `overlap` frames are carried into the next video.
    """
    def __init__(self, max_frames: int = 32, min_frames: int = 8, hand_tolerance: int = 6, overlap: int = 0):
        if not 0 <= overlap < max_frames:
            raise ValueError("Frame overlap must be in [0, max_frames).")
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.hand_tolerance = hand_tolerance
        self.overlap = overlap
        self._frames: list[torch.Tensor] = []
        self._landmarks: list[np.ndarray] = []
        self._frame_indices: list[int] = []
        self._abscent_count = 0 # hand abscence streak
        self._next_frame_idx = 0 # if no index supplied
        self._active = False

    def push(self, frame_bgr: np.ndarray, landmarks: dict, hands_present: dict, frame_idx: Optional[int] = None) -> Optional[dict]:
        """
        Adds one frame + landmark pair. `frame_idx` is supplied for videofiles (not live), it's the frame's position in upload.
        Returns: {'frames', 'landmarks', 'frame_idx'} or None.
        """
        is_hand_present = hands_present.get('left', False) or hands_present.get('right', False)
        idx = frame_idx if frame_idx is not None else self._next_frame_idx
        self._next_frame_idx = idx + 1

        if is_hand_present:
            self._abscent_count = 0
            self._active = True
        elif self._active:
            self._abscent_count += 1

        if self._active:
            self._frames.append(preprocess_frame(frame_bgr))
            self._landmarks.append(pack_landmarks(landmarks))
            self._frame_indices.append(idx)

        is_hand_abscent = self._abscent_count >= self.hand_tolerance # after the hand threshold hit
        is_fcap_reached = len(self._frames) >= self.max_frames
        if self._active and (is_hand_abscent or is_fcap_reached):
            return self.flush(keep_last=is_fcap_reached and not is_hand_abscent)
        return None

    def flush(self, keep_last:bool = False) -> Optional[dict]:
        """End the current clip and reset buffer. Returns None if buffer is too short."""
        frames, landmarks, indices = self._frames, self._landmarks, self._frame_indices
        self.reset()
        
        if keep_last and self.overlap:
            # Populate the next one with this ones' end
            self._frames, self._landmarks = frames[-self.overlap:], landmarks[-self.overlap:]
            self._frame_indices = indices[-self.overlap:]
            self._active = True

        if len(frames) < self.min_frames:
            # Discard very short clips
            return None
        return {'frames': frames, 'landmarks': landmarks, 'frame_idx': indices}

    def reset(self):
        """Discards any in-progress clip, e.g. when recognition is toggled off mid-gesture."""
        self._frames, self._landmarks = [], []
        self._frame_indices = [] 
        self._abscent_count, self._active = 0, False


def video_to_tensors(video: dict, max_frames: int, resample: bool = False) -> dict:
    """
    Make tensors from a buffered video dictionary and resample it to exactly `max_frames` frames with find_shared_index.
    It's a safeguard against wrong-length videos, that resamples if needed.
    Returns: ready tensors: landmarks (1, T, 3, H, W), frames (1, T, LANDMARK_N, 3), padding (1, T).
    """
    n_frames = len(video['frames'])
    n_landmarks = len(video['landmarks'])
    n = min(n_frames, n_landmarks) # safeguard against the limit

    if resample:
        # video becomes (max_frames)
        n = max_frames
        idx = find_shared_index(n_landmarks, n_frames, max_frames)
    else:
        if n > max_frames:
            raise ValueError(f"Video has {n} frames, with a {max_frames} limit. Slice it into separate parts before inference.")
        idx = np.arange(n)

    frames = torch.stack([video['frames'][i] for i in idx][:n])
    landmarks = np.stack([video['landmarks'][i] for i in idx][:n])

    return {
        'frames': frames.unsqueeze(0),
        'landmarks': torch.from_numpy(landmarks).float().unsqueeze(0),
        'pad_mask': torch.zeros(1, n, dtype=torch.float32)
    }