from typing import Optional
import numpy as np

from api.video_processing.video_buffers import GestureBuffer
from api.mediapipe_detection.LandmarkConfig import LandmarkConfig
from api.mediapipe_detection.LandmarkExtractor import LandmarkExtractor
from api.prediction.pred_decoders import TaskType
from api.config import LIVE_OVERLAP_FRAMES, MP_LANDMARKER_PATH, MAX_CLIP_FRAMES, LIVE_MAX_CLIP_FRAMES, LIVE_MIN_CLIP_FRAMES, LIVE_MISSING_TOLERANCE


class LiveDetectionSession:
    """
    A per-client session made for live video stream recognition.
    Creates a LandmarkExtractor on startup, and needs to be .close()-d after connection end.
    Landmarks are extracted regardless of main model recognition.
    """
    def __init__(
        self,
        landmark_config: Optional[LandmarkConfig] = None,
        frame_limit: int = MAX_CLIP_FRAMES,
        max_frames: int = LIVE_MAX_CLIP_FRAMES,
        min_frames: int = LIVE_MIN_CLIP_FRAMES,
        overlap: int = LIVE_OVERLAP_FRAMES,
        hand_tolerance: int = LIVE_MISSING_TOLERANCE
        ):
        config = landmark_config or LandmarkConfig(MP_LANDMARKER_PATH)
        self.extractor = LandmarkExtractor(config)
        self.frame_limit, self.max_frames, self.min_frames = frame_limit, max_frames, min_frames
        self.overlap = overlap
        self.hand_tol = hand_tolerance
        self.buffer = GestureBuffer(max_frames, min_frames, hand_tolerance, overlap)
        self.task = TaskType.CLASSIFICATION # default
        self.enable_recognition = False

    def create_buffer(self, task: TaskType) -> GestureBuffer:
        """Create a buffer depending on the current model task."""
        if task == TaskType.CLASSIFICATION:
            return GestureBuffer(self.max_frames, self.min_frames, self.hand_tol, overlap=self.overlap)
        elif task == TaskType.CTC:
            return GestureBuffer(self.max_frames, self.min_frames, self.hand_tol)

    def switch_task_to(self, task:TaskType):
        """Configures a new buffer if a model task was changed."""
        if task != self.task:
            self.task = task
            self.buffer = self.create_buffer(task)
    
    def toggle_recognition(self, enabled: bool):
        """Toggle main gesture recognition on/off. Resets the inner buffer."""
        self.enable_recognition = enabled
        if not enabled:
            self.buffer.reset()

    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Returns
        - `landmarks`: for drawing overlays, copy is sent to buffer
        - `hands_present`: {'left': bool, 'right': bool}
        - `video_result`: a filled dict from GestureBuffer, or None if not ready yet/disabled. Needs to be resampled before prediction.
        """
        video_result = None
        landmarks, hands_present = self.extractor.detect_frame_landmarks(frame_bgr)
        if self.enable_recognition:
            video_result = self.buffer.push(frame_bgr, landmarks, hands_present)
        return {'landmarks': landmarks, 'hands_present': hands_present, 'video_result': video_result}

    def close(self):
        """Free mediapipe detector memory."""
        self.extractor.close()