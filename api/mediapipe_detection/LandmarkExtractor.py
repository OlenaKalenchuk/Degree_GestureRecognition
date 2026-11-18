import mediapipe as mp
from mediapipe.tasks.python import vision

from api.mediapipe_detection.LandmarkConfig import LandmarkConfig

import numpy as np
from pathlib import Path
from typing import Optional
import cv2

from api.config import MP_LANDMARKER_PATH


class LandmarkExtractor:
    """
    Manages the mediapipe landmark extraction, saving and loading.
    Uses Hand and Pose Landmark solution (locate 33 pose, 21 hand landmarks per hand).
    Utilises a config class 'LandmarkConfig'.
    """
    NUM_COORDS = 3 # x y z dim

    def __init__(self, config: Optional[LandmarkConfig] = None, video_reset_limit: int = 100):
        self.config = config if config is not None else LandmarkConfig(MP_LANDMARKER_PATH)
        self._video_reset_limit = video_reset_limit # detector rebuild
        self._video_count = 0 # since reset
        self._timestamp_ms = 0
        self.pose_detector = None
        self.hand_detector = None
        self._create_detectors()

    def _create_detectors(self):
        """Creates the pose/hand landmark detection models from config options and resets the timestamp clock."""
        self._close_detectors()
        detector_options = self.config.get_landmarker_options()
        if self.config.use_pose:
            self.pose_detector = vision.PoseLandmarker.create_from_options(detector_options['pose'])
        self.hand_detector = vision.HandLandmarker.create_from_options(detector_options['hands'])
        self._timestamp_ms = 0
        self._video_count = 0

    def _close_detectors(self):
        if self.pose_detector is not None:
            self.pose_detector.close()
        if self.hand_detector is not None:
            self.hand_detector.close()
        self.pose_detector = None
        self.hand_detector = None

    def close(self):
        """Closes landmarker models."""
        self._close_detectors()

    # If needed
    @staticmethod
    def sample_frames(frames, max_frames: int, stride: int = 1):
        """
        Skip `frames` data with set `stride`, crop to `max_frames`, shortening extracted video data.
        """
        result = frames[::stride]
        return result[:max_frames]

    @staticmethod
    def normalize_hand(landmarks: np.ndarray):
        """
        Center coordinates around the wrist (index 0), treating it as the origin (0,0,0) in world coordinates.
        Scale landmarks by max distance from origin.
        `landmarks` - (.., 21, 3) coordinates for any HAND.
        Returns: new coordinates with same (.., 21, 3) landmark format.
        """
        wrist = landmarks[:, 0:1, :] # origin landmarks[0]
        centered = landmarks - wrist
        scale = np.linalg.norm(centered, axis=-1) # scale = (.., 21)
        scale = scale.max(axis=-1, keepdims=True) + 1e-6 # landmark with furthest distance in coords. scale = (.., 1)

        return centered / scale[..., None] # broadcast over centered

    @staticmethod
    def crop_frame_bbox(bbox, frame, frame_h: int, frame_w: int, padding: float = 0.10):
        """
        Crop a `frame` to `bbox` bounding range of (x1, y1, x2, y2). The range is extended by `padding` (in percentage, relative to bbox size, per side) to ensure missing no details.
        Returns: cropped frame; bounding region within original frame bounds as [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = bbox
        bb_w, bb_h = x2 - x1, y2 - y1 # width / height
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 # center

        # apply padding
        x1, x2 = cx - int((bb_w / 2) * (1 + padding)), cx + int((bb_w / 2) * (1 + padding))
        y1, y2 = cy - int((bb_h / 2) * (1 + padding)), cy + int((bb_h / 2) * (1 + padding))
        # crop to frame bounds
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame_w, int(x2)), min(frame_h, int(y2))

        frame_cropped = frame[y1:y2, x1:x2]
        return frame_cropped, [x1, y1, x2, y2]

    def _landmarker_result_to_arrays(self, hand_result, pose_result):
        """
        Convert landmarker detection result into an array tensor. A utility method used in detection.
        Typical input includes: pose_landmarks, hand_landmarks.
        Returns: landmarks (LANDMARK_N, 3) for each TYPE - pose, left hand, right hand. '0' coordinates means undetected.
        """
        lhand_lm = np.zeros((len(self.config.HAND_LANDMARKS), self.NUM_COORDS), dtype=np.float32)
        rhand_lm = np.zeros((len(self.config.HAND_LANDMARKS), self.NUM_COORDS), dtype=np.float32)

        # landmarks are filtered by config
        if hand_result.hand_landmarks:
            for handedness, hand_landmarks in zip(hand_result.handedness, hand_result.hand_landmarks): # handedness - left/right
                hand = handedness[0].category_name.lower()
                if hand == 'left':
                    lhand_lm = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks], dtype=np.float32)[self.config.HAND_LANDMARKS]
                elif hand == 'right':
                    rhand_lm = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks], dtype=np.float32)[self.config.HAND_LANDMARKS]
        result = {"left_hand": lhand_lm, "right_hand": rhand_lm}
        if self.config.use_pose:
            pose_lm = np.zeros((len(self.config.POSE_LANDMARKS), self.NUM_COORDS), dtype=np.float32)
            if pose_result is not None and pose_result.pose_landmarks:
                pose_lm = np.array([(lm.x, lm.y, lm.z) for lm in pose_result.pose_landmarks[0]], dtype=np.float32)[self.config.POSE_LANDMARKS] # pose_landmarks holds multiple poses
            result["pose"] = pose_lm
        return result

    @staticmethod
    def _check_hands_present(hand_result) -> dict:
        """
        Check which hands are detected in a detection result. Used for live gesture window gating.
        Returns: {'left': bool, 'right': bool}
        """
        present = {'left': False, 'right': False}
        if hand_result.hand_landmarks:
            for handedness in hand_result.handedness:
                hand = handedness[0].category_name.lower()
                if hand in present:
                    present[hand] = True
        return present

    # HandLandmarker is weird in video mode, for each new video the timestamp is not reset and it throws an error.
    def detect_frame_landmarks(self, frame_bgr: np.ndarray, bbox: Optional[tuple] = None, frame_interval_ms: int = 33):
        """
        Detect landmarks for ONE frame. `frame_interval_ms` should reflect real capture fps (e.g. 1000/fps).
        Returns: landmarks {'pose','left_hand','right_hand'} with (LANDMARK_N, 3) for this frame,
                hands_present {'left': bool, 'right': bool}.
        """
        frame = frame_bgr
        if bbox is not None:
            h, w = frame.shape[:2]
            frame, _ = self.crop_frame_bbox(bbox, frame, frame_h=h, frame_w=w)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        self._timestamp_ms += frame_interval_ms
        if self.config.detection_mode == "video":
            hand_result = self.hand_detector.detect_for_video(mp_image, self._timestamp_ms)
            pose_result = self.pose_detector.detect_for_video(mp_image, self._timestamp_ms) if self.config.use_pose else None
        else:
            hand_result = self.hand_detector.detect(mp_image)
            pose_result = self.pose_detector.detect(mp_image) if self.config.use_pose else None

        landmarks = self._landmarker_result_to_arrays(hand_result, pose_result)
        hands_present = self._check_hands_present(hand_result)
        return landmarks, hands_present

    def reset_stream(self):
        """
        Call when a live session ends. Just an outer call to creating new detectors.
        """
        self._create_detectors()

    def toggle_pose(self, use: Optional[bool]):
        if use is None:
            self.config.use_pose = not self.config.use_pose
        else:
            self.config.use_pose = use

    @staticmethod
    def load_landmarks(file_path: str):
        """Loads landmarks from a *.npz* file."""
        with np.load(file_path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    @staticmethod
    def save_landmarks(landmarks, sign_labels: np.ndarray, video_ids: np.ndarray, frame_index: np.ndarray, output_path: str = 'landmarks.npz'):
        """
        Saves landmark data as a *.npz* file, at `output_path` specified.
        `landmarks` - coordinates, a dictionary with per-landmark-type arrays; `sign_labels` - gloss; `video_ids` - extracted videos, `frame_index` - row indexation table for video frames.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "left_hand": landmarks["left_hand"],
            "right_hand": landmarks["right_hand"],
            "sign_labels": sign_labels,
            "video_ids": video_ids,
            "frame_index": frame_index,
        }
        if "pose" in landmarks:
            data["pose"] = landmarks["pose"]
        np.savez_compressed(output_path, **data)
        print(f"Extracted landmarks saved as '{output_path}'.")

    # HandLandmarker is weird in video mode, for each new video the timestamp is not reset and it throws an error.
    # def detect_frames_landmarks(self, frame_paths: list, bbox: Optional[tuple | dict] = None):
    #     """
    #     Detect landmarks for given frames. Can be given a frame bounding box `bbox` as (x1, y1, x2, y2) or per-frame.
    #     Returns: per-frame (N,3) detection results for each frame as a list of each 'TYPE' landmark coordinates._
    #     """
    #     if self._video_count >= self._video_reset_limit:
    #         self._create_detectors()
    #     if len(frame_paths) == 0:
    #         print('WARNING: no frames were read. Check the video and requested range.')

    #     # landmarks_pose, landmarks_lhand, landmarks_rhand = [], [], []
    #     detector_timestamp = self._timestamp_ms

    #     # Read video feed
    #     for i, f_path in enumerate(frame_paths):
    #         frame = cv2.imread(str(f_path))
    #         if frame is None:
    #             raise ValueError(f"Failed to read the frame file: {f_path}")
    #         # crop to bbox if given
    #         if bbox is not None:
    #             if isinstance(bbox, dict) and i in bbox:
    #                 frame_bbox = bbox[i]
    #             else:
    #                 frame_bbox = bbox
    #             h, w = frame.shape[:2]
    #             frame, _ = self.crop_frame_bbox(frame_bbox, frame, frame_w=w, frame_h=h)

    #         # BGR to RGB because Mediapipe has no corresponding SBGR
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    #         # increase ts for video-mode internal tracker
    #         timestamp_ms = detector_timestamp + int((i / 30.0) * 1000) #  hardcoded 30 fps
    #         if self.config.detection_mode == "video":
    #             hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
    #             pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms) if self.config.use_pose else None
    #         else:
    #             hand_result = self.hand_detector.detect(mp_image)
    #             pose_result = self.pose_detector.detect(mp_image) if self.config.use_pose else None

    #         # add frame landmarks to the output
    #         landmarks = self._landmarker_result_to_arrays(hand_result, pose_result)
    #         hands_present = self._check_hands_present(hand_result)
    #         # if self.config.use_pose:
    #         #     landmarks_pose.append(landmarks['pose'])
    #         # landmarks_lhand.append(landmarks['left_hand'])
    #         # landmarks_rhand.append(landmarks['right_hand'])
    #         self._timestamp_ms = timestamp_ms + 1

    #     self._video_count += 1
    #     # if self.config.use_pose:
    #     #     return { 'pose': landmarks_pose, 'left_hand': landmarks_lhand, 'right_hand': landmarks_rhand }
    #     # return {'left_hand': landmarks_lhand, 'right_hand': landmarks_rhand }
    #     return landmarks, hands_present