from api.config import HAND_LANDMARKS, POSE_LANDMARKS
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from typing import Literal

class LandmarkConfig:
    """
    A config class for LandmarkExtractor. Holds values that are not typically changed during landmark detection process.
    """

    def __init__(
            self, 
            lm_model_paths: dict, 
            min_detection_confidence: float = 0.5, 
            min_tracking_confidence: float = 0.5,
            default_detection_mode: Literal["video", "image"] = "video",
            num_hands: Literal[1, 2] = 2, 
            use_pose: bool = True
            ):
        """
        `lm_model_paths` format example: {"pose": "pose_landmarker.task", "hands": "hand_landmarker.task"}
        """
        if default_detection_mode not in ("video", "image"):
            raise ValueError(f"Detection mode must be either 'video' or 'image': {default_detection_mode}.")
        self.MIN_DETECTION_CONFIDENCE = min_detection_confidence
        self.MIN_TRACKING_CONFIDENCE = min_tracking_confidence # # if fails, detection occurs. better set to lower
        self.MODEL_PATHS = lm_model_paths
        self.HAND_LANDMARKS = HAND_LANDMARKS
        self.POSE_LANDMARKS = POSE_LANDMARKS
        self.default_detection_mode = default_detection_mode
        self.detection_mode = default_detection_mode # CURRENT
        self.use_pose = use_pose
        self.num_hands = num_hands

    def change_detection_mode(self, detection_mode: Literal["video", "image"] = "video"):
        if detection_mode not in ("video", "image"):
            raise ValueError(f"Detection mode must be either 'video' or 'image': {detection_mode}.")
        self.detection_mode = detection_mode
        print(f'Landmark extraction mode set to "{detection_mode}".')

    def get_landmarker_options(self):
        hands_path = str(self.MODEL_PATHS['hands'])
        pose_path = str(self.MODEL_PATHS['pose'])
        if not self.MODEL_PATHS['hands'].exists:
             raise FileNotFoundError(f"Model file not found at: {hands_path}")
        if not self.MODEL_PATHS['pose'].exists:
             raise FileNotFoundError(f"Model file not found at: {pose_path}")
        hand_options = vision.HandLandmarkerOptions(
            base_options = BaseOptions(model_asset_path=hands_path),
            running_mode = vision.RunningMode.VIDEO if self.detection_mode == 'video' else vision.RunningMode.IMAGE,
            num_hands = self.num_hands, # 2 = both
            min_hand_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options = BaseOptions(model_asset_path=pose_path),
            running_mode = vision.RunningMode.VIDEO if self.detection_mode == 'video' else vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE
        ) if self.use_pose else None
        return {
            "hands": hand_options,
            "pose": pose_options
        }