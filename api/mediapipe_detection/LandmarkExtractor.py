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
      pose_lm = np.zeros((len(self.config.POSE_LANDMARKS), self.NUM_COORDS), dtype=np.float32)

      # landmarks are filtered by config
      if hand_result.hand_landmarks:
        for handedness, hand_landmarks in zip(hand_result.handedness, hand_result.hand_landmarks): # handedness - left/right
            hand = handedness[0].category_name.lower()
            if hand == 'left':
              lhand_lm = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks], dtype=np.float32)[self.config.HAND_LANDMARKS]
            elif hand == 'right':
              rhand_lm = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks], dtype=np.float32)[self.config.HAND_LANDMARKS]
      if pose_result.pose_landmarks:
        pose_lm = np.array([(lm.x, lm.y, lm.z) for lm in pose_result.pose_landmarks[0]], dtype=np.float32)[self.config.POSE_LANDMARKS] # pose_landmarks holds multiple poses

      return {
          'pose': pose_lm, 'left_hand': lhand_lm, 'right_hand': rhand_lm
      }

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
  def detect_video_landmarks(self, video_path: str, start_frame: int = 1, end_frame: int = -1, fps: Optional[float] = None, bbox: Optional[tuple] = None):
      """
      Detect landmarks for each frame in a given videofile. Can be given a set `fps`, gloss boundaries [`start_frame`, `end_frame`], and a frame bounding box `bbox` (x1, y1, x2, y2).
      In WLASL `fps` is different from video fps, typically == *25*; `end_frame` == *-1* means until the end.
      Returns: (FRAMES, TYPE_N, LANDMARK_N, 3) detection results for each frame as a list of each 'TYPE' landmark coordinates._
      """
      if self._video_count >= self._video_reset_limit:
        self._create_detectors()

      cap = cv2.VideoCapture(video_path)
      fps = fps or cap.get(cv2.CAP_PROP_FPS)
      total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

      # define start and end frame for detection
      start_frame = max(1, start_frame)
      # start is after the video end
      if start_frame > total_frames:
        start_frame = 1
        end_frame = total_frames
      # until the end handle
      if end_frame < 0 or end_frame > total_frames:
        end_frame = total_frames

      landmarks_pose, landmarks_lhand, landmarks_rhand = [], [], []
      detector_pose_results, detector_hand_results = [], []
      frame_idx = 1 # in video
      detect_frame_idx = 0 # in the [start:end] sequence, just in case
      detector_timestamp = self._timestamp_ms

      # Read video feed
      while cap.isOpened() and frame_idx <= end_frame:
          ret, frame = cap.read()
          if not ret:
            break
          if frame_idx >= start_frame:
              # crop to bbox if given
              if bbox is not None:
                frame, _ = self.crop_frame_bbox(bbox, frame, frame_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), frame_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

              # BGR to RGB because Mediapipe has no corresponding SBGR
              frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

              # increase ts for video-mode internal tracker
              #timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1000)
              timestamp_ms = detector_timestamp + int((detect_frame_idx / fps) * 1000)
              if self.config.detection_mode == "video":
                hand_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
                pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
              else:
                hand_result = self.hand_detector.detect(mp_image)
                pose_result = self.pose_detector.detect(mp_image)

              # add frame landmarks to the output
              landmarks = self._landmarker_result_to_arrays(hand_result, pose_result)
              landmarks_pose.append(landmarks['pose'])
              landmarks_lhand.append(landmarks['left_hand'])
              landmarks_rhand.append(landmarks['right_hand'])
              detector_hand_results.append(hand_result)
              detector_pose_results.append(pose_result)
              detect_frame_idx += 1
              self._timestamp_ms = timestamp_ms + 1

          frame_idx += 1

      cap.release()
      self._video_count += 1
      if detect_frame_idx == 0:
        print('WARNING: no frames were read. Check the video and requested range.')
      return { 'pose': landmarks_pose, 'left_hand': landmarks_lhand, 'right_hand': landmarks_rhand },  { 'pose': detector_pose_results, 'hand': detector_hand_results }

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
          pose_result = self.pose_detector.detect_for_video(mp_image, self._timestamp_ms)
      else:
          hand_result = self.hand_detector.detect(mp_image)
          pose_result = self.pose_detector.detect(mp_image)

      landmarks = self._landmarker_result_to_arrays(hand_result, pose_result)
      hands_present = self._check_hands_present(hand_result)
      return landmarks, hands_present

  def reset_stream(self):
      """
      Call when a live session ends. Just an outer call to creating new detectors.
      """
      self._create_detectors()

  # def detect_for_dataset(self, video_dir: str, video_meta_df: pd.DataFrame, video_path_column: str = 'video_path', sign_column: str = 'sign', normalize: bool = True, save_results: bool = True, save_path: str = '/landmarks/WLASL_landmarks.npz'):
  #     """
  #     Detect landmarks for videos specified in a `video_meta_df` dataset. Videos should be available as .mp4 in `video_dir` folder.
  #     Optionally, save extracted data; normalize hand landmarks, centering coordinates around the wrist.
  #     Returns: A dictionary of resulting arrays - landmarks, video_ids, sign_labels, frame_index.
  #     """

  #     # determine source availabilty
  #     if video_meta_df is None or video_path_column not in video_meta_df.columns or sign_column not in video_meta_df.columns:
  #       raise ValueError('No video metadata available - dataframe is unidentified or wrong column specified.')
  #     if not Path(video_dir).exists():
  #       raise ValueError('No video files available - `video_dir` path does not exist.')

  #     landmarks_data = {"pose": [], "left_hand": [], "right_hand": []} # landmarks
  #     video_ids, sign_labels = [], [] # detected videos and their labels

  #     for video_data in tqdm(video_meta_df.itertuples(), total=len(video_meta_df), desc="Extracting landmarks"):
  #         # no check exists, it was cleaned for base dataset
  #         # using getattr just in case
  #         video_path = getattr(video_data, video_path_column)
  #         video_id = getattr(video_data, 'video_id', Path(video_path).stem)
  #         gloss = getattr(video_data, sign_column, "Unknown")
  #         fps = getattr(video_data, 'fps', None)
  #         start_frame = getattr(video_data, 'frame_start', 1)
  #         end_frame = getattr(video_data, 'frame_end', -1)
  #         bbox = getattr(video_data, 'bbox', None)

  #         # extract
  #         lm, _ = self.detect_video_landmarks(str(video_path), start_frame, end_frame, fps, bbox)
  #         lm = {k: np.stack(v, axis=0) for k, v in lm.items()} # Make each (F, LANDMARK_N, 3)
  #         if normalize:
  #           lm['left_hand'] = self.normalize_hand(lm['left_hand'])
  #           lm['right_hand'] = self.normalize_hand(lm['right_hand'])

  #         # append per landmark type
  #         for lm_key in landmarks_data:
  #           landmarks_data[lm_key].append(lm[lm_key])
  #         video_ids.append(video_id)
  #         sign_labels.append(gloss)

  #     # constructing video indexing - landmark row from-to ranges per video. 'pose' just because any would fit
  #     frame_totals = np.array([lm.shape[0] for lm in landmarks_data['pose']]) # frames per video, e.g. 30 30 25...
  #     frame_ends = np.cumsum(frame_totals) # e.g. 30 60 85...
  #     frame_starts = frame_ends - frame_totals
  #     frame_index = np.stack([frame_starts, frame_ends], axis=1)  # (num_videos, 2) indexation table
  #     # now cut frame ids
  #     landmarks_data = {k: np.concatenate(v, axis=0) for k, v in landmarks_data.items()}
  #     video_ids = np.array(video_ids)
  #     sign_labels = np.array(sign_labels)
  #     print(f"Landmark extraction completed. Total videos processed: {len(video_ids)}  ({landmarks_data['pose'].shape[0]} frames).")

  #     if save_results:
  #       self.save_landmarks(landmarks_data, sign_labels, video_ids, frame_index, output_path=save_path)
  #     return {
  #         **landmarks_data, # unpack dict
  #         'sign_labels': sign_labels,
  #         'video_ids': video_ids,
  #         'frame_index': frame_index
  #     }

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
      np.savez_compressed(
          output_path,
          pose = landmarks['pose'],
          left_hand = landmarks['left_hand'],
          right_hand = landmarks['right_hand'],
          sign_labels = sign_labels,
          video_ids = video_ids,
          frame_index = frame_index
          )
      print(f"Extracted landmarks saved as '{output_path}'.")