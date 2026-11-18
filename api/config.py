from pathlib import Path
import torchvision.transforms as T

# ===== SAVED MODELS =====

# Serves as model config: saved_models/model_configs.json
# each entry has: model_id, weights_path, class_map_path, device 
MODEL_CONFIG_PATH = Path(__file__).parent / "saved_models" / "model_configs.json"

# Saved models & class maps: saved_models/<weights_file>
MODEL_DIR  = Path(__file__).parent / "saved_models" 


# ===== FILE UPLOADS =====

FILE_TARGET_FPS = None # None = keep video fps
FILE_MAX_RESOLUTION = None # shrink by the longer side before detection



# ===== MEDIAPIPE =====

# Mediapipe extracted landmarks
HAND_LANDMARKS = list(range(21)) # for both hands 0-21
POSE_LANDMARKS = [11, 12, 13, 14, 15, 16] # shoulders and arms
NUM_LANDMARKS = (len(HAND_LANDMARKS) * 2) + len(POSE_LANDMARKS) # default: 48
NUM_COORDS = 3 # x y z dim

# Mediapipe pose & hand landmarker task path: api/mediapipe/pose_landmarker_full.task, api/mediapipe/hand_landmarker.task
MP_LANDMARKER_PATH = {
    'pose': Path(__file__).parent / "mediapipe_detection" / "pose_landmarker_full.task",
    'hands': Path(__file__).parent / "mediapipe_detection" / "hand_landmarker.task"
}



# ===== VIDEO PREPROCESSING =====

# Video preprocessing settings
MAX_CLIP_FRAMES = 32 # frame per video threshold set by CNN
FRAME_SIZE = (224, 224) # (W, H)

WINDOW_MAX_FRAMES = 128 # max frames per call
WINDOW_MIN_FRAMES = 8 # min frames per call
WINDOW_OVERLAP = 32 # frames carried into the next call


# Model image normalization (ImageNet default)
FRAME_NORM_MEAN = [0.485, 0.456, 0.406]
FRAME_NORM_STD = [0.229, 0.224, 0.225]

FRAME_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize(FRAME_SIZE),
    T.ToTensor(),
    T.Normalize(mean=FRAME_NORM_MEAN, std=FRAME_NORM_STD)
])



# ===== LIVE FEED SETTINGS =====

LIVE_MAX_CLIP_FRAMES = 60 # limit for live gesture
LIVE_MIN_CLIP_FRAMES = 8
LIVE_OVERLAP_FRAMES = 16 # frames carried into the next video portion when a hand is up 
LIVE_MISSING_TOLERANCE = 6 # consecutive frames with no hand gesture before window closing