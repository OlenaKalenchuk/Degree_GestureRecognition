import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from enum import Enum
import logging

from api.config import MODEL_CONFIG_PATH, MODEL_DIR
from api.mediapipe_detection.LandmarkExtractor import LandmarkExtractor
from api.mediapipe_detection.LandmarkConfig import LandmarkConfig


try:
    import onnxruntime as ort
except ImportError:
    ort = None
logger = logging.getLogger("gesture_api.preload")


class SaveMethod(Enum):
    FULL = 1, # torch.save
    WEIGHTS_ONLY = 2,  # state_dict
    ONNX = 3 # .onnx

def preload_models(config_path: Path | str = MODEL_CONFIG_PATH)-> dict[str, str]:
    """
    Load every model listed in models_config.json.
    Bad entries are skipped (with reason stated), the rest are loaded normally.
    """
    config_path = Path(config_path)
    results: dict[str, str] = {}
    if not config_path.exists():
        logger.warning("Config %s is abscent, no model entries found - the API will start without any preloaded models.", config_path)
        return results

    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    for cfg in configs:
        model_id = cfg.get("model_id", "<unknown>")
        try:
            model_manager.load(
                model_id=model_id,
                weights_path=cfg["weights_path"],
                class_map_path=cfg["class_map_path"],
                save_type=SaveMethod[cfg.get("save_type", "FULL")],
                device=cfg.get("device", "cpu")
            )
            results[model_id] = "loaded"
        except Exception as exc:
            results[model_id] = f"Fail: {exc}"
    
    # Apply active model
    if model_manager.active_id:
        logger.info("Active model defaulted to '%s' (first loaded).", model_manager.active_id)
    else:
        logger.warning("No models loaded successfully - all prediction endpoints will return 503.")
    return results


class TorchModelWrapper:
    """
    Uniform torch model wrapper regardless of backbone architecture.
    Loaded models become this class's backbone.
    """
    # For GestureDualModel - torch.Size([1, 200, 256])
    def __init__(self, backbone: nn.Module, device: str):
        super().__init__()
        self.backbone = backbone
        self.device = torch.device(device)

    def predict(self, x: torch.Tensor, y: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Input: landmarks, frames, padding. Returns: (B, num_classes)"""
        x.to(self.device)
        y.to(self.device)
        pad_mask.to(self.device)
        output = self.backbone(x, y, pad_mask)
        return output.cpu()    

class OnnxModelWrapper:
    """
    Uniform ONNX InferenceSession model wrapper.
    Input: ["landmarks", "frames", "padding_mask"], Output: ["output"].
    """
    INPUT_NAMES = {"landmarks": "landmarks", "frames": "frames", "pad_mask": "padding_mask"}
    OUTPUT_NAME = "output"
 
    def __init__(self, weights_path: str, device: str):
        if ort is None:
            raise ImportError("Module onnxruntime is not installed - can't load ONNX models.")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(weights_path, providers=providers)
        self.device = device
 
    def predict(self, landmarks: torch.Tensor, frames: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Returns: (B, num_classes) on cpu."""
        feeds = {
            self.INPUT_NAMES["landmarks"]: landmarks.cpu().numpy(),
            self.INPUT_NAMES["frames"]: frames.cpu().numpy(),
            self.INPUT_NAMES["pad_mask"]: pad_mask.cpu().numpy(),
        }
        outputs = self.session.run([self.OUTPUT_NAME], feeds)
        return torch.from_numpy(outputs[0])

class ModelEntry:
    """
    Entry of one of the available models.
    """
    def __init__(
        self,
        model_id: str,
        model,
        class_map: dict[str, int],
        device: str
        ):
        self.model_id = model_id
        self.model = model
        self.class_map = class_map
        self.device = device



class ModelManager:
    """
    Main model helper class, available throughout the app.
    """
    def __init__(self):
        self._registry: dict[str, ModelEntry] = {}
        self._active_id: Optional[str] = None
        self._lm_extractor = LandmarkExtractor()

    @property
    def active_model(self) -> Optional[ModelEntry]:
        if self._active_id is None:
            return None
        return self._registry[self._active_id]

    @property
    def active_id(self) -> Optional[str]:
        return self._active_id
    
    @property
    def class_map(self) -> dict[str, int]:
        entry = self.active_model
        if entry is None:
            raise RuntimeError("No active model found - check if the /models/load was called.")
        return entry.class_map
    
    @property
    def class_names(self) -> list[str]:
        return list(self.class_map.keys())

    @property
    def lm_extractor(self) -> LandmarkExtractor:
        return self._lm_extractor

    def get_model_list(self) -> list[dict]:
        return [
            {
                "model_id": entry.model_id,
                "class_count": len(entry.class_map),
                "device": entry.device,
                "is_active": (entry.model_id == self._active_id),
            }
            for entry in self._registry.values()
        ]
    
    def _load_classes(self, fpath: str | Path) -> dict[str, int]:
        """
        Load class map from a json for label decoding.
        """
        if not Path(fpath).exists():
            raise FileNotFoundError(f"Class map json not found at: {fpath}")
        with open(fpath, "r", encoding="utf-8-sig") as f:
            label_dict = json.load(f)
        class_map = {name: int(label) for name, label in label_dict.items()}
        return class_map


    def load(
        self,
        model_id: str,
        weights_path: str,
        class_map_path: str,
        save_type: SaveMethod = SaveMethod.FULL,
        device: str = "cpu",
        ) -> None:
        """
        Load a model from disk.
        Supported save methods:
        - FULL: full nn.Module (torch.save(model)).
        - WEIGHTS_ONLY: checkpoint dict with 'state_dict' AND 'config' - mirrors model architecture.
        - ONNX: via exported .onnx graph. 
        """

        path = MODEL_DIR / weights_path
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {weights_path}")
        class_map = self._load_classes(MODEL_DIR / class_map_path) 

        if save_type == SaveMethod.ONNX:
            wrapper = OnnxModelWrapper(str(path), device)
        elif save_type == SaveMethod.FULL:
            module = torch.load(path, map_location=device, weights_only=False)
            if not isinstance(module, nn.Module):
                raise ValueError(f"FULL save_type expects a pickled nn.Module at {weights_path}, got {type(module)}.")
            wrapper = TorchModelWrapper(module, device)
        elif save_type == SaveMethod.WEIGHTS_ONLY:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            module = self._build_model(checkpoint)
            wrapper = TorchModelWrapper(module, device)
        else:
            raise ValueError(f"Unsupported model save type: {save_type}")
        
        self._registry[model_id] = ModelEntry(model_id, wrapper, class_map, device)
        # Auto-activate first
        if self._active_id is None:
            self._active_id = model_id
    
    def _build_model(self, checkpoint: dict) -> nn.Module:
        """
        Builds the model from scratch. Requires a 'config' and 'state_dict' in the checkpoint payload.
        """
        if "config" not in checkpoint:
            raise ValueError( "Can't rebuild the model: WEIGHTS_ONLY checkpoint has no saved 'config'.")
        config = checkpoint["config"]

        # TODO: Need to load the model from weights - model class instance reconstructed from 'config' entries:
        # model = MyModel(num_classes=config['num_classes]...)
        # model.load_state_dict(checkpoint["state_dict"])
        # return model
        
        raise NotImplementedError("TBD: Loading models from weights is currently not supported.")
    

    def switch(self, model_id: str) -> None:
        """
        Switch active model
        """
        if model_id not in self._registry:
            raise KeyError(f"Model '{model_id}' is not loaded - load it with load() before calling.")
        self._active_id = model_id

    def unload(self, model_id: str) -> None:
        """
        Remove model from current registry
        """
        if model_id not in self._registry:
            raise KeyError(f"Can't unload model - model '{model_id}' not found in registry.")
        del self._registry[model_id]
        if self._active_id == model_id:
            self._active_id = next(iter(self._registry), None)


model_manager = ModelManager()