from pydantic import BaseModel, Field
from typing import Literal, Optional


# Model info responses
class ModelInfo(BaseModel):
    model_id: str
    device: str
    is_active: bool

class ModelListResponse(BaseModel):
    active_model: Optional[str]
    models: list[ModelInfo]
    class_count: int


# Model requests
class LoadModelRequest(BaseModel):
    model_id: str = Field(..., description="Unique name for this model, e.g. 'my_rnn_v1'")
    weights_path: str = Field(..., description="Path to the .pt/.pth file")
    class_map_path: str = Field(..., description="Path to the .json file")
    save_type: Literal["FULL", "WEIGHTS_ONLY"]  = Field(
        default= 'FULL',
        description= "Loader build method for the right wrapper. "
                     "Default value is FULL save - torch.save(model). "
                     "ONNX save - .onnx import with ort session. "
                     "WEIGHTS_ONLY save - state-dict and config."
    )
    device: str = Field(default="cpu", description="'cpu', 'cuda', or 'cuda:0'")

class SwitchModelRequest(BaseModel):
    model_id: str = Field(..., description="model_id of a previously loaded model")


# Prediction responses
class GesturePrediction(BaseModel):
    label: str
    class_index: int
    confidence: float = Field(..., ge=0.0, le=1.0)

class WindowPrediction(BaseModel):
    window_index: int
    start_frame: int
    end_frame: int
    start_time: Optional[float] = None # s
    end_time: Optional[float] = None # s
    top1: GesturePrediction # single max prediction
    top3: list[GesturePrediction] # top k predictions

class PredictResponse(BaseModel):
    active_model: Optional[str]
    n_frames: int
    windows: list[WindowPrediction]