from pydantic import BaseModel, Field
from typing import Literal, Optional


# Model info responses
class ModelInfo(BaseModel):
    model_id: str
    device: str
    is_active: bool
    task: Literal["classification", "ctc"] = "classification"
    class_count: int

class ModelListResponse(BaseModel):
    active_model: Optional[str]
    active_class_count: int
    models: list[ModelInfo]


# Model requests
class LoadModelRequest(BaseModel):
    model_id: str = Field(..., description="Unique name for this model, e.g. 'my_rnn_v1'")
    weights_path: str = Field(..., description="Path to the .pt/.pth/.onnx file")
    class_map_path: str = Field(..., description="Path to the .json file")
    save_type: Literal["FULL", "WEIGHTS_ONLY", "ONNX"]  = Field(
        default= 'FULL',
        description= "Loader build method for the right wrapper. "
                     "Default value is FULL save - torch.save(model). "
                     "ONNX save - .onnx import with ort session. "
                     "WEIGHTS_ONLY save - state-dict and config."
    )
    device: str = Field(default="cpu", description="'cpu', 'cuda', or 'cuda:0'")
    task: Literal["classification", "ctc"] = Field(default="classification", description="Decides the output decoder and model type.")
    blank_index: int = Field(default=0, description="CTC specific: <blank> class index in the class map.")

class SwitchModelRequest(BaseModel):
    model_id: str = Field(..., description="model_id of a previously loaded model")


# Prediction responses:

# Something dedicated to mark the start/end times of window/sequence predictions with
class FrameTimeMark(BaseModel):
    start_frame: int
    end_frame: int
    start_time: Optional[float] = None # s
    end_time: Optional[float] = None # s

# General prediction class
class GesturePrediction(BaseModel):
    label: str
    class_index: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

# Window-based classification prediciton
class WindowPrediction(BaseModel):
    window_index: int
    top1: GesturePrediction # single max prediction
    top3: list[GesturePrediction] # top k predictions

# CTC-based prediction
class SequencePrediction(BaseModel):
    segment_index: int
    text: str # complete sequence prediction
    tokens: list[GesturePrediction] # the decoded tokens which correspond to a single sequence

class PredictResponse(BaseModel):
    active_model: Optional[str]
    n_frames: int
    windows: list[WindowPrediction]
    sequences: list[SequencePrediction] # TODO: maybe separate it to a different response? Code smell ._.
    task: Literal["classification", "ctc"] = "classification"