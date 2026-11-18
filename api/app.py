import asyncio
import base64
import json
import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import logging
from contextlib import asynccontextmanager

import numpy as np

from api.mediapipe_detection import LandmarkExtractor
from api.prediction.cam_session import LiveDetectionSession
from api.prediction.model_utils import SaveMethod, preload_models, model_manager
from api.prediction.predict_video import predict_from_live, predict_from_video
from api.schemas import LoadModelRequest, ModelListResponse, PredictResponse, SwitchModelRequest
from api.video_processing.video_buffers import video_to_tensors

# ================ Init =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gesture_api")
lm_extractor_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Start up: preloading models from models_config.json..")
    results = preload_models()
 
    for model_id, status in results.items():
        level = logging.INFO if status == "loaded" else logging.WARNING
        logger.log(level, "  %-30s %s", model_id, status)
 
    yield
 
    # Shutdown
    logger.info("Shutting down..")

app = FastAPI(
    title="Recognition API",
    version="1.0.0",
    description=(
        "API managing gesture prediction from uploaded video or live camera frames. "
        "Built with PyTorch models and FastAPI."
    ),
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================ Endpoints =================

@app.get("/health", summary="Endpoint health check")
def health() -> JSONResponse:
    active = model_manager.active_model
    return JSONResponse(
        content = {
            "status": "ok",
            "active_model": model_manager.active_id,
            "models": len(model_manager.get_model_list()),
            "classes": len(active.class_map) if active else 0
        },
    status_code=200
    )


# Model-specific endpoints ---

@app.post("/models/load", summary="Load a model from weights")
def load_model(req: LoadModelRequest) -> dict:
    """
    Register a new model through .pt file.

    - **model_id**: alias.
    - **weights_path**: save path the server can reach.
    - **class_map_path**: json label map for requested model
    - **save_type**: "FULL" (entire model object), "WEIGHTS_ONLY" (state-dict), "ONNX" (.onnx graph).
    - **device**: "cpu", "cuda", or "cuda:0".
    """
    try:
        save_method = SaveMethod[req.save_type] # "FULL" → SaveMethod.FULL
        model_manager.load(
            model_id=req.model_id,
            weights_path=req.weights_path,
            class_map_path=req.class_map_path,
            save_type=save_method,
            device=req.device,
        )
        logger.info("Loaded model '%s' from '%s'", req.model_id, req.weights_path)

        return {
            "status": "success",
            "model_id": req.model_id,
            "active_model": model_manager.active_id,
        }
    
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc


@app.get("/models", response_model=ModelListResponse, summary="List loaded models")
def list_models() -> ModelListResponse:
    active = model_manager.active_model
    return ModelListResponse(
        active_model=model_manager.active_id,
        active_class_count=len(active.class_map) if active else 0, # CURRENT
        models=model_manager.get_model_list() # type: ignore
    )

@app.put("/models/active", summary="Switch the active model")
def switch_model(req: SwitchModelRequest) -> dict:
    """Switch which model is used for predictions."""
    try:
        model_manager.switch(req.model_id)
        return {"status": "switched", "active_model": model_manager.active_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/models/{model_id}", summary="Unload a model, removing it from list")
def unload_model(model_id: str) -> dict:
    try:
        model_manager.unload(model_id)
        return {"status": "removed", "active_model": model_manager.active_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# Prediction endpoints ---
@app.post(
    "/predict/file",
    response_model=PredictResponse,
    summary="Predict gestures from an uploaded video file",
)
async def predict_file(file: UploadFile = File(..., description="Video file (mp4, avi, mov, …)")) -> PredictResponse:
    """
    Upload a video file; the API segments the clip into sliding windows with a frame limit and by hand presence.
    Returns a prediction per segment.
    """
    if model_manager.active_model is None:
        raise HTTPException(status_code=503, detail="Can't predict - No model loaded.")
 
    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Can't predict - Uploaded file is empty.")
 
    try:
        async with lm_extractor_lock:
            response = predict_from_video(video_bytes, model_manager.lm_extractor)
        preds_len = len(response.windows) if response.task == "classification" else len(response.sequences)
        logger.info(
            "File '%s': %d frames -> %d gesture window(s)",
            file.filename, response.n_frames, preds_len,
        )
        return response
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.websocket("/ws/predict/camera")
async def ws_camera_predict(
    websocket: WebSocket,
    enable_landmarks: bool = Query(
        default=True,
        description="Include frame landmarks in each response, for overlay.",
        )
    ) -> None:
    """
    WebSocket endpoint for real-time camera gesture recognition.
    Uses frame buffering. Client can send the camera frames, or toggle prediction `{"recognition": true|false}`,
    without turning off landmark detection.
    Returns **per-frame** json:
    ```
    {
      "hands_present": {"left": bool, "right": bool},
      "landmarks": {"pose": [...], "left_hand": [...], "right_hand": [...]},
      "prediction": {"top1": {...}, "top3": [...]} # if prediction is enabled at all
    }
    # CTC send {"task": "CTC", "text": "...", "tokens": [...]} instead of topk
    ```
    """
    await websocket.accept()
    live_session = LiveDetectionSession()
    live_session.toggle_recognition(True)
    logger.info("WebSocket camera session opened!")
 
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
 
            if "recognition" in msg:
                live_session.toggle_recognition(bool(msg["recognition"]))
                logger.info(f"Recognition toggled: {msg["recognition"]}")
                await websocket.send_json({"status": "recognition " + ("enabled" if live_session.enable_recognition else "disabled")})
                continue
 
            b64 = msg.get("frame")
            if not b64:
                await websocket.send_json({"error": "Missing 'frame' key."})
                continue
 
            img_bytes = base64.b64decode(b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                await websocket.send_json({"error": "Could not decode frame."})
                continue

            entry = model_manager.active_model
            if entry is not None:
                live_session.switch_task_to(entry.task, entry.use_pose)
            result = live_session.process_frame(frame_bgr)
            response = {"hands_present": result["hands_present"]}
            if enable_landmarks:
                response["landmarks"] = {k: v.tolist() for k, v in result["landmarks"].items()}
 
            if result["video_result"] is not None:
                if entry is None:
                    response["error"] = "No model loaded on server."
                else:
                    response["prediction"] = predict_from_live(result["video_result"], live_session.frame_limit).model_dump(mode="json")
 
            await websocket.send_json(response)
 
    except WebSocketDisconnect:
        logger.info("WebSocket camera session closed.")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        await websocket.close(code=1011)
    finally:
        live_session.close()  