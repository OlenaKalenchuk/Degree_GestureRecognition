// from API's schemas:
// ---- Model management ----

export interface ModelInfo {
  model_id: string;
  device: string;
  is_active: boolean;
}

export interface ModelListResponse {
  active_model: string | null;
  models: ModelInfo[];
  class_count: number;
}

export type SaveType = "FULL" | "WEIGHTS_ONLY" | "ONNX";

export interface LoadModelRequest {
  model_id: string;
  weights_path: string;
  class_map_path: string;
  save_type: SaveType;
  device: string;
}

// ---- Predictions ----

export interface GesturePrediction {
  label: string;
  class_index: number;
  confidence: number; // 0..1
}

export interface WindowPrediction {
  window_index: number;
  start_frame: number;
  end_frame: number;
  start_time?: number | null; // seconds
  end_time?: number | null; // seconds
  top1: GesturePrediction;
  top3: GesturePrediction[];
}

export interface PredictResponse {
  active_model: string | null;
  n_frames: number;
  windows: WindowPrediction[];
  // TODO: NOT part of the current backend schema 
  landmarks?: LandmarkFrame[];
}

// ---- Landmarks ----

// (N, 3) arrays with [x, y, z]
export interface LandmarkFrame {
  pose: number[][]; // may be abscent
  left_hand: number[][];
  right_hand: number[][];
}

export interface HandsPresent {
  left: boolean;
  right: boolean;
}

// ---- WebSocket camera ----

// client -> server
export type WebSockMessage = { frame: string } | { recognition: boolean };

// server -> client
export interface WebSockResponse {
  hands_present?: HandsPresent;
  landmarks?: LandmarkFrame;
  prediction?: {
    top1: GesturePrediction;
    top3: GesturePrediction[];
  };
  status?: string;
  error?: string;
}
