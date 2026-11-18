// from API's schemas:
// ---- Model management ----

export type TaskType = "ctc" | "classification";

export interface ModelInfo {
  model_id: string;
  device: string;
  is_active: boolean;
  task: TaskType;
  class_count: number;
}

export interface ModelListResponse {
  active_model: string | null;
  active_class_count: number;
  models: ModelInfo[];
}

export type SaveType = "FULL" | "WEIGHTS_ONLY" | "ONNX";

export interface LoadModelRequest {
  model_id: string;
  weights_path: string;
  class_map_path: string;
  save_type: SaveType;
  device: string;
  task: TaskType;
  blank_index: number;
}

// ---- Predictions ----

export interface GesturePrediction {
  label: string;
  class_index: number;
  confidence: number; // 0..1
  start_frame?: number | null;
  end_frame?: number | null;
}

export interface WindowPrediction {
  window_index: number;
  top1: GesturePrediction;
  top3: GesturePrediction[];
}

export interface SequencePrediction {
  segment_index: number;
  text: string;
  tokens: GesturePrediction[];
}

export interface PredictionInfo {
  active_model: string | null;
  n_frames: number;
  // TODO: this is NOT a part of the current backend schema 
  landmarks?: LandmarkFrame[];
}

export interface TopkPredictResponse extends PredictionInfo {
  task: "classification"; // code smell again
  windows: WindowPrediction[];
}

export interface CTCPredictResponse extends PredictionInfo {
  task: "ctc";
  sequences: SequencePrediction[];
}

export type PredictResponse = TopkPredictResponse | CTCPredictResponse;

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
  prediction?: PredictResponse;
  status?: string;
  error?: string;
}
