import { apiClient } from "@features/api/client"
import type { ModelListResponse, LoadModelRequest } from "@/types/api";

export async function fetchModels(): Promise<ModelListResponse> {
  const { data } = await apiClient.get<ModelListResponse>("/models");
  return data;
}

export async function switchActiveModel(modelId: string): Promise<void> {
  await apiClient.put("/models/active", { model_id: modelId });
}

// Just for convenience, not used:
export async function unloadModel(modelId: string): Promise<void> {
  await apiClient.delete(`/models/${modelId}`);
}

export async function loadModel(req: LoadModelRequest): Promise<void> {
  await apiClient.post("/models/load", req);
}