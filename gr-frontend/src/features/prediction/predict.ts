import { apiClient } from "@features/api/client";
import type { PredictResponse } from "@/types/api";

// Predict a full, single video file upload: POST /predict/file
export async function predictFile(file: File): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);

  const { data } = await apiClient.post<PredictResponse>("/predict/file", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}
