import { useCallback, useEffect, useState } from "react";
import { fetchModels, switchActiveModel } from "@features/model-switch/models";
import type { ModelInfo } from "@/types/api";

interface UseModelsResult {
  models: ModelInfo[];
  activeModelId: string | null;
  isLoading: boolean;
  isSwitching: boolean;
  error: string | null;
  switchModel: (modelId: string) => Promise<void>;
  refresh: () => Promise<void>;
}

// Loads the model list at startup
export function useModels(): UseModelsResult {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSwitching, setIsSwitching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchModels();
      setModels(data.models);
      setActiveModelId(data.active_model);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load models.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const switchModel = useCallback(async (modelId: string) => {
    setIsSwitching(true);
    setError(null);
    const previous = activeModelId;
    setActiveModelId(modelId);
    try {
      await switchActiveModel(modelId);
    } catch (err) {
      setActiveModelId(previous); // revert
      setError(err instanceof Error ? err.message : "Failed to switch model.");
    } finally {
      setIsSwitching(false);
    }
  }, [activeModelId]);

  return { models, activeModelId, isLoading, isSwitching, error, switchModel, refresh };
}
