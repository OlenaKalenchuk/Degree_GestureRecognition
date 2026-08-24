import { useAppContext } from "@/core/AppContext";

// Switch between loaded models
export function ModelSelector() {
  const { models, activeModelId, isLoading, isSwitching, switchModel } = useAppContext();

  if (isLoading) return <span className="model-selector model-selector--muted">Loading models…</span>;
  if (models.length === 0) return <span className="model-selector model-selector--muted">No models loaded</span>;

  return (
    <label className="model-selector">
      <span className="model-selector-label">Model</span>
      <select
        value={activeModelId ?? ""}
        disabled={isSwitching}
        onChange={(e) => switchModel(e.target.value)}
        >
        {models.map((m) => (
          <option key={m.model_id} value={m.model_id}>
            {m.model_id} ({m.device})
          </option>
        ))}
      </select>
    </label>
  );
}
