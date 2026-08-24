import { createContext, useContext, type ReactNode } from "react";
import { useModels } from "@/hooks/useModels";

type ModelsState = ReturnType<typeof useModels>;

const AppContext = createContext<ModelsState | null>(null);

// App wrapper so any component can read the global shared context
export function AppProvider({ children }: { children: ReactNode }) {
  const modelsState = useModels();
  return <AppContext.Provider value={modelsState}>{children}</AppContext.Provider>;
}

// Global app context
export function useAppContext(): ModelsState {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("App context must be used in AppProvider.");
  return ctx;
}
