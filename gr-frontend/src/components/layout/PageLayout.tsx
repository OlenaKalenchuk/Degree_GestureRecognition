import { ModelSelector } from "@features/model-switch/ModelSelector";

export function PageBody({ children }: { children: React.ReactNode }) {
  return (
    <main className="app-body">
        {children}
    </main>
  );
}

export function PageHeader() {
  return (
    <header className="app-header">
      <h1 className="app-title">Gesture Recognition</h1>
      <ModelSelector />
    </header>
  );
}
