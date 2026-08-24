import { AppProvider } from "@/core/AppContext";
import { useState } from "react";
import { SplitPanel } from "../components/layout/SplitColumnPanel";
import { VideoPanel } from "@features/video/VideoPanel";
import { PredictionPanel } from "@features/prediction/PredictionPanel";
import type { PredictionEntry } from "@/utils/predictionHelpers";
import { PageHeader, PageBody } from "../components/layout/PageLayout";
// For now, style imports will be like this:
import "@/styles/theme.css";
import "@/styles/layout.css";
import "@/styles/video.css";
import "@/styles/predictions.css";

/*
 * This is the Main UI for the recognition API.
 * API base URL is read from the VITE_API_URL.
 */

/**
 * Context - whatever is used by element/function
 * Result - hook return value
 */

export default function RecognitionApp() {
    const [entries, setEntries] = useState<PredictionEntry[]>([]);

    return (
        <AppProvider>
            <div className="app-root">
                <PageHeader />
                <PageBody>
                    <SplitPanel
                    left={<VideoPanel onEntriesChange={setEntries} />}
                    right={<PredictionPanel entries={entries} />}
                    />
                </PageBody>
            </div>
        </AppProvider>
    );
}