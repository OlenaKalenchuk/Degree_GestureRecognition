import axios from "axios";

/**
 * This is where the API connection is handled.
 * If there's something wrong with it, check the port in the configs.
 * Endpoints are in other features.
*/

// VITE_API_BASE_URL by default, otherwise fall back to "/api"
const baseURL = import.meta.env.VITE_API_BASE_URL
  ? `${import.meta.env.VITE_API_BASE_URL}`
  : "/api";

export const apiClient = axios.create({
  baseURL,
  timeout: 30_000
});

// WebSocket URL
export function getWsBaseUrl(): string {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL.replace(/^http/, "ws");
  }
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws`;
}
