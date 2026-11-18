import type { Explanation, Health, ModelInfo, Prediction, SystemInfo, TripSummary } from "./types";

const STORAGE_KEY = "pathsnap.apiUrl";

export function defaultApiUrl(): string {
  const env = (import.meta.env.VITE_API_URL as string | undefined)?.trim();
  if (env) return env.replace(/\/$/, "");
  const host = typeof window !== "undefined" && window.location.hostname ? window.location.hostname : "localhost";
  return `http://${host}:8000`;
}

export function loadApiUrl(): string {
  try {
    const v = window.localStorage.getItem(STORAGE_KEY);
    if (v) return v;
  } catch {
    /* storage unavailable (private mode) - fall back to default */
  }
  return defaultApiUrl();
}

export function saveApiUrl(url: string) {
  try {
    window.localStorage.setItem(STORAGE_KEY, url);
  } catch {
    /* ignore */
  }
}

export class ApiError extends Error {
  constructor(message: string, public status: number | null) {
    super(message);
  }
}

async function request<T>(base: string, path: string, init?: RequestInit, timeoutMs = 15000): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(`${base}${path}`, { ...init, signal: ctrl.signal });
  } catch (e) {
    const aborted = (e as Error).name === "AbortError";
    throw new ApiError(
      aborted ? `Request timed out after ${Math.round(timeoutMs / 1000)} s` : `Cannot reach the API at ${base}`,
      null,
    );
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(`${res.status}: ${detail}`, res.status);
  }
  try {
    return (await res.json()) as T;
  } catch {
    throw new ApiError("Malformed response from the API", res.status);
  }
}

export const api = {
  health: (b: string) => request<Health>(b, "/health", undefined, 5000),
  system: (b: string) => request<SystemInfo>(b, "/api/system/info"),
  model: (b: string) => request<ModelInfo>(b, "/api/model/info"),
  trips: (b: string) => request<TripSummary[]>(b, "/api/trips", undefined, 30000),
  predict: (b: string, id: string) =>
    request<Prediction>(b, `/api/predict/${encodeURIComponent(id)}`, { method: "POST" }, 180000),
  explain: (b: string, id: string, fix: number) =>
    request<Explanation>(b, `/api/results/${encodeURIComponent(id)}/explain/${fix}`),
};
