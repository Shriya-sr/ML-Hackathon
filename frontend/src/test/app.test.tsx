import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import TripList from "../components/TripList";
import MetricsPanel from "../components/MetricsPanel";
import App from "../App";
import { routePolylines, projectOnSegment, pct } from "../geo";
import { prediction, trips } from "./fixtures";

vi.mock("../components/MapView", () => ({
  default: ({ pred, loading }: { pred: unknown; loading: boolean }) => (
    <div data-testid="map">{loading ? "map-loading" : pred ? "map-with-route" : "map-empty"}</div>
  ),
}));

describe("TripList", () => {
  it("renders every trip and reports the selection", () => {
    const onSelect = vi.fn();
    render(<TripList trips={trips} selected={null} loading={false} onSelect={onSelect} error={null} conn="online" onRetry={() => {}} />);
    expect(screen.getByText("Trip 00000052")).toBeInTheDocument();
    expect(screen.getByText("Trip 00000077")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Trip 00000077"));
    expect(onSelect).toHaveBeenCalledWith("00000077");
  });

  it("shows an offline state with retry", () => {
    const onRetry = vi.fn();
    render(<TripList trips={[]} selected={null} loading={false} onSelect={() => {}} error="Cannot reach the API" conn="offline" onRetry={onRetry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Cannot reach the API");
    fireEvent.click(screen.getByText("Retry connection"));
    expect(onRetry).toHaveBeenCalled();
  });

  it("disables buttons while a prediction is running", () => {
    render(<TripList trips={trips} selected="00000052" loading onSelect={() => {}} error={null} conn="online" onRetry={() => {}} />);
    expect(screen.getAllByRole("button")[0]).toBeDisabled();
  });
});

describe("MetricsPanel", () => {
  it("shows server metrics and the HMM delta", () => {
    render(<MetricsPanel pred={prediction} loading={false} />);
    expect(screen.getAllByText("100.0%").length).toBeGreaterThan(0);
    expect(screen.getByText("+33.3 pts")).toBeInTheDocument();
  });
  it("states clearly when metrics are unavailable", () => {
    render(<MetricsPanel pred={{ ...prediction, metrics: { available: false, reason: "no reference route" } }} loading={false} />);
    expect(screen.getByText(/Metrics unavailable/)).toBeInTheDocument();
  });
  it("renders an empty state", () => {
    render(<MetricsPanel pred={null} loading={false} />);
    expect(screen.getByText(/Results appear here/)).toBeInTheDocument();
  });
});

describe("geo helpers", () => {
  it("joins touching arcs and splits at jumps", () => {
    const lines = routePolylines([1, 2, 3], prediction.arcs);
    expect(lines.length).toBe(2);
    expect(lines[0].length).toBe(3);
  });
  it("projects onto a segment", () => {
    const p = projectOnSegment([1, 0.5], [0, 0], [0, 1]);
    expect(p[0]).toBeCloseTo(0);
    expect(p[1]).toBeCloseTo(0.5);
  });
  it("formats missing values as n/a", () => {
    expect(pct(null)).toBe("n/a");
    expect(pct(0.5)).toBe("50.0%");
  });
});

describe("App integration (mocked fetch)", () => {
  const ok = (body: unknown) => Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn((url: string, init?: RequestInit) => {
      if (url.endsWith("/health")) return ok({ status: "ok", model_loaded: true, mode: "DEVELOPMENT", n_trips: 2, uptime_s: 1 });
      if (url.endsWith("/api/system/info")) return ok({ mode: "DEVELOPMENT", partition: "student_test", n_trips: 2, api_version: "1", note: "dev" });
      if (url.endsWith("/api/model/info")) return ok({ model: "m", feature_version: "v1", n_features: 21, hmm_params: {} });
      if (url.endsWith("/api/trips")) return ok(trips);
      if (url.includes("/api/predict/00000052") && init?.method === "POST") return ok(prediction);
      if (url.includes("/api/predict/")) return Promise.resolve(new Response(JSON.stringify({ detail: "unknown trip id" }), { status: 404 }));
      return Promise.reject(new TypeError("network"));
    }));
  });
  afterEach(() => vi.unstubAllGlobals());

  it("connects, lists trips, runs a prediction and shows results", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByTestId("conn-pill")).toHaveTextContent("API online"));
    expect(screen.getByText(/Development mode/)).toBeInTheDocument();
    fireEvent.click(await screen.findByText("Trip 00000052"));
    await waitFor(() => expect(screen.getByTestId("map")).toHaveTextContent("map-with-route"));
    expect(screen.getByText("Classifier-only vs classifier + HMM")).toBeInTheDocument();
  });

  it("shows an error with retry when the API rejects a trip", async () => {
    render(<App />);
    fireEvent.click(await screen.findByText("Trip 00000077"));
    await waitFor(() => expect(screen.getByText(/Prediction failed/)).toBeInTheDocument());
    expect(screen.getByText("Retry")).toBeInTheDocument();
  });

  it("reports an unreachable API", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("network"))));
    render(<App />);
    await waitFor(() => expect(screen.getByTestId("conn-pill")).toHaveTextContent("API offline"));
    expect(screen.getByText("Retry connection")).toBeInTheDocument();
  });
});
