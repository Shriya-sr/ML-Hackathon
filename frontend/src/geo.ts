import type { LatLng } from "./types";

/** Project a point onto segment AB in a local equirectangular frame.
 *  Used only for the visual "snap" animation of already-predicted arcs. */
export function projectOnSegment(p: LatLng, a: LatLng, b: LatLng): LatLng {
  const k = Math.cos((p[0] * Math.PI) / 180);
  const ax = a[1] * k, ay = a[0], bx = b[1] * k, by = b[0], px = p[1] * k, py = p[0];
  const dx = bx - ax, dy = by - ay;
  const L2 = dx * dx + dy * dy;
  let t = L2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / L2 : 0;
  t = Math.max(0, Math.min(1, t));
  return [ay + t * dy, (ax + t * dx) / k];
}

export function lerp(a: LatLng, b: LatLng, t: number): LatLng {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

export function bearingDeg(a: LatLng, b: LatLng): number {
  const k = Math.cos((a[0] * Math.PI) / 180);
  return (Math.atan2((b[1] - a[1]) * k, b[0] - a[0]) * 180) / Math.PI;
}

/** Split an arc sequence into drawable polylines, breaking where consecutive arcs do not touch. */
export function routePolylines(route: number[], arcs: Record<string, [LatLng, LatLng]>): LatLng[][] {
  const lines: LatLng[][] = [];
  let cur: LatLng[] = [];
  for (const id of route) {
    const g = arcs[String(id)];
    if (!g) continue;
    const [a, b] = g;
    const last = cur[cur.length - 1];
    if (last && Math.abs(last[0] - a[0]) < 1e-7 && Math.abs(last[1] - a[1]) < 1e-7) {
      cur.push(b);
    } else {
      if (cur.length > 1) lines.push(cur);
      cur = [a, b];
    }
  }
  if (cur.length > 1) lines.push(cur);
  return lines;
}

export const pct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "n/a" : `${(v * 100).toFixed(d)}%`;
