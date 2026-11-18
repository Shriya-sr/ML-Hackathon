from __future__ import annotations

from pathlib import Path
import numpy as np
from .data import Trip


def _route_parts(nodes: np.ndarray, arcs: np.ndarray, route: np.ndarray) -> list[list[list[float]]]:
    parts: list[list[list[float]]] = []
    current: list[list[float]] = []
    previous_end = None
    for arc_id in np.asarray(route, dtype=int):
        start, end = arcs[arc_id]
        if current and previous_end != start:
            if len(current) > 1:
                parts.append(current)
            current = []
        if not current:
            current.append(nodes[start, [1, 0]].tolist())
        current.append(nodes[end, [1, 0]].tolist())
        previous_end = end
    if len(current) > 1:
        parts.append(current)
    return parts


def save_trip_map(trip: Trip, predicted_arc_ids: np.ndarray | None, output: str | Path,
                  include_reference: bool = True) -> Path:
    try:
        import folium
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to generate Folium maps") from exc
    center = [float(np.mean(trip.gps[:, 1])), float(np.mean(trip.gps[:, 0]))]
    m = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")
    folium.PolyLine(trip.gps[:, [1, 0]].tolist(), color="#1769aa", weight=4, opacity=.8, tooltip="GPS observations").add_to(m)
    if include_reference and trip.route is not None:
        for coords in _route_parts(trip.nodes, trip.arcs, trip.route):
            folium.PolyLine(coords, color="#2e8b57", weight=5, opacity=.8, tooltip="Reference route").add_to(m)
    if predicted_arc_ids is not None:
        for coords in _route_parts(trip.nodes, trip.arcs, np.asarray(predicted_arc_ids, dtype=int)):
            folium.PolyLine(coords, color="#e53935", weight=5, opacity=.8, tooltip="Predicted route").add_to(m)
    folium.LayerControl().add_to(m)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); m.save(str(output))
    return output
