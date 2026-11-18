"""Interactive Folium maps built from real data and real model outputs.

Layers (each toggleable in the LayerControl):
  * road network near the trajectory (grey)
  * raw GPS fixes (blue)
  * reference route (green)            - only for offline development maps
  * classifier-only prediction (orange, dashed)
  * HMM reconstructed route (purple) with direction arrows
  * start / end markers, GPS-gap markers, HMM chain breaks
"""
from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
from folium.plugins import PolyLineTextPath

COL = {"gps": "#1f6feb", "network": "#8a8a8a", "ref": "#1a9850", "clf": "#f46d43", "hmm": "#7b3294",
       "gap": "#d73027", "break": "#e7298a"}


def _segments(net, arc_ids):
    a, b = net.arc_lonlat(np.asarray(arc_ids, dtype=np.int64))
    return [[(float(a[i, 1]), float(a[i, 0])), (float(b[i, 1]), float(b[i, 0]))] for i in range(len(a))]


def _route_lines(net, route):
    """Join consecutive touching arcs into polylines (breaks where they do not touch)."""
    lines, cur, last_v = [], [], None
    for arc in route:
        u, v = int(net.u[arc]), int(net.v[arc])
        pu = (float(net.nodes_lonlat[u, 1]), float(net.nodes_lonlat[u, 0]))
        pv = (float(net.nodes_lonlat[v, 1]), float(net.nodes_lonlat[v, 0]))
        if last_v is not None and u == last_v:
            cur.append(pv)
        else:
            if len(cur) > 1:
                lines.append(cur)
            cur = [pu, pv]
        last_v = v
    if len(cur) > 1:
        lines.append(cur)
    return lines


def _legend(items):
    rows = "".join(f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{n}</div>' for n, c in items)
    return folium.Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;'
        f'border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);font:12px sans-serif">{rows}</div>')


def trip_map(lat, lon, net, title: str, near_arcs=None, reference=None, clf_route=None, hmm_route=None,
             gaps=None, breaks=None, max_points: int = 4000, max_network: int = 6000, gps_radius: float = 2,
             gps_opacity: float = 0.8) -> folium.Map:
    lat, lon = np.asarray(lat), np.asarray(lon)
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=14, tiles="OpenStreetMap",
                   control_scale=True)
    m.get_root().html.add_child(folium.Element(
        f'<h4 style="position:fixed;top:8px;left:60px;z-index:9999;background:white;padding:4px 10px;'
        f'border-radius:4px;font:600 14px sans-serif">{title}</h4>'))
    legend = []
    if near_arcs is not None and len(near_arcs):
        arcs = np.asarray(near_arcs)
        rev = net.reverse_arc[arcs]
        arcs = arcs[(rev < 0) | (arcs < rev)][:max_network]
        fg = folium.FeatureGroup(name="Road network (near trajectory)", show=True)
        for seg in _segments(net, arcs):
            folium.PolyLine(seg, color=COL["network"], weight=1.5, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("Road network", COL["network"]))
    if reference is not None:
        fg = folium.FeatureGroup(name="Reference route", show=True)
        for line in _route_lines(net, reference):
            folium.PolyLine(line, color=COL["ref"], weight=6, opacity=0.6, tooltip="reference route").add_to(fg)
        fg.add_to(m)
        legend.append(("Reference route", COL["ref"]))
    if clf_route is not None:
        fg = folium.FeatureGroup(name="Classifier-only prediction", show=True)
        for line in _route_lines(net, clf_route):
            folium.PolyLine(line, color=COL["clf"], weight=3, opacity=0.9, dash_array="6,6",
                            tooltip="classifier-only").add_to(fg)
        fg.add_to(m)
        legend.append(("Classifier-only route", COL["clf"]))
    if hmm_route is not None:
        fg = folium.FeatureGroup(name="HMM reconstructed route", show=True)
        for line in _route_lines(net, hmm_route):
            pl = folium.PolyLine(line, color=COL["hmm"], weight=4, opacity=0.9, tooltip="HMM + Viterbi")
            pl.add_to(fg)
            if len(line) > 3:
                PolyLineTextPath(pl, "    ►    ", repeat=True, offset=5,
                                 attributes={"fill": COL["hmm"], "font-size": "11"}).add_to(fg)
        fg.add_to(m)
        legend.append(("HMM reconstructed route", COL["hmm"]))
    step = max(1, int(np.ceil(len(lat) / max_points)))
    fg = folium.FeatureGroup(name="Raw GPS fixes", show=True)
    for i in range(0, len(lat), step):
        folium.CircleMarker([float(lat[i]), float(lon[i])], radius=gps_radius, color=COL["gps"], fill=True,
                            fill_opacity=gps_opacity, weight=0, tooltip=f"fix {i}").add_to(fg)
    fg.add_to(m)
    legend.append(("Raw GPS fix", COL["gps"]))
    folium.Marker([float(lat[0]), float(lon[0])], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([float(lat[-1]), float(lon[-1])], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    if gaps:
        fg = folium.FeatureGroup(name="GPS gaps > 30 s", show=True)
        for i, dt in gaps:
            folium.CircleMarker([float(lat[i]), float(lon[i])], radius=9, color=COL["gap"], fill=False, weight=2,
                                tooltip=f"gap {dt:.0f} s before fix {i}").add_to(fg)
        fg.add_to(m)
        legend.append(("GPS gap", COL["gap"]))
    if breaks:
        fg = folium.FeatureGroup(name="HMM chain breaks", show=True)
        for i, reason in breaks:
            folium.CircleMarker([float(lat[i]), float(lon[i])], radius=7, color=COL["break"], fill=True,
                                fill_opacity=0.3, tooltip=f"chain restart at fix {i}: {reason}").add_to(fg)
        fg.add_to(m)
        legend.append(("HMM chain break", COL["break"]))
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]], padding=(20, 20))
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(legend))
    return m


def save_map(m: folium.Map, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
    return path
