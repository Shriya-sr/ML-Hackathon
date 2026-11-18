"""Directed road network in projected metres.

* nodes  -> (x, y)
* arcs   -> directed straight segments from_node -> to_node, arc_id = row index
* CSR adjacency for bounded shortest-path queries used by the HMM transitions
"""
from __future__ import annotations

import heapq

import numpy as np

from .distance import bearing_xy
from .projection import LocalProjector
from .spatial_index import SegmentIndex


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, projector: LocalProjector):
        self.projector = projector
        self.nodes_lonlat = nodes_lonlat
        self.arcs = arcs
        self.V, self.E = len(nodes_lonlat), len(arcs)
        self.nx, self.ny = projector.to_xy(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        u, v = arcs[:, 0], arcs[:, 1]
        self.u, self.v = u, v
        self.ax, self.ay = self.nx[u], self.ny[u]
        self.bx, self.by = self.nx[v], self.ny[v]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing_xy(self.bx - self.ax, self.by - self.ay)
        self.out_degree = np.bincount(u, minlength=self.V)
        self.in_degree = np.bincount(v, minlength=self.V)
        # reverse-arc lookup (u,v) -> arc id of (v,u), -1 if one-way
        key = u.astype(np.int64) * self.V + v
        rkey = v.astype(np.int64) * self.V + u
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rkey)
        pos = np.clip(pos, 0, self.E - 1)
        hit = key[order][pos] == rkey
        self.reverse_arc = np.where(hit, order[pos], -1)
        # CSR adjacency (from-node -> outgoing arcs)
        o = np.argsort(u, kind="stable")
        self.csr_arcs = o
        self.csr_ptr = np.concatenate([[0], np.cumsum(self.out_degree)])
        self._index: SegmentIndex | None = None
        self._sp_cache: dict[int, tuple[float, dict[int, float], dict[int, int]]] = {}

    @property
    def index(self) -> SegmentIndex:
        if self._index is None:
            self._index = SegmentIndex(self.ax, self.ay, self.bx, self.by)
        return self._index

    # ------------------------------------------------------------------ paths
    def shortest_from(self, source: int, limit: float) -> dict[int, float]:
        """Bounded Dijkstra from ``source`` node; returns {node: metres} for all
        nodes reachable within ``limit``. Results are cached per source and reused
        whenever a cached search covered at least the requested limit."""
        return self._search(source, limit)[0]

    def _search(self, source: int, limit: float):
        cached = self._sp_cache.get(source)
        if cached is not None and cached[0] >= limit:
            return cached[1], cached[2]
        dist = {source: 0.0}
        pred: dict[int, int] = {}          # node -> arc used to reach it
        heap = [(0.0, source)]
        ptr, csr, vv, ln = self.csr_ptr, self.csr_arcs, self.v, self.length
        done = set()
        while heap:
            d, n = heapq.heappop(heap)
            if n in done:
                continue
            done.add(n)
            for k in range(ptr[n], ptr[n + 1]):
                a = int(csr[k])
                m = int(vv[a])
                nd = d + ln[a]
                if nd <= limit and nd < dist.get(m, np.inf):
                    dist[m] = nd
                    pred[m] = a
                    heapq.heappush(heap, (nd, m))
        self._sp_cache[source] = (limit, dist, pred)
        return dist, pred

    def connecting_arcs(self, arc_a: int, arc_b: int, limit: float = 4000.0) -> list[int] | None:
        """Intermediate arcs on the shortest drivable path from the end of arc_a to
        the start of arc_b ([] if directly connected, None if unreachable)."""
        src, dst = int(self.v[arc_a]), int(self.u[arc_b])
        if src == dst:
            return []
        dist, pred = self._search(src, limit)
        if dst not in dist:
            return None
        path, n = [], dst
        while n != src:
            a = pred[n]
            path.append(a)
            n = int(self.u[a])
        return path[::-1]

    def network_distance(self, arc_a: int, frac_a: float, arc_b: int, frac_b: float,
                         limit: float) -> float:
        """Driving distance from position ``frac_a`` on arc_a to ``frac_b`` on arc_b.
        Returns inf if arc_b is not reachable within ``limit`` metres.

        Same arc: |frac_b - frac_a| * len (small backward moves are GPS noise)."""
        if arc_a == arc_b:
            return abs(frac_b - frac_a) * self.length[arc_a]
        head = (1.0 - frac_a) * self.length[arc_a]
        tail = frac_b * self.length[arc_b]
        budget = limit - head - tail
        if budget < 0:
            return np.inf
        reach = self.shortest_from(int(self.v[arc_a]), max(budget, 50.0))
        mid = reach.get(int(self.u[arc_b]))
        if mid is None or mid > budget:
            return np.inf
        return head + mid + tail

    def connected(self, arc_a: int, arc_b: int) -> bool:
        """True if arc_b can directly follow arc_a (shared node, direction-consistent)."""
        return bool(arc_a == arc_b or self.v[arc_a] == self.u[arc_b])

    def arc_lonlat(self, arc_ids):
        arc_ids = np.asarray(arc_ids, dtype=np.int64)
        a = self.nodes_lonlat[self.u[arc_ids]]
        b = self.nodes_lonlat[self.v[arc_ids]]
        return a, b
