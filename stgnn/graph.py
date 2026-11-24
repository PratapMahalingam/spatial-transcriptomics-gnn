"""Spatial neighbourhood graphs and the degree-preserving rewired null."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_edges(coords: np.ndarray, k: int = 6) -> np.ndarray:
    """Undirected k-nearest-neighbour graph as unique (u, v) pairs with u < v.

    Each spot is joined to its k nearest spots and the result is symmetrised, so
    every node has degree >= k. On the Visium hexagonal grid k = 6 is the ring of
    immediately adjacent spots.
    """
    n = len(coords)
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)
    # guard against duplicated coordinates putting a node outside its own first slot
    ok = rows != cols
    u, v = np.minimum(rows[ok], cols[ok]), np.maximum(rows[ok], cols[ok])
    return np.unique(np.stack([u, v], 1), axis=0)


def degrees(edges: np.ndarray, n_nodes: int) -> np.ndarray:
    return np.bincount(edges.reshape(-1), minlength=n_nodes)


def rewire_degree_preserving(edges: np.ndarray, n_nodes: int, rng: np.random.Generator,
                             swaps_per_edge: int = 10) -> tuple[np.ndarray, int]:
    """Randomise an undirected simple graph by double-edge swaps.

    (a, b), (c, d) -> (a, d), (c, b). Every node keeps its exact degree and the
    graph stays simple (no self-loops, no duplicate edges); swaps that would
    break that are rejected. With ``swaps_per_edge`` = 10 the graph is far past
    the point where it remembers the original wiring.

    Returns the new edge list (u < v) and the number of accepted swaps.
    """
    e = [(int(a), int(b)) for a, b in edges]
    m = len(e)
    if m < 2:
        return np.asarray(e, dtype=np.int64).reshape(-1, 2), 0
    n = int(n_nodes)

    def key(a, b):
        return a * n + b if a < b else b * n + a

    present = {key(a, b) for a, b in e}
    target, done, tries = swaps_per_edge * m, 0, 0
    max_tries = 20 * target
    chunk = 65536
    while done < target and tries < max_tries:
        ii = rng.integers(m, size=chunk)
        jj = rng.integers(m, size=chunk)
        flip = rng.random(chunk) < 0.5
        for i, j, f in zip(ii.tolist(), jj.tolist(), flip.tolist()):
            tries += 1
            if i == j:
                continue
            a, b = e[i]
            c, d = e[j]
            if f:
                c, d = d, c
            if a == d or c == b:
                continue
            k1, k2 = key(a, d), key(c, b)
            if k1 == k2 or k1 in present or k2 in present:
                continue
            present.discard(key(a, b))
            present.discard(key(c, d))
            present.add(k1)
            present.add(k2)
            e[i] = (a, d) if a < d else (d, a)
            e[j] = (c, b) if c < b else (b, c)
            done += 1
            if done >= target:
                break
    return np.asarray(e, dtype=np.int64), done


def edge_homophily(edges: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of edges joining two nodes with the same label."""
    if len(edges) == 0:
        return float("nan")
    return float(np.mean(labels[edges[:, 0]] == labels[edges[:, 1]]))


def boundary_mask(edges: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """True for nodes with at least one neighbour carrying a different label."""
    diff = labels[edges[:, 0]] != labels[edges[:, 1]]
    out = np.zeros(len(labels), dtype=bool)
    out[edges[diff, 0]] = True
    out[edges[diff, 1]] = True
    return out
