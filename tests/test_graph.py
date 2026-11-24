from collections import Counter

import numpy as np
import pytest

from stgnn.graph import (boundary_mask, degrees, edge_homophily, knn_edges,
                         rewire_degree_preserving)
from stgnn.synthetic import hex_grid


def banded_graph(n_rows=20, n_cols=20):
    coords = hex_grid(n_rows, n_cols)
    labels = (coords[:, 0] // (coords[:, 0].max() / 4 + 1e-9)).astype(int)
    return coords, labels, knn_edges(coords, 6)


def test_knn_on_hex_grid_gives_six_neighbours_in_the_interior():
    coords = hex_grid(12, 12)
    e = knn_edges(coords, 6)
    deg = degrees(e, len(coords))
    r = np.repeat(np.arange(12), 12)
    c = np.tile(np.arange(12), 12)
    # edge spots have < 6 spots at one spacing and borrow farther ones, so check
    # spots at least three rows/cols in from the edge
    interior = (r > 2) & (r < 9) & (c > 2) & (c < 9)
    assert np.all(deg[interior] == 6)
    assert np.all(deg >= 6)
    # undirected, unique, no self-loops
    assert np.all(e[:, 0] < e[:, 1])
    assert len(np.unique(e, axis=0)) == len(e)
    # interior neighbours are at exactly one spot spacing
    d = np.linalg.norm(coords[e[:, 0]] - coords[e[:, 1]], axis=1)
    both_inner = interior[e[:, 0]] & interior[e[:, 1]]
    assert np.allclose(d[both_inner], 100.0)


@pytest.mark.parametrize("seed", range(4))
def test_rewiring_keeps_every_degree_and_stays_simple(seed):
    coords, labels, e = banded_graph()
    n = len(coords)
    new, n_swaps = rewire_degree_preserving(e, n, np.random.default_rng(seed), swaps_per_edge=5)
    assert n_swaps == 5 * len(e)
    assert np.array_equal(degrees(new, n), degrees(e, n))
    assert np.all(new[:, 0] < new[:, 1])                       # no self-loops, canonical
    assert len({tuple(x) for x in new.tolist()}) == len(new)    # no duplicate edges
    assert Counter(new.reshape(-1).tolist()) == Counter(e.reshape(-1).tolist())


def test_rewiring_destroys_spatial_structure():
    coords, labels, e = banded_graph()
    new, _ = rewire_degree_preserving(e, len(coords), np.random.default_rng(0))
    overlap = len({tuple(x) for x in e.tolist()} & {tuple(x) for x in new.tolist()}) / len(e)
    assert overlap < 0.05
    assert edge_homophily(e, labels) > 0.8
    chance = np.sum(np.bincount(labels) ** 2) / len(labels) ** 2
    assert abs(edge_homophily(new, labels) - chance) < 0.08
    d_old = np.linalg.norm(coords[e[:, 0]] - coords[e[:, 1]], axis=1).mean()
    d_new = np.linalg.norm(coords[new[:, 0]] - coords[new[:, 1]], axis=1).mean()
    assert d_new > 5 * d_old


def test_rewiring_is_reproducible():
    coords, _, e = banded_graph()
    a, _ = rewire_degree_preserving(e, len(coords), np.random.default_rng(11))
    b, _ = rewire_degree_preserving(e, len(coords), np.random.default_rng(11))
    assert np.array_equal(a, b)


def test_boundary_mask():
    labels = np.array([0, 0, 1, 1])
    e = np.array([[0, 1], [1, 2], [2, 3]])
    assert boundary_mask(e, labels).tolist() == [False, True, True, False]
