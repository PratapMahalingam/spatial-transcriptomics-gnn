"""Synthetic layered tissue for the smoke test. NOT real data.

Spots sit on a Visium-like hexagonal grid. Layers are horizontal bands with wavy
boundaries; each layer has its own weak expression programme, so a single spot
is noisy but its neighbours carry the same signal. Donors add a gene-wise shift.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .data import Section


def hex_grid(n_rows: int, n_cols: int, spacing: float = 100.0) -> np.ndarray:
    r, c = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    x = (c + 0.5 * (r % 2)) * spacing
    y = r * spacing * np.sqrt(3) / 2
    return np.stack([y.ravel(), x.ravel()], 1).astype(float)


def make_synthetic(n_donors: int = 3, sections_per_donor: int = 2, n_rows: int = 30, n_cols: int = 30,
                   n_genes: int = 300, n_layers: int = 5, signal: float = 0.15, seed: int = 0) -> list[Section]:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n_genes)
    programme = rng.normal(0.0, signal, (n_layers, n_genes))
    gene_ids = np.array([f"SYNTH{g:04d}" for g in range(n_genes)])
    sections = []
    for d in range(n_donors):
        donor_shift = rng.normal(0.0, 0.2, n_genes)
        for s in range(sections_per_donor):
            coords = hex_grid(n_rows, n_cols)
            y, x = coords[:, 0], coords[:, 1]
            depth = y / y.max() + 0.06 * np.sin(2 * np.pi * x / x.max() * rng.uniform(0.5, 1.5) + rng.uniform(0, 6.3))
            layer = np.clip((depth * n_layers).astype(int), 0, n_layers - 1)
            log_rate = base + programme[layer] + donor_shift + rng.normal(0, 0.3, (len(layer), n_genes))
            counts = rng.poisson(np.exp(log_rate)).astype(np.float32)
            name = f"synth_d{d}_s{s}"
            sections.append(Section(
                name=name, donor=f"donor{d}", counts=sp.csr_matrix(counts), coords=coords,
                labels=np.array([f"L{k + 1}" for k in layer]),
                barcodes=np.array([f"{name}_{i}" for i in range(len(layer))]), gene_ids=gene_ids,
            ))
    return sections
