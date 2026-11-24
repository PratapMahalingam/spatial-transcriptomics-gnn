"""Loading the LIBD human DLPFC Visium sections (Maynard et al., 2021).

Each section needs three public files:

* ``{sample}_filtered_feature_bc_matrix.h5``   10x Space Ranger count matrix (AWS)
* ``{sample}_tissue_positions_list.txt``       spot coordinates (HumanPilot GitHub)
* ``barcode_level_layer_map.tsv``              manual layer annotation, all sections
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Sample -> donor, from the Maynard et al. (2021) study design: 3 donors x 4 sections.
SAMPLE_DONOR = {
    "151507": "Br5292", "151508": "Br5292", "151509": "Br5292", "151510": "Br5292",
    "151669": "Br5595", "151670": "Br5595", "151671": "Br5595", "151672": "Br5595",
    "151673": "Br8100", "151674": "Br8100", "151675": "Br8100", "151676": "Br8100",
}
LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]


@dataclass
class Section:
    name: str
    donor: str
    counts: sp.csr_matrix      # spots x genes, raw UMI counts
    coords: np.ndarray         # spots x 2, tissue coordinates (full-res pixels)
    labels: np.ndarray         # spots, layer label strings
    barcodes: np.ndarray
    gene_ids: np.ndarray

    @property
    def n_spots(self) -> int:
        return self.counts.shape[0]


def read_10x_h5(path: str | Path):
    """Read a Cell Ranger / Space Ranger v3 HDF5 matrix.

    Returns (counts as spots x genes CSR, barcodes, gene_ids, gene_names).
    The file stores genes x barcodes in CSC layout, which is barcodes x genes CSR.
    """
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        n_genes, n_cells = (int(v) for v in m["shape"][:])
        counts = sp.csr_matrix(
            (m["data"][:].astype(np.float32), m["indices"][:], m["indptr"][:]),
            shape=(n_cells, n_genes),
        )
        barcodes = m["barcodes"][:].astype(str)
        gene_ids = m["features/id"][:].astype(str)
        gene_names = m["features/name"][:].astype(str)
    return counts, barcodes, gene_ids, gene_names


def read_positions(path: str | Path) -> pd.DataFrame:
    """Space Ranger v1 ``tissue_positions_list.txt`` (no header)."""
    return pd.read_csv(
        path, header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"],
    ).set_index("barcode")


def read_layer_map(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=["barcode", "sample", "layer"],
                       dtype={"sample": str})


def assemble_section(name, donor, counts, barcodes, gene_ids, positions: pd.DataFrame,
                     layer_map: pd.DataFrame) -> Section:
    """Keep in-tissue spots that carry a manual layer label, in matrix order."""
    lab = layer_map[layer_map["sample"] == name].set_index("barcode")["layer"]
    pos = positions.reindex(barcodes)
    keep = (pos["in_tissue"].to_numpy() == 1) & np.isin(barcodes, lab.index.to_numpy())
    idx = np.flatnonzero(keep)
    return Section(
        name=name, donor=donor,
        counts=counts[idx].tocsr(),
        coords=pos.iloc[idx][["pxl_row", "pxl_col"]].to_numpy(dtype=np.float64),
        labels=lab.reindex(barcodes[idx]).to_numpy(dtype=str),
        barcodes=barcodes[idx], gene_ids=gene_ids,
    )


def load_section(data_dir: str | Path, name: str, layer_map: pd.DataFrame) -> Section:
    d = Path(data_dir)
    counts, barcodes, gene_ids, _ = read_10x_h5(d / f"{name}_filtered_feature_bc_matrix.h5")
    positions = read_positions(d / f"{name}_tissue_positions_list.txt")
    return assemble_section(name, SAMPLE_DONOR[name], counts, barcodes, gene_ids,
                            positions, layer_map)


def load_dlpfc(data_dir: str | Path, samples=None) -> list[Section]:
    d = Path(data_dir)
    layer_map = read_layer_map(d / "barcode_level_layer_map.tsv")
    samples = list(samples or SAMPLE_DONOR)
    sections = [load_section(d, s, layer_map) for s in samples]
    ref = sections[0].gene_ids
    for s in sections[1:]:
        if not np.array_equal(s.gene_ids, ref):
            raise ValueError(f"gene order differs in section {s.name}")
    return sections
