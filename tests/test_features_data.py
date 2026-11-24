import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from stgnn.data import assemble_section, read_10x_h5
from stgnn.features import Preprocessor, log_normalize


def test_log_normalize_matches_definition():
    c = sp.csr_matrix(np.array([[1, 3, 0], [0, 0, 0], [5, 5, 10]], dtype=float))
    out = log_normalize(c, target_sum=100).toarray()
    expected = np.log1p(np.array([[25, 75, 0], [0, 0, 0], [25, 25, 50]], dtype=float))
    assert np.allclose(out, expected, atol=1e-5)


def test_preprocessor_is_fitted_on_training_data_only():
    rng = np.random.default_rng(0)
    train = [sp.csr_matrix(rng.poisson(1.0, (60, 40)).astype(float)) for _ in range(2)]
    test_a = sp.csr_matrix(rng.poisson(1.0, (30, 40)).astype(float))
    test_b = sp.csr_matrix(rng.poisson(5.0, (30, 40)).astype(float))
    p1 = Preprocessor(n_hvg=20, n_pcs=5).fit(train)
    p2 = Preprocessor(n_hvg=20, n_pcs=5).fit(train)
    p1.transform(test_a)
    p2.transform(test_b)
    assert np.array_equal(p1.hvg_, p2.hvg_)
    assert np.allclose(p1.mean_, p2.mean_) and np.allclose(p1.pca_.components_, p2.pca_.components_)
    assert p1.transform(test_a).shape == (30, 5)


def _write_10x(path, dense, barcodes, genes):
    m = sp.csc_matrix(dense.T)  # genes x cells, CSC over cells
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = m.data, m.indices, m.indptr
        g["shape"] = np.array(m.shape)
        g["barcodes"] = np.array(barcodes, dtype="S")
        g["features/id"] = np.array(genes, dtype="S")
        g["features/name"] = np.array(genes, dtype="S")


def test_read_10x_and_label_join(tmp_path):
    dense = np.arange(12, dtype=float).reshape(4, 3)   # 4 spots x 3 genes
    bcs, genes = ["A-1", "B-1", "C-1", "D-1"], ["g1", "g2", "g3"]
    _write_10x(tmp_path / "m.h5", dense, bcs, genes)
    counts, barcodes, gene_ids, _ = read_10x_h5(tmp_path / "m.h5")
    assert np.array_equal(counts.toarray(), dense)
    assert list(barcodes) == bcs and list(gene_ids) == genes

    pos = pd.DataFrame({"in_tissue": [1, 1, 0, 1], "array_row": 0, "array_col": 0,
                        "pxl_row": [10, 20, 30, 40], "pxl_col": [1, 2, 3, 4]}, index=bcs)
    # D-1 is in tissue but unlabelled; C-1 is labelled but outside tissue
    lab = pd.DataFrame({"barcode": ["A-1", "B-1", "C-1", "X-1"], "sample": ["s1"] * 3 + ["s2"],
                        "layer": ["L1", "WM", "L3", "L2"]})
    s = assemble_section("s1", "d", counts, barcodes, gene_ids, pos, lab)
    assert list(s.barcodes) == ["A-1", "B-1"]
    assert list(s.labels) == ["L1", "WM"]
    assert np.array_equal(s.counts.toarray(), dense[:2])
    assert np.array_equal(s.coords, [[10, 1], [20, 2]])
