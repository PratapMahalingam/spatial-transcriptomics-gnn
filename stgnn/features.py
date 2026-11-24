"""Expression preprocessing. Everything learned from data is fitted on training sections only."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA


def log_normalize(counts: sp.csr_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    """Per-spot library-size normalisation then log1p. Uses only the spot itself."""
    counts = sp.csr_matrix(counts, dtype=np.float32)
    totals = np.asarray(counts.sum(1)).ravel()
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    out = sp.diags(scale.astype(np.float32)) @ counts
    out = out.tocsr()
    out.data = np.log1p(out.data)
    return out


class Preprocessor:
    """Highly variable genes -> z-score -> PCA, all fitted on training data."""

    def __init__(self, n_hvg: int = 2000, n_pcs: int = 50, min_frac: float = 0.01,
                 clip: float = 10.0, seed: int = 0):
        self.n_hvg, self.n_pcs, self.min_frac, self.clip, self.seed = n_hvg, n_pcs, min_frac, clip, seed

    def fit(self, mats: list[sp.csr_matrix]) -> "Preprocessor":
        X = sp.vstack(mats).tocsr()
        frac = np.asarray((X > 0).mean(0)).ravel()
        mean = np.asarray(X.mean(0)).ravel()
        var = np.asarray(X.multiply(X).mean(0)).ravel() - mean ** 2
        var = np.where(frac >= self.min_frac, var, -np.inf)
        n_hvg = int(min(self.n_hvg, np.isfinite(var).sum()))
        self.hvg_ = np.sort(np.argsort(var)[::-1][:n_hvg])
        Xh = X[:, self.hvg_].toarray()
        self.mean_ = Xh.mean(0)
        self.std_ = np.maximum(Xh.std(0), 1e-6)
        Z = self._scale(Xh)
        n_pcs = int(min(self.n_pcs, Z.shape[0] - 1, Z.shape[1]))
        self.pca_ = PCA(n_components=n_pcs, svd_solver="randomized", random_state=self.seed).fit(Z)
        return self

    def _scale(self, Xh: np.ndarray) -> np.ndarray:
        return np.clip((Xh - self.mean_) / self.std_, -self.clip, self.clip)

    def transform(self, mat: sp.csr_matrix) -> np.ndarray:
        Z = self._scale(mat[:, self.hvg_].toarray())
        return self.pca_.transform(Z).astype(np.float32)
