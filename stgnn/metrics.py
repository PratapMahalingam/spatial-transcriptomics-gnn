"""Section-level metrics and paired statistics."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score


def section_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the layers present in this section's ground truth.

    Some sections (donor Br5595) contain no L1/L2 spots; averaging over absent
    classes would give them an F1 of 0 for reasons unrelated to the model.
    Predictions of an absent layer still count as errors for the true layer.
    """
    labels = np.unique(y_true)
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired comparison of per-section scores a vs b (same sections, same order)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    res = {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()),
           "median_diff": float(np.median(d)), "n": int(len(d)),
           "n_a_better": int((d > 0).sum())}
    if np.allclose(d, 0):
        res["wilcoxon_p"] = 1.0
    else:
        res["wilcoxon_p"] = float(wilcoxon(a, b, alternative="two-sided").pvalue)
    return res


def bootstrap_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int = 10000,
                 level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, float)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def per_layer_f1(preds, classes: list, variants: list):
    """Pooled per-layer F1 from a predictions table (columns: truth + one per variant)."""
    import pandas as pd

    rows = []
    for v in variants:
        if v in preds:
            f = f1_score(preds["truth"], preds[v], labels=classes, average=None, zero_division=0)
            rows.append({"variant": v, **{c: float(x) for c, x in zip(classes, f)}})
    return pd.DataFrame(rows)
