"""Redraw the figures (and per-layer F1 table) from a finished run's CSV outputs (no retraining)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stgnn.metrics import per_layer_f1
from stgnn.figures import boundary_plot, per_section_plot, spatial_maps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--map-section", default="151673")
    args = ap.parse_args(argv)
    m = json.loads((args.results / "metrics.json").read_text())
    tag = "SYNTHETIC - not a result" if m.get("data") == "SYNTHETIC" else ""
    runs = pd.read_csv(args.results / "per_section_runs.csv", dtype={"section": str})
    preds = pd.read_csv(args.results / "predictions_seed0.csv.gz", dtype={"section": str})
    variants = [c for c in preds.columns if c not in {"section", "barcode", "x", "y", "truth", "boundary"}]
    per_layer_f1(preds, m["classes"], variants).to_csv(args.results / "per_layer_f1_seed0.csv", index=False)
    per_section_plot(runs, args.results, tag)
    boundary_plot(runs, args.results, tag)
    sec = args.map_section if args.map_section in set(preds.section) else preds.section.iloc[-1]
    spatial_maps(preds, m["classes"], sec, args.results, tag)


if __name__ == "__main__":
    main()
