"""Leave-one-donor-out layer annotation: MLP vs GNN on the real vs rewired spatial graph.

Real data:   python run_experiment.py --data-dir data --out results
Smoke test:  python run_experiment.py --synthetic --out /tmp/synthetic_smoke
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from stgnn.data import load_dlpfc
from stgnn.experiment import VARIANTS, Config, run_cv, summarise
from stgnn.metrics import per_layer_f1
from stgnn.figures import boundary_plot, per_section_plot, spatial_maps
from stgnn.synthetic import make_synthetic


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--synthetic", action="store_true", help="run on synthetic data (smoke test, not a result)")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--map-section", default="151673", help="section drawn in the spatial map figure")
    args = ap.parse_args(argv)

    cfg = Config(k=args.k, seeds=tuple(range(args.seeds)), max_epochs=args.max_epochs,
                 patience=args.patience, variants=tuple(args.variants))
    if args.synthetic:
        sections, tag = make_synthetic(), "SYNTHETIC - not a result"
        cfg.n_hvg, cfg.n_pcs = 200, 20
    else:
        sections, tag = load_dlpfc(args.data_dir), ""
    print(f"{len(sections)} sections, {sum(s.n_spots for s in sections)} labelled spots", flush=True)

    t0 = time.time()
    results, gstats, preds, classes = run_cv(sections, cfg)
    summary = summarise(results, gstats)
    summary.update({"data": "SYNTHETIC" if args.synthetic else "LIBD DLPFC Visium (Maynard et al., 2021)",
                    "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                    "runtime_s": round(time.time() - t0, 1), "classes": classes})

    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "per_section_runs.csv", index=False)
    gstats.to_csv(args.out / "graph_stats.csv", index=False)
    preds.to_csv(args.out / "predictions_seed0.csv.gz", index=False)
    per_layer_f1(preds, classes, list(cfg.variants)).to_csv(args.out / "per_layer_f1_seed0.csv", index=False)
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    per_section_plot(results, args.out, tag)
    boundary_plot(results, args.out, tag)
    map_sec = args.map_section if args.map_section in set(preds.get("section", [])) else sections[-1].name
    spatial_maps(preds, classes, map_sec, args.out, tag)

    print("\nmean per-section macro-F1 (95% bootstrap CI over sections)")
    for v, r in summary["variants"].items():
        lo, hi = r["macro_f1_ci95"]
        print(f"  {v:13s} {r['macro_f1_mean']:.3f} [{lo:.3f}, {hi:.3f}]")
    print("\npaired comparisons (Wilcoxon signed-rank over sections)")
    for c in summary["comparisons"]:
        print(f"  {c['a']:13s} - {c['b']:13s} {c['mean_diff']:+.3f}  p={c['wilcoxon_p']:.4f}  "
              f"({c['n_a_better']}/{c['n']} sections better)")
    g = summary["graph"]
    print(f"\nedge homophily: real {g['homophily_real_mean']:.3f}, rewired {g['homophily_rewired_mean']:.3f}, "
          f"chance {g['homophily_chance_mean']:.3f}")
    print(f"wrote {args.out}/ in {summary['runtime_s']} s")
    return summary


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
