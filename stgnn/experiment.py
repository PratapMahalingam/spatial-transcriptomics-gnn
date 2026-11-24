"""Leave-one-donor-out comparison of expression-only, real-graph and rewired-graph models."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .data import Section
from .features import Preprocessor, log_normalize
from .graph import boundary_mask, edge_homophily, knn_edges, rewire_degree_preserving
from .metrics import bootstrap_ci, paired_test, section_macro_f1
from .models import build_model
from .train import fit, predict, stack_sections

# variant name -> (architecture, graph used for training AND testing)
VARIANTS = {
    "mlp": ("mlp", "none"),
    "sage_nograph": ("sage", "none"),
    "sage_real": ("sage", "real"),
    "sage_rewired": ("sage", "rewired"),
    "gcn_real": ("gcn", "real"),
    "gcn_rewired": ("gcn", "rewired"),
}

# (a, b, question)
COMPARISONS = [
    ("sage_real", "mlp", "does the real spatial graph beat expression alone?"),
    ("sage_real", "sage_rewired", "is the gain from real spatial structure rather than extra edges?"),
    ("sage_rewired", "mlp", "do random edges with the same degrees help on their own?"),
    ("sage_nograph", "mlp", "is the architecture alone better (no neighbours)?"),
    ("sage_real", "sage_nograph", "neighbours vs same architecture without neighbours"),
    ("gcn_real", "mlp", "GCN: real graph vs expression alone"),
    ("gcn_real", "gcn_rewired", "GCN: real vs rewired graph"),
]


@dataclass
class Config:
    k: int = 6
    n_hvg: int = 2000
    n_pcs: int = 50
    hidden: int = 128
    dropout: float = 0.3
    lr: float = 5e-3
    weight_decay: float = 5e-4
    max_epochs: int = 300
    patience: int = 30
    swaps_per_edge: int = 10
    seeds: tuple = (0, 1, 2, 3, 4)
    variants: tuple = tuple(VARIANTS)
    keep_predictions_seed: int = 0
    log: bool = True


@dataclass
class Prepared:
    sections: list
    classes: list
    y: dict = field(default_factory=dict)
    lognorm: dict = field(default_factory=dict)
    real_edges: dict = field(default_factory=dict)
    boundary: dict = field(default_factory=dict)


def prepare(sections: list[Section], cfg: Config) -> Prepared:
    classes = sorted({c for s in sections for c in s.labels})
    cidx = {c: i for i, c in enumerate(classes)}
    p = Prepared(sections=sections, classes=classes)
    for s in sections:
        p.y[s.name] = np.array([cidx[c] for c in s.labels], dtype=np.int64)
        p.lognorm[s.name] = log_normalize(s.counts)
        p.real_edges[s.name] = knn_edges(s.coords, cfg.k)
        p.boundary[s.name] = boundary_mask(p.real_edges[s.name], s.labels)
    return p


def _rewired(p: Prepared, cfg: Config, seed: int, cache: dict) -> dict:
    if seed not in cache:
        out, info = {}, {}
        for j, s in enumerate(p.sections):
            rng = np.random.default_rng([seed, j, 7919])
            e, n_sw = rewire_degree_preserving(p.real_edges[s.name], s.n_spots, rng, cfg.swaps_per_edge)
            out[s.name] = e
            info[s.name] = n_sw
        cache[seed] = (out, info)
    return cache[seed]


def _choose_val(train_sections: list[Section], fold_seed: int) -> set:
    """One validation section per training donor, chosen at random but fixed per fold."""
    rng = np.random.default_rng(fold_seed)
    by_donor: dict[str, list[str]] = {}
    for s in train_sections:
        by_donor.setdefault(s.donor, []).append(s.name)
    val = set()
    for d in sorted(by_donor):
        names = sorted(by_donor[d])
        if len(names) > 1:
            val.add(names[rng.integers(len(names))])
    if not val:  # single-section donors: fall back to one section overall
        val.add(sorted(s.name for s in train_sections)[0])
    return val


def run_cv(sections: list[Section], cfg: Config):
    """Returns (per-section results DataFrame, graph stats DataFrame, seed-0 predictions)."""
    torch.set_num_threads(max(1, torch.get_num_threads()))
    p = prepare(sections, cfg)
    n_classes = len(p.classes)
    rewire_cache: dict = {}
    rows, preds = [], {}
    donors = sorted({s.donor for s in sections})
    for fi, test_donor in enumerate(donors):
        test = [s for s in sections if s.donor == test_donor]
        rest = [s for s in sections if s.donor != test_donor]
        val_names = _choose_val(rest, fold_seed=1000 + fi)
        train = [s for s in rest if s.name not in val_names]
        val = [s for s in rest if s.name in val_names]
        pre = Preprocessor(cfg.n_hvg, cfg.n_pcs, seed=0).fit([p.lognorm[s.name] for s in train])
        feats = {s.name: pre.transform(p.lognorm[s.name]) for s in sections}
        for seed in cfg.seeds:
            rewired, _ = _rewired(p, cfg, seed, rewire_cache)
            graphs = {"none": {s.name: np.zeros((0, 2), np.int64) for s in sections},
                      "real": p.real_edges, "rewired": rewired}
            for vname in cfg.variants:
                arch, gkind = VARIANTS[vname]
                g = graphs[gkind]
                mk = lambda ss: stack_sections([feats[s.name] for s in ss], [g[s.name] for s in ss],
                                               [p.y[s.name] for s in ss])
                tr, va, te = mk(train), mk(val), mk(test)
                torch.manual_seed(seed)
                model = build_model(arch, tr.x.shape[1], n_classes, hidden=cfg.hidden, dropout=cfg.dropout)
                t0 = time.time()
                info = fit(model, tr, va, n_classes, lr=cfg.lr, weight_decay=cfg.weight_decay,
                           max_epochs=cfg.max_epochs, patience=cfg.patience)
                yhat = predict(model, te)
                for i, s in enumerate(test):
                    a, b = te.section_ptr[i], te.section_ptr[i + 1]
                    yt, yp = p.y[s.name], yhat[a:b]
                    bnd = p.boundary[s.name]
                    correct = yt == yp
                    rows.append({
                        "test_donor": test_donor, "seed": seed, "variant": vname, "section": s.name,
                        "macro_f1": section_macro_f1(yt, yp), "accuracy": float(correct.mean()),
                        "acc_boundary": float(correct[bnd].mean()) if bnd.any() else np.nan,
                        "acc_interior": float(correct[~bnd].mean()) if (~bnd).any() else np.nan,
                        "n_spots": int(len(yt)), "n_boundary": int(bnd.sum()),
                        "best_epoch": info["best_epoch"], "best_val_f1": info["best_val_f1"],
                        "val_sections": ",".join(sorted(val_names)),
                    })
                    if seed == cfg.keep_predictions_seed:
                        preds.setdefault(s.name, {})[vname] = yp
                if cfg.log:
                    f1s = [r["macro_f1"] for r in rows[-len(test):]]
                    print(f"[fold {test_donor} seed {seed}] {vname:13s} test macro-F1 {np.mean(f1s):.3f} "
                          f"(val {info['best_val_f1']:.3f}, epoch {info['best_epoch']}, {time.time() - t0:.1f}s)",
                          flush=True)
    gstats = []
    for s in sections:
        e_real = p.real_edges[s.name]
        row = {"section": s.name, "donor": s.donor, "n_spots": s.n_spots, "n_edges": len(e_real),
               "mean_degree": 2 * len(e_real) / s.n_spots,
               "homophily_real": edge_homophily(e_real, s.labels),
               "boundary_fraction": float(p.boundary[s.name].mean())}
        hs, sw = [], []
        for seed in cfg.seeds:
            if seed in rewire_cache:
                hs.append(edge_homophily(rewire_cache[seed][0][s.name], s.labels))
                sw.append(rewire_cache[seed][1][s.name])
        row["homophily_rewired"] = float(np.mean(hs)) if hs else np.nan
        row["swaps_accepted_mean"] = float(np.mean(sw)) if sw else np.nan
        row["layer_fraction_sq_sum"] = float(np.sum((pd.Series(s.labels).value_counts(normalize=True)) ** 2))
        gstats.append(row)
    pred_df = []
    for s in sections:
        if s.name in preds:
            d = pd.DataFrame({"section": s.name, "barcode": s.barcodes,
                              "x": s.coords[:, 1], "y": s.coords[:, 0],
                              "truth": s.labels, "boundary": p.boundary[s.name]})
            for v, yp in preds[s.name].items():
                d[v] = np.asarray(p.classes)[yp]
            pred_df.append(d)
    return pd.DataFrame(rows), pd.DataFrame(gstats), (pd.concat(pred_df) if pred_df else pd.DataFrame()), p.classes


def summarise(results: pd.DataFrame, gstats: pd.DataFrame, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    per_sec = results.groupby(["variant", "section"], as_index=False)[
        ["macro_f1", "accuracy", "acc_boundary", "acc_interior"]].mean()
    donor_of = results.drop_duplicates("section").set_index("section")["test_donor"]
    wide = per_sec.pivot(index="section", columns="variant", values="macro_f1").sort_index()
    out = {"n_sections": int(len(wide)), "n_seeds": int(results["seed"].nunique()),
           "variants": {}, "comparisons": [], "per_donor_macro_f1": {}, "graph": {}}
    for v in wide.columns:
        x = wide[v].to_numpy()
        seed_means = results[results.variant == v].groupby("seed")["macro_f1"].mean()
        sub = per_sec[per_sec.variant == v]
        out["variants"][v] = {
            "macro_f1_mean": float(x.mean()), "macro_f1_ci95": bootstrap_ci(x, rng),
            "macro_f1_seed_sd": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
            "accuracy_mean": float(sub["accuracy"].mean()),
            "acc_boundary_mean": float(sub["acc_boundary"].mean()),
            "acc_interior_mean": float(sub["acc_interior"].mean()),
        }
    for a, b, q in COMPARISONS:
        if a in wide and b in wide:
            r = paired_test(wide[a].to_numpy(), wide[b].to_numpy())
            d = (wide[a] - wide[b]).groupby(donor_of.reindex(wide.index)).mean()
            r.update({"a": a, "b": b, "question": q, "diff_ci95": bootstrap_ci((wide[a] - wide[b]).to_numpy(), rng),
                      "mean_diff_by_donor": {k: float(v) for k, v in d.items()}})
            out["comparisons"].append(r)
    if {"mlp", "sage_real", "sage_rewired"} <= set(wide.columns):
        g_real = (wide["sage_real"] - wide["mlp"]).mean()
        g_rew = (wide["sage_rewired"] - wide["mlp"]).mean()
        out["share_of_sage_gain_reproduced_by_rewired_graph"] = float(g_rew / g_real) if g_real != 0 else None
    for dn, sub in per_sec.assign(donor=per_sec.section.map(donor_of)).groupby("donor"):
        out["per_donor_macro_f1"][dn] = {v: float(s["macro_f1"].mean()) for v, s in sub.groupby("variant")}
    out["graph"] = {
        "mean_degree": float(gstats["mean_degree"].mean()),
        "homophily_real_mean": float(gstats["homophily_real"].mean()),
        "homophily_rewired_mean": float(gstats["homophily_rewired"].mean()),
        "homophily_chance_mean": float(gstats["layer_fraction_sq_sum"].mean()),
        "boundary_fraction_mean": float(gstats["boundary_fraction"].mean()),
    }
    out["per_section_macro_f1"] = {s: {v: float(wide.loc[s, v]) for v in wide.columns} for s in wide.index}
    return out
