"""Figures for the CV results."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ORDER = ["mlp", "sage_nograph", "sage_rewired", "sage_real", "gcn_rewired", "gcn_real"]
NICE = {"mlp": "MLP\n(expression)", "sage_nograph": "SAGE\nno edges", "sage_rewired": "SAGE\nrewired graph",
        "sage_real": "SAGE\nreal graph", "gcn_rewired": "GCN\nrewired graph", "gcn_real": "GCN\nreal graph"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _stamp(fig, tag):
    if tag:
        fig.text(0.995, 0.005, tag, ha="right", va="bottom", fontsize=9, color="#d03b3b", weight="bold")


def per_section_plot(results: pd.DataFrame, out: Path, tag: str = ""):
    per = results.groupby(["variant", "section", "test_donor"], as_index=False)["macro_f1"].mean()
    order = [v for v in ORDER if v in set(per.variant)]
    wide = per.pivot(index=["section", "test_donor"], columns="variant", values="macro_f1")[order]
    donors = sorted(wide.index.get_level_values(1).unique())
    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor=SURFACE)
    _style(ax)
    xs = np.arange(len(order))
    for (sec, dn), row in wide.iterrows():
        c = SERIES[donors.index(dn) % len(SERIES)]
        ax.plot(xs, row.to_numpy(), "-", color=c, lw=1, alpha=0.55)
        ax.plot(xs, row.to_numpy(), "o", ms=6, color=c, mec=SURFACE, mew=1.5)
    means = wide.mean(0).to_numpy()
    ax.plot(xs, means, "_", ms=28, mew=2.5, color=INK, zorder=5)
    for x, m in zip(xs, means):
        ax.text(x + 0.18, m, f"{m:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xticks(xs, [NICE[v] for v in order], fontsize=9)
    ax.set_ylabel("macro-F1 on held-out section", color=INK2)
    ax.set_title("Layer annotation by model (each line = one held-out section; bar = mean)",
                 loc="left", fontsize=11, color=INK)
    handles = [plt.Line2D([], [], marker="o", ls="-", color=SERIES[i % len(SERIES)], label=f"donor {d} (held out)")
               for i, d in enumerate(donors)]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _stamp(fig, tag)
    fig.tight_layout()
    fig.savefig(out / "per_section_macro_f1.png", dpi=150)
    plt.close(fig)


def boundary_plot(results: pd.DataFrame, out: Path, tag: str = ""):
    per = results.groupby("variant")[["acc_interior", "acc_boundary"]].mean()
    order = [v for v in ORDER if v in per.index]
    per = per.loc[order]
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    _style(ax)
    xs, w = np.arange(len(order)), 0.36
    for k, (col, lab) in enumerate([("acc_interior", "interior spots (all neighbours same layer)"),
                                     ("acc_boundary", "boundary spots (>= 1 neighbour in another layer)")]):
        vals = per[col].to_numpy()
        ax.bar(xs + (k - 0.5) * (w + 0.02), vals, w, color=SERIES[k], label=lab)
        for x, v in zip(xs + (k - 0.5) * (w + 0.02), vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(xs, [NICE[v] for v in order], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy", color=INK2)
    ax.set_title("Where the graph helps: accuracy at layer interiors vs layer boundaries",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    _stamp(fig, tag)
    fig.tight_layout()
    fig.savefig(out / "boundary_vs_interior.png", dpi=150)
    plt.close(fig)


def spatial_maps(preds: pd.DataFrame, classes: list, section: str, out: Path, tag: str = ""):
    d = preds[preds.section == section]
    if d.empty:
        return
    panels = [c for c in ["truth", "mlp", "sage_rewired", "sage_real"] if c in d.columns]
    titles = {"truth": "manual annotation", "mlp": "MLP (expression only)",
              "sage_rewired": "SAGE, rewired graph", "sage_real": "SAGE, real graph"}
    cmap = {c: SERIES[i % len(SERIES)] for i, c in enumerate(classes)}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.6), facecolor=SURFACE)
    for ax, p in zip(np.atleast_1d(axes), panels):
        ax.set_facecolor(SURFACE)
        ax.scatter(d["x"], -d["y"], c=[cmap[v] for v in d[p]], s=3.5, linewidths=0)
        if p != "truth":
            acc = float((d[p] == d["truth"]).mean())
            ax.set_title(f"{titles[p]}\naccuracy {acc:.2f}", fontsize=10, color=INK)
        else:
            ax.set_title(f"{titles[p]}\nsection {section}", fontsize=10, color=INK)
        ax.set_aspect("equal")
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap[c], label=c) for c in classes]
    fig.legend(handles=handles, loc="lower center", ncol=len(classes), frameon=False, fontsize=9)
    _stamp(fig, tag)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(out / f"spatial_map_{section}.png", dpi=150)
    plt.close(fig)
