"""Plots (PNG) and CSV output."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (fixed order, never cycled) and chart ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MAX_COLORED = 7  # more lines than this: the rest are drawn thin and gray

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
    "font.family": "sans-serif", "lines.solid_capstyle": "round",
})


def _pct(ax, axis="y"):
    from matplotlib.ticker import PercentFormatter
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1.0, decimals=0))


def save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equity(results, path: Path, title: str, ranking: list[str] | None = None) -> Path:
    """Equity curves (growth of 1, log scale) above their drawdowns. results: BacktestResult list."""
    order = ranking or [r.label for r in results]
    by_label = {r.label: r for r in results}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    first = results[0].equity
    bench = first["benchmark"] / first["benchmark"].iloc[0]
    ax.plot(first["date"], bench, color=MUTED, lw=1.4, ls="--", label="Benchmark", zorder=1)
    for k, label in enumerate(order):
        r = by_label[label]
        e = r.equity["equity"] / r.equity["equity"].iloc[0]
        dd = r.equity["equity"] / r.equity["equity"].cummax() - 1
        colored = k < MAX_COLORED
        style = dict(color=SERIES[k] if colored else AXIS, lw=1.8 if colored else 0.8, zorder=3 if colored else 2)
        ax.plot(r.equity["date"], e, label=label if colored else None, **style)
        ax2.plot(r.equity["date"], dd, **style)
    if len(order) > MAX_COLORED:
        ax.plot([], [], color=AXIS, lw=0.8, label=f"{len(order) - MAX_COLORED} more variants")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (log)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", ncol=1 if len(order) < 4 else 2)
    ax2.set_ylabel("Drawdown")
    _pct(ax2)
    fig.align_ylabels()
    return save(fig, path)


def plot_event_curves(curves: pd.DataFrame, summary: pd.DataFrame, path: Path, title: str,
                      dimension: str, min_events: int) -> Path | None:
    c = curves[curves["dimension"] == dimension]
    buckets = list(dict.fromkeys(c["bucket"]))
    if not buckets:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color=AXIS, lw=1)
    for k, b in enumerate(buckets[:len(SERIES)]):
        cb = c[(c["bucket"] == b) & (c["n"] > 0)]
        n0 = int(cb["n"].iloc[0]) if len(cb) else 0
        color = SERIES[k]
        thin = n0 < min_events
        ax.plot(cb["day"], cb["mean"], color=color, lw=1.0 if thin else 1.8, ls=":" if thin else "-",
                label=f"{b} (n={n0}{', few' if thin else ''})")
        ax.fill_between(cb["day"], cb["mean"] - 2 * cb["sem"], cb["mean"] + 2 * cb["sem"], color=color, alpha=0.12, lw=0)
        if dimension == "all":
            ax.plot(cb["day"], cb["p25"], color=color, lw=0.8, ls="--", label="25th / 75th percentile")
            ax.plot(cb["day"], cb["p75"], color=color, lw=0.8, ls="--")
        s = summary[(summary["dimension"] == dimension) & (summary["bucket"] == b)]
        if len(s):
            pk = s.iloc[0]
            ax.plot([pk["peak_day"]], [pk["mean_at_peak"]], "o", ms=5, color=color, mec=SURFACE, mew=1.5)
    for d in (63, 126, 252, 504):
        if d <= c["day"].max():
            ax.axvline(d, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("Trading days after entry")
    ax.set_ylabel("Mean abnormal return vs benchmark")
    _pct(ax)
    ax.set_title(title, loc="left", pad=20)
    ax.text(0, 1.015, "Line: mean, band: ±2 standard errors, dot: peak of the smoothed mean",
            transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    ax.legend(loc="upper left")
    return save(fig, path)


def plot_event_grid(curves: pd.DataFrame, path: Path, title: str, dim_a: list[str], dim_b: list[str],
                    combo: str, min_events: int) -> Path | None:
    c = curves[curves["dimension"] == combo]
    if c.empty:
        return None
    fig, axes = plt.subplots(len(dim_a), len(dim_b), figsize=(3.2 * len(dim_b), 2.4 * len(dim_a)),
                             sharex=True, sharey=True, squeeze=False)
    for i, a in enumerate(dim_a):
        for j, b in enumerate(dim_b):
            ax = axes[i][j]
            ax.axhline(0, color=AXIS, lw=0.8)
            cb = c[(c["bucket"] == f"{a} | {b}") & (c["n"] > 0)]
            n0 = int(cb["n"].iloc[0]) if len(cb) else 0
            if len(cb):
                ax.fill_between(cb["day"], cb["p25"], cb["p75"], color=SERIES[0], alpha=0.10, lw=0)
                ax.fill_between(cb["day"], cb["mean"] - 2 * cb["sem"], cb["mean"] + 2 * cb["sem"], color=SERIES[0], alpha=0.25, lw=0)
                ax.plot(cb["day"], cb["mean"], color=SERIES[0], lw=1.5 if n0 >= min_events else 0.8)
            ax.set_title(f"{a} · {b}  (n={n0})", fontsize=8, loc="left", color=INK if n0 >= min_events else MUTED)
            _pct(ax)
    fig.suptitle(title + "\nmean (line), ±2 SE (dark band), 25th-75th percentile (light band)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.supxlabel("Trading days after entry", fontsize=9, color=INK2)
    fig.tight_layout()
    return save(fig, path)


def plot_walkforward(table: pd.DataFrame, all_test: pd.DataFrame, metric: str, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(table))
    for x, w in zip(xs, table["window"]):
        vals = all_test[all_test["window"] == w][metric].astype(float).values
        ax.scatter(np.full(len(vals), x), vals, s=22, color=AXIS, zorder=2, label="all variants" if x == 0 else None)
    ax.scatter(xs, table[f"test_{metric}"].astype(float), s=60, color=SERIES[0], zorder=3,
               edgecolor=SURFACE, linewidth=1.5, label="variant chosen on train window")
    ax.set_xticks(xs, [f"{r.test_start:%Y}-{r.test_end:%Y}" for r in table.itertuples()])
    ax.set_xlabel("Test window")
    ax.set_ylabel(f"Test {metric}")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_title(title, loc="left")
    ax.legend(loc="best")
    return save(fig, path)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path
