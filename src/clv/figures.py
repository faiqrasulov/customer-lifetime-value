"""Report figures (static PNGs for the README)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter, StrMethodFormatter  # noqa: E402

from clv import bgnbd  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "lines.linewidth": 2,
    "lines.markersize": 6.5,
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_frequency_buckets(buckets: pd.DataFrame, holdout_weeks: float, path: Path) -> None:
    """Fader plot 1: mean holdout purchases by calibration frequency, actual vs predicted."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    pos = np.arange(len(buckets))
    ax.plot(pos, buckets["actual"], "o-", color=INK, label="Actual")
    ax.plot(pos, buckets["bgnbd"], "s-", color=BLUE, label="BG/NBD prediction")
    ax.plot(pos, buckets["historical_rate"], "^--", color=ORANGE, lw=1.6,
            label="Baseline: past rate x/T extrapolated")
    ax.set_xticks(pos, [f"{k}\nn={n:,}" for k, n in zip(buckets.index, buckets["n"], strict=True)])
    ax.set_xlabel("Repeat purchase days in calibration (Dec 2010 - Sep 9, 2011)")
    ax.set_ylabel(f"Mean purchase days in {holdout_weeks:.0f}-week holdout")
    ax.set_title("Holdout purchases by calibration frequency (UK customers)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    _save(fig, path)


def fig_tracking(tracking: pd.DataFrame, calibration_end: str, path: Path) -> None:
    """Fader plot 2: weekly and cumulative repeat purchases, actual vs BG/NBD."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    t = tracking["week_start"]
    cut = pd.Timestamp(calibration_end)
    ax1.plot(t, tracking["actual"], color=INK, lw=1.6, label="Actual")
    ax1.plot(t, tracking["expected"], color=BLUE, label="BG/NBD expected")
    ax1.set_title("Weekly repeat purchase days")
    ax1.set_ylabel("Repeat purchase days per week")
    ax2.plot(t, tracking["cum_actual"], color=INK, lw=1.6, label="Actual")
    ax2.plot(t, tracking["cum_expected"], color=BLUE, label="BG/NBD expected")
    ax2.set_title("Cumulative repeat purchase days")
    ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for ax in (ax1, ax2):
        ax.axvline(cut, color=INK2, ls="--", lw=1.1)
        ax.text(cut, ax.get_ylim()[1] * 0.97, "  holdout \u2192", color=INK2, fontsize=9, va="top")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.legend(loc="upper left")
    _save(fig, path)


def fig_value_concentration(gain: dict, lorenz: np.ndarray, top_share: dict, path: Path) -> None:
    """Pareto curve of predicted CLV, and holdout revenue captured by each ranking."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    share = np.linspace(0.0, 1.0, lorenz.size)
    ax1.plot(share, lorenz, color=BLUE)
    ax1.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1.2)
    for q in (0.1, 0.2):
        y = lorenz[int(q * (lorenz.size - 1))]
        ax1.plot([q], [y], "o", color=BLUE, mec=SURFACE, mew=2, ms=8)
        ax1.annotate(f"top {q:.0%} of customers:\n{y:.0%} of predicted value", (q, y),
                     xytext=(12, -22), textcoords="offset points", fontsize=9, color=INK)
    ax1.set_title("Pareto curve of predicted 12-month CLV")
    ax1.set_xlabel("Share of customers (ranked by predicted CLV)")
    ax1.set_ylabel("Share of total predicted CLV")

    s = gain["share"]
    ax2.plot(s, gain["oracle"], color=MUTED, ls="--", lw=1.4, label="Perfect foresight")
    ax2.plot(s, gain["bgnbd_gg_clv"], color=BLUE,
             label=f"BG/NBD + Gamma-Gamma CLV (top 10%: {top_share['bgnbd_gg_clv']:.1%})")
    ax2.plot(s, gain["past_spend"], color=AQUA,
             label=f"Past spend (top 10%: {top_share['past_spend']:.1%})")
    ax2.plot(s, gain["rfm_score"], color=ORANGE,
             label=f"RFM score (top 10%: {top_share['rfm_score']:.1%})")
    ax2.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1.2, label="Random")
    ax2.axvline(0.1, color=GRID, lw=1.2)
    ax2.set_title("Holdout revenue captured by each ranking")
    ax2.set_xlabel("Share of customers targeted (ranked at Sep 9, 2011)")
    ax2.set_ylabel("Share of actual holdout revenue")
    ax2.legend(loc="lower right")
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
    _save(fig, path)


def fig_p_alive(params: bgnbd.BGNBDParams, scores: pd.DataFrame, path: Path) -> None:
    """P(alive) over frequency x recency for a customer first seen at the start of the data."""
    T = float(scores["T"].max())
    max_x = int(min(np.quantile(scores["frequency"], 0.995), 60))
    xs = np.arange(0, max_x + 1)
    recency = np.linspace(0.0, T, 120)
    xx, rr = np.meshgrid(xs, recency, indexing="ij")
    prob = bgnbd.p_alive(params, xx, np.where(xx > 0, rr, 0.0), T)
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    mesh = ax.pcolormesh(recency, xs, prob, cmap=cmap, vmin=0, vmax=1, shading="nearest",
                         rasterized=True)
    cs = ax.contour(recency, xs, prob, levels=[0.2, 0.5, 0.8], colors=[SURFACE], linewidths=1.2)
    ax.clabel(cs, fmt=lambda v: f"{v:.0%}", fontsize=8.5, colors=SURFACE)
    old = scores[scores["T"] >= T - 1.0]
    ax.scatter(old["recency"], np.minimum(old["frequency"], max_x), s=7, color=ORANGE,
               alpha=0.55, lw=0, label=f"Customers first seen Dec 1-7, 2010 (n={len(old):,})")
    ax.grid(False)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, max_x + 0.5)
    ax.set_xlabel("Recency: weeks from first to last purchase day")
    ax.set_ylabel("Repeat purchase days (frequency)")
    ax.set_title(f"P(alive) at Dec 9, 2011 for a customer observed for T = {T:.0f} weeks")
    ax.legend(loc="upper left", markerscale=2.5)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("P(still active)", color=INK2)
    cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    cb.outline.set_visible(False)
    _save(fig, path)


def make_figures(main: dict, business: dict, out: Path) -> None:
    """Write the four report figures for the main split and the full-data model."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig_frequency_buckets(main["_buckets"], main["holdout_weeks"],
                              out / "fader_frequency_buckets.png")
        fig_tracking(main["_tracking"], main["split"]["calibration_end"], out / "tracking.png")
        fig_value_concentration(main["_gain"], business["_lorenz"],
                                main["top_decile_revenue_capture"],
                                out / "value_concentration.png")
        fig_p_alive(business["_bg"], business["_scores"], out / "p_alive_heatmap.png")
