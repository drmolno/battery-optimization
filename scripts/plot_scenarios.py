"""Figure: what the day-ahead price scenarios look like for one delivery day.

Usage:
    python scripts/plot_scenarios.py               # picks a volatile 15-min-era day
    python scripts/plot_scenarios.py --day 2026-02-10
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bessopt.config import RESULTS_DIR
from bessopt.data import entsoe
from bessopt.forecast.features import daily_matrix
from bessopt.forecast.probabilistic import TAUS, copula_correlation, sample_scenarios

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def pick_day(Q: pd.DataFrame, Y: pd.DataFrame) -> pd.Timestamp:
    """Most volatile 15-min-era day whose 90 % interval coverage is >= 0.85."""
    best, best_range = None, -1.0
    for d in Q.index.get_level_values("date").unique():
        if d < pd.Timestamp("2025-10-01"):
            continue
        q, y = Q.loc[d], Y.loc[d].to_numpy()
        cov = ((y >= q.loc[0.05].to_numpy()) & (y <= q.loc[0.95].to_numpy())).mean()
        if cov >= 0.85 and np.ptp(y) > best_range:
            best, best_range = d, np.ptp(y)
    return best


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None)
    ap.add_argument("--n", type=int, default=8, help="scenarios drawn per panel")
    args = ap.parse_args()

    Q = pd.read_parquet(RESULTS_DIR / "quantiles_er_c182.parquet")
    F = pd.concat(
        [pd.read_parquet(RESULTS_DIR / f) for f in ("forecasts_lear_w730_r7_pre2025.parquet", "forecasts_lear_w730_r7.parquet")]
    ).sort_index()
    F = F[~F.index.duplicated(keep="last")]
    Y = daily_matrix(entsoe.day_ahead_prices("2021-01-01", "2026-10-14"), "15min")

    day = pd.Timestamp(args.day) if args.day else pick_day(Q, Y)
    q = Q.loc[day].to_numpy()
    y = Y.loc[day].to_numpy()
    R = copula_correlation(F, Y, day)
    rng = np.random.default_rng(7)
    Xc = sample_scenarios(q, args.n, R, rng)
    Xi = sample_scenarios(q, args.n, None, rng)
    t = np.arange(96) / 4

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    (a, b), (c, d) = axes
    ylim = (min(q.min(), y.min(), Xc.min(), Xi.min()) - 10, max(q.max(), y.max(), Xc.max(), Xi.max()) + 10)
    k = {tau: i for i, tau in enumerate(TAUS)}

    # A: marginal distribution per slot
    a.fill_between(t, q[k[0.05]], q[k[0.95]], step="post", color=BLUE, alpha=0.18, lw=0, label="90 % interval")
    a.fill_between(t, q[k[0.25]], q[k[0.75]], step="post", color=BLUE, alpha=0.38, lw=0, label="50 % interval")
    a.step(t, F.loc[day].to_numpy(), where="post", color=BLUE, lw=2, label="LEAR point forecast")
    a.step(t, y, where="post", color=ORANGE, lw=2, label="Actual price")
    style(a, "A. Forecast distribution per quarter-hour")
    a.legend(frameon=False, fontsize=8.5, loc="upper left")

    # B / C: scenarios
    for ax, X, title in ((b, Xc, "B. Copula scenarios (dependence across slots)"),
                         (c, Xi, "C. Independent sampling (no dependence)")):
        for s in range(len(X)):
            ax.step(t, X[s], where="post", color=BLUE, lw=1, alpha=0.75)
        ax.step(t, y, where="post", color=ORANGE, lw=2)
        style(ax, title)
        ax.text(0.01, 0.97, f"{len(X)} scenarios (blue), actual (orange)", transform=ax.transAxes,
                fontsize=8.5, color=INK2, va="top")
    for ax in (a, b, c):
        ax.set_ylim(*ylim)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlabel("Hour of day", fontsize=9, color=INK2)
        ax.set_ylabel("EUR/MWh", fontsize=9, color=INK2)

    # D: copula correlation
    im = d.imshow(R, cmap="Blues", vmin=0, vmax=1, extent=(0, 24, 24, 0))
    d.set_xticks(range(0, 25, 6))
    d.set_yticks(range(0, 25, 6))
    d.set_xlabel("Hour of day", fontsize=9, color=INK2)
    d.set_ylabel("Hour of day", fontsize=9, color=INK2)
    d.tick_params(colors=INK2, labelsize=8.5)
    d.set_title("D. Copula correlation of forecast errors", loc="left", fontsize=10.5, color=INK, pad=8)
    cb = fig.colorbar(im, ax=d, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    cb.outline.set_visible(False)

    fig.suptitle(f"SE4 day-ahead price scenarios for {day:%Y-%m-%d} (forecast made 12:00 the day before)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    out = RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"scenarios_{day:%Y-%m-%d}.png"
    fig.savefig(path, facecolor="white")
    print(path)


if __name__ == "__main__":
    main()
