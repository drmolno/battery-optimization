"""Figure: cumulative profit of perfect foresight, LEAR and naive (README hero).

Reads the daily backtest results written by run_lear.py.

Usage:
    python scripts/plot_cumulative.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from bessopt.config import PROJECT_ROOT, RESULTS_DIR

TAG = "lear_w730_r7"
SERIES = [  # (file key, label, colour, line style)
    ("perfect_foresight", "Perfect foresight", "#52514e", (0, (4, 2))),
    ("lear", "LEAR forecast", "#2a78d6", "-"),
    ("naive_persistence", "Naive forecast", "#eb6834", "-"),
]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    daily = {k: pd.read_csv(RESULTS_DIR / f"daily_{k}_{TAG}.csv", parse_dates=["date"], index_col="date")["profit_eur"]
             for k, *_ in SERIES}
    pf_total = daily["perfect_foresight"].sum()

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
    for key, label, colour, ls in SERIES:
        cum = daily[key].cumsum() / 1e6
        ax.plot(cum.index, cum.to_numpy(), color=colour, lw=2, ls=ls, label=label)
        share = daily[key].sum() / pf_total
        ax.annotate(
            f"{label}\n{cum.iloc[-1]:.2f} M EUR ({share:.0%})",
            xy=(cum.index[-1], cum.iloc[-1]), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=9, color=INK,
        )

    switch = pd.Timestamp("2025-10-01")
    ax.axvline(switch, color=INK2, lw=0.8, ls=":")
    ax.text(switch, ax.get_ylim()[1] * 0.97, "  15-minute market\n  from 1 Oct 2025", fontsize=8.5, color=INK2, va="top")

    ax.set_ylabel("Cumulative profit after wear (M EUR)", fontsize=9.5, color=INK2)
    ax.set_title("SE4 day-ahead trading, 10 MW / 20 MWh battery, Jan 2025 - Oct 2026",
                 loc="left", fontsize=11, color=INK, pad=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)

    out = PROJECT_ROOT / "docs" / "cumulative_profit.png"
    fig.savefig(out, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
