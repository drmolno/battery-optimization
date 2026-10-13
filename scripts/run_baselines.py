"""Backtest perfect-foresight and naive-persistence dispatch on SE4 prices.

Usage:
    python scripts/run_baselines.py --start 2024-01-01 --end 2025-01-01
    python scripts/run_baselines.py --source entsoe --solver GUROBI
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from bessopt.backtest import NaivePersistence, PerfectForesight, run_backtest
from bessopt.config import RESULTS_DIR, BatterySpec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--source", choices=["elpris", "entsoe"], default="elpris")
    ap.add_argument("--solver", default=None)
    args = ap.parse_args()

    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    # One extra day of history so the naive strategy has a forecast on day one.
    fetch_start = start - dt.timedelta(days=2)
    if args.source == "entsoe":
        from bessopt.data.entsoe import day_ahead_prices
    else:
        from bessopt.data.elpris import day_ahead_prices
    prices = day_ahead_prices(fetch_start, end)
    print(f"Loaded {len(prices)} price periods, {prices.index.min()} -> {prices.index.max()}")

    spec = BatterySpec()
    results = [
        run_backtest(PerfectForesight(spec, args.solver), prices, start, end, spec),
        run_backtest(NaivePersistence(spec, args.solver), prices, start, end, spec),
    ]
    summary = pd.DataFrame([r.summary() for r in results])
    pf = summary.loc[summary.index[0], "profit_eur"]
    summary["capture_vs_pf"] = summary["profit_eur"] / pf
    pd.set_option("display.float_format", "{:,.2f}".format)
    print(summary.to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"{args.source}_{start}_{end}"
    summary.to_csv(RESULTS_DIR / f"baselines_{tag}.csv")
    for r in results:
        r.daily.to_csv(RESULTS_DIR / f"daily_{r.strategy}_{tag}.csv")
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
