"""How much does the end-of-day minimum state of charge matter?

Runs perfect foresight and LEAR with the end-of-day minimum at several
levels (5 % = no rule beyond the normal limit).

Usage:
    python scripts/run_end_level.py --start 2025-01-01 --end 2026-10-13
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt

import pandas as pd

from bessopt.backtest import PerfectForesight, PointForecast, run_backtest
from bessopt.config import RESULTS_DIR, BatterySpec
from bessopt.data import entsoe

LEVELS = [0.05, 0.25, 0.50]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-10-13")
    ap.add_argument("--solver", default="HIGHS")
    args = ap.parse_args()
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)

    F = pd.read_parquet(RESULTS_DIR / "forecasts_lear_w730_r7.parquet")
    prices = entsoe.day_ahead_prices(start - dt.timedelta(days=8), end)

    rows = []
    for level in LEVELS:
        spec = dataclasses.replace(BatterySpec(), soc_end_min_frac=level)
        for strat in (PerfectForesight(spec, args.solver), PointForecast(F, "lear", spec, args.solver)):
            d = run_backtest(strat, prices, start, end, spec).daily
            rows.append(
                {
                    "end_min": f"{level:.0%}",
                    "strategy": strat.name,
                    "profit_eur_per_day": d["profit_eur"].mean(),
                    "cycles_per_day": d["full_cycles"].mean(),
                    "mean_end_soc_mwh": d["soc_end"].mean(),
                }
            )
            print(rows[-1], flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / f"end_level_{start}_{end}.csv", index=False)
    print(out.pivot(index="end_min", columns="strategy", values="profit_eur_per_day").round(0).to_string())


if __name__ == "__main__":
    main()
