"""Wear (degradation) cost: planning value vs true value.

The optimizer plans with a wear cost kappa_plan (EUR per MWh discharged). Every
strategy is then charged the *true* wear cost kappa_true. Because the backtest
records discharged energy per day, kappa_true is applied after the run, so
only kappa_plan needs separate runs.

kappa_true central value: replacement cost / lifetime discharged energy
  = 20 MWh x 80 kEUR/MWh / (7,000 cycles x 18 MWh usable) ~ 13 EUR/MWh,
rounded to 15; low/high cases 8 and 22.

Usage:
    python scripts/run_wear.py --start 2025-01-01 --end 2026-10-13
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt

import pandas as pd

from bessopt.backtest import PerfectForesight, PointForecast, run_backtest
from bessopt.config import RESULTS_DIR, BatterySpec
from bessopt.data import entsoe

PLAN = [0, 5, 10, 15, 20, 30, 50]
TRUE = {"low": 8.0, "central": 15.0, "high": 22.0}


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
    for k in PLAN:
        spec = dataclasses.replace(BatterySpec(), degradation_eur_per_mwh=float(k))
        for strat in (PerfectForesight(spec, args.solver), PointForecast(F, "lear_point", spec, args.solver)):
            r = run_backtest(strat, prices, start, end, spec)
            d = r.daily
            row = {
                "strategy": strat.name,
                "kappa_plan": k,
                "revenue_eur_per_day": d["revenue_eur"].mean(),
                "discharged_mwh_per_day": d["discharged_mwh"].mean(),
                "cycles_per_day": d["full_cycles"].mean(),
            }
            for name, kt in TRUE.items():
                row[f"net_{name}"] = row["revenue_eur_per_day"] - kt * row["discharged_mwh_per_day"]
            rows.append(row)
            print(f"{strat.name:<18} plan={k:>3}  revenue {row['revenue_eur_per_day']:7.0f}  "
                  f"cycles/day {row['cycles_per_day']:.2f}  net@15 {row['net_central']:7.0f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / f"wear_sweep_{start}_{end}.csv", index=False)
    pd.set_option("display.float_format", "{:,.1f}".format)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
