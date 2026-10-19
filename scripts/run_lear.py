"""Rolling out-of-sample LEAR forecasts for SE4, scored on accuracy and profit.

Usage:
    python scripts/run_lear.py --start 2025-01-01 --end 2026-10-13 --refit-every 7
    python scripts/run_lear.py --reuse    # skip fitting, reuse saved forecasts
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from bessopt.backtest import NaivePersistence, PerfectForesight, PointForecast, run_backtest
from bessopt.config import RESULTS_DIR, BatterySpec
from bessopt.data import entsoe
from bessopt.forecast import load_dataset, rolling_forecast
from bessopt.forecast.evaluate import errors_by_day, summarize_errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default=str(dt.date.today()))
    ap.add_argument("--window", type=int, default=730)
    ap.add_argument("--refit-every", type=int, default=7)
    ap.add_argument("--solver", default=None)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--holidays", action="store_true", help="add Swedish holiday flags (no measurable gain)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"lear_w{args.window}_r{args.refit_every}" + ("_hol" if args.holidays else "")
    fc_path = RESULTS_DIR / f"forecasts_{tag}.parquet"
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)

    if args.reuse and fc_path.exists():
        fc = pd.read_parquet(fc_path)
    else:
        X, Y = load_dataset("2021-01-01", end + dt.timedelta(days=2), holidays=args.holidays)
        print(f"Dataset: X {X.shape}, Y {Y.dropna().shape}. Fitting {tag} ...")
        fc, coefs = rolling_forecast(X, Y, start, end, args.window, args.refit_every)
        fc.to_parquet(fc_path)
        nz = pd.DataFrame({c["fit_date"].iloc[0]: (c.drop(columns="fit_date") != 0).mean(axis=1) for c in coefs})
        nz.to_csv(RESULTS_DIR / f"selection_freq_{tag}.csv")

    prices = entsoe.day_ahead_prices(start - dt.timedelta(days=8), end)

    err = summarize_errors(errors_by_day({"lear": fc}, prices, start, end))
    pd.set_option("display.float_format", "{:,.3f}".format)
    print("\nForecast accuracy (EUR/MWh, native resolution):")
    print(err.to_string())

    spec = BatterySpec()
    results = [
        run_backtest(PerfectForesight(spec, args.solver), prices, start, end, spec),
        run_backtest(NaivePersistence(spec, args.solver), prices, start, end, spec),
        run_backtest(PointForecast(fc, "lear", spec, args.solver), prices, start, end, spec),
    ]
    summary = pd.DataFrame([r.summary() for r in results])
    summary["capture_vs_pf"] = summary["profit_eur"] / summary.loc["perfect_foresight", "profit_eur"]
    pd.set_option("display.float_format", "{:,.2f}".format)
    print("\nBacktest:")
    print(summary.to_string())

    summary.to_csv(RESULTS_DIR / f"backtest_{tag}.csv")
    err.to_csv(RESULTS_DIR / f"accuracy_{tag}.csv")
    for r in results:
        r.daily.to_csv(RESULTS_DIR / f"daily_{r.strategy}_{tag}.csv")


if __name__ == "__main__":
    main()
