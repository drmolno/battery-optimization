"""Backtest stochastic bid curves against deterministic strategies on SE4.

All scenario-based strategies share one ScenarioSource, so they see identical
scenarios each day. Everything is settled at realized prices.

Requires (from earlier scripts):
    results/forecasts_lear_w730_r7[_pre2025].parquet   (run_lear.py)
    results/quantiles_er_c182.parquet                    (run_probabilistic.py)

Usage:
    python scripts/run_stochastic.py --start 2025-11-01 --end 2026-01-01      # pilot
    python scripts/run_stochastic.py --start 2025-01-01 --end 2026-10-13 --tag full
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import pandas as pd

from bessopt.backtest import NaivePersistence, PerfectForesight, PointForecast, run_backtest
from bessopt.backtest.strategies import ExpectedValue, ScenarioSource, StochasticCurves
from bessopt.config import RESULTS_DIR, BatterySpec
from bessopt.data import entsoe
from bessopt.forecast.features import daily_matrix
from bessopt.forecast.probabilistic import distribution_mean


def load_inputs(quantiles: str):
    F = pd.concat(
        [pd.read_parquet(RESULTS_DIR / f) for f in ("forecasts_lear_w730_r7_pre2025.parquet", "forecasts_lear_w730_r7.parquet")]
    ).sort_index()
    F = F[~F.index.duplicated(keep="last")]
    Q = pd.read_parquet(RESULTS_DIR / f"quantiles_{quantiles}.parquet")
    return F, Q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-11-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--n-scen", type=int, default=50)
    ap.add_argument("--points", type=int, nargs="+", default=[8], help="curve breakpoints K (several = sweep)")
    ap.add_argument("--exact", action="store_true", help="also run the exact MILP for the first K")
    ap.add_argument("--penalty", type=float, default=50.0, help="imbalance penalty EUR/MWh")
    ap.add_argument("--solver", default="GUROBI")
    ap.add_argument("--moment-match", action="store_true", help="match scenario means to the distribution mean")
    ap.add_argument("--quantiles", default="er_pool2c_c182", help="results/quantiles_<name>.parquet")
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    F, Q = load_inputs(args.quantiles)
    prices_all = entsoe.day_ahead_prices("2021-01-01", end + dt.timedelta(days=1))
    Y = daily_matrix(prices_all, "15min")
    prices = prices_all[prices_all.index >= pd.Timestamp(start - dt.timedelta(days=8)).tz_localize(prices_all.index.tz)]

    spec = BatterySpec()
    source = ScenarioSource(Q, F, Y, n_scen=args.n_scen, moment_match=args.moment_match)

    # Deterministic benchmarks from the forecast distribution itself (no sampling).
    days = [d for d in Q.index.get_level_values("date").unique() if start <= d.date() < end]
    dist_mean = pd.DataFrame({d: distribution_mean(Q.loc[d].to_numpy()) for d in days}, index=Q.columns).T
    dist_median = pd.DataFrame({d: Q.loc[(d, 0.5)].to_numpy() for d in days}, index=Q.columns).T

    strategies = [
        PerfectForesight(spec, args.solver),
        NaivePersistence(spec, args.solver),
        PointForecast(F, "lear_point", spec, args.solver),
        PointForecast(dist_median, "dist_median", spec, args.solver),
        PointForecast(dist_mean, "dist_mean", spec, args.solver),
        ExpectedValue(source, spec, args.solver),
    ]
    for K in args.points:
        st = StochasticCurves(source, spec, args.solver, n_points=K)
        st.name = f"stochastic_K{K}"
        strategies.append(st)
    if args.exact:
        st = StochasticCurves(source, spec, args.solver, n_points=args.points[0], power_rule="exclusive")
        st.name = f"stochastic_K{args.points[0]}_exact"
        strategies.append(st)

    results = []
    for s in strategies:
        t0 = time.perf_counter()
        r = run_backtest(s, prices, start, end, spec, imbalance_penalty=args.penalty)
        print(f"{s.name:<26} {time.perf_counter() - t0:7.1f}s  {r.summary()['profit_eur_per_day']:8.1f} EUR/day", flush=True)
        results.append(r)

    summary = pd.DataFrame([r.summary() for r in results])
    pf = summary.loc["perfect_foresight", "profit_eur"]
    summary["capture_vs_pf"] = summary["profit_eur"] / pf
    if "expected_value" in summary.index:
        ev = summary.loc["expected_value", "profit_eur_per_day"]
        summary["vs_ev_eur_per_day"] = summary["profit_eur_per_day"] - ev
    pd.set_option("display.float_format", "{:,.2f}".format)
    print(summary.to_string())

    tag = f"{args.tag}_{args.quantiles}_{start}_{end}_S{args.n_scen}" + ("_mm" if args.moment_match else "")
    summary.to_csv(RESULTS_DIR / f"stochastic_summary_{tag}.csv")
    pd.concat({r.strategy: r.daily for r in results}, names=["strategy"]).to_csv(
        RESULTS_DIR / f"stochastic_daily_{tag}.csv"
    )


if __name__ == "__main__":
    main()
