"""Probabilistic SE4 price forecasts: QRA vs empirical residuals, plus copula
scenarios scored against independent sampling.

Requires out-of-sample LEAR forecasts covering a calibration period before
--start (results/forecasts_lear_w730_r7_pre2025.parquet) and the test period
(results/forecasts_lear_w730_r7.parquet).

Usage:
    python scripts/run_probabilistic.py --start 2025-01-01 --end 2026-10-13
    python scripts/run_probabilistic.py --reuse     # rescore saved quantiles
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from bessopt.config import RESULTS_DIR
from bessopt.data import entsoe
from bessopt.forecast.features import daily_matrix
from bessopt.forecast.probabilistic import (
    copula_correlation,
    energy_score,
    pinball,
    rolling_quantiles,
    sample_scenarios,
    variogram_score,
)


def load_point_forecasts() -> pd.DataFrame:
    parts = [
        pd.read_parquet(RESULTS_DIR / "forecasts_lear_w730_r7_pre2025.parquet"),
        pd.read_parquet(RESULTS_DIR / "forecasts_lear_w730_r7.parquet"),
    ]
    F = pd.concat(parts).sort_index()
    return F[~F.index.duplicated(keep="last")]


def score_marginals(Q: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in Q.index.get_level_values("date").unique():
        q = Q.loc[d].to_numpy()
        y = Y.loc[d].to_numpy()
        tau = Q.loc[d].index.to_numpy()
        lo90, hi90 = q[tau == 0.05][0], q[tau == 0.95][0]
        lo50, hi50 = q[tau == 0.25][0], q[tau == 0.75][0]
        rows.append(
            {
                "date": d,
                "pinball": pinball(q, y, tau).mean(),
                "mae_median": np.abs(q[tau == 0.5][0] - y).mean(),
                "cov90": ((y >= lo90) & (y <= hi90)).mean(),
                "cov50": ((y >= lo50) & (y <= hi50)).mean(),
                "width90": (hi90 - lo90).mean(),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-10-13")
    ap.add_argument("--calib", type=int, default=182)
    ap.add_argument("--n-scen", type=int, default=200)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    F = load_point_forecasts()
    Y = daily_matrix(entsoe.day_ahead_prices("2021-01-01", end + pd.Timedelta(days=1)), "15min")

    quantiles = {}
    for method in ("er", "qra"):
        path = RESULTS_DIR / f"quantiles_{method}_c{args.calib}.parquet"
        if args.reuse and path.exists():
            quantiles[method] = pd.read_parquet(path)
        else:
            print(f"Fitting {method} ...", flush=True)
            quantiles[method] = rolling_quantiles(F, Y, start, end, method, args.calib)
            quantiles[method].to_parquet(path)

    regime = lambda idx: np.where(idx >= pd.Timestamp("2025-10-01"), "15min", "hourly")
    marg = pd.concat(
        [score_marginals(Q, Y).assign(method=m) for m, Q in quantiles.items()]
    )
    marg["regime"] = regime(marg.index)
    summary = marg.groupby(["regime", "method"]).mean()
    pd.set_option("display.float_format", "{:,.3f}".format)
    print("\nMarginal scores (EUR/MWh; coverage nominal 0.90 / 0.50):")
    print(summary.to_string())

    # Joint scenarios: copula vs independent sampling on QRA marginals.
    rng = np.random.default_rng(0)
    Q = quantiles["qra"]
    rows = []
    for i, d in enumerate(Q.index.get_level_values("date").unique()):
        q, y = Q.loc[d].to_numpy(), Y.loc[d].to_numpy()
        R = copula_correlation(F, Y, d, args.calib)
        for name, corr in (("copula", R), ("independent", None)):
            X = sample_scenarios(q, args.n_scen, corr, rng)
            rows.append(
                {"date": d, "scenarios": name, "energy": energy_score(X, y), "variogram": variogram_score(X, y)}
            )
        if (i + 1) % 100 == 0:
            print(f"  scenarios scored for {i + 1} days", flush=True)
    joint = pd.DataFrame(rows).set_index("date")
    joint["regime"] = regime(joint.index)
    print("\nJoint scenario scores (lower is better):")
    print(joint.groupby(["regime", "scenarios"]).mean().to_string())

    marg.to_csv(RESULTS_DIR / f"prob_marginal_daily_c{args.calib}.csv")
    joint.to_csv(RESULTS_DIR / f"prob_joint_daily_c{args.calib}.csv")
    summary.to_csv(RESULTS_DIR / f"prob_marginal_summary_c{args.calib}.csv")


if __name__ == "__main__":
    main()
