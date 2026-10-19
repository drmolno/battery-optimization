"""Does the 15-minute forecast pay off? Compare, in the 15-minute era:

* perfect foresight on actual 15-min prices vs on hourly averages of them
  (how much value lies within the hour), and
* LEAR at 15-min resolution vs the same forecast averaged to hourly.

All schedules are settled at actual 15-minute prices.

Usage:
    python scripts/run_resolution_check.py --start 2025-10-01 --end 2026-10-13
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from bessopt.backtest import PerfectForesight, PointForecast, run_backtest
from bessopt.config import RESULTS_DIR, BatterySpec
from bessopt.data import entsoe
from bessopt.data.timegrid import to_resolution


class HourlyPerfectForesight(PerfectForesight):
    """Optimizes on hourly averages of actual prices (hourly-constant schedule)."""

    name = "perfect_foresight_hourly"

    def __init__(self, hourly_prices: pd.Series, *a, **kw):
        super().__init__(*a, **kw)
        self.hourly_prices = hourly_prices

    def schedule(self, day, grid, soc0, known_prices, actual_prices=None):
        return super().schedule(day, grid, soc0, known_prices, self.hourly_prices.reindex(grid))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-10-13")
    args = ap.parse_args()
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)

    prices = entsoe.day_ahead_prices(start - dt.timedelta(days=8), end)
    hourly = to_resolution(to_resolution(prices, "1h"), "15min")
    fc = pd.read_parquet(RESULTS_DIR / "forecasts_lear_w730_r7.parquet")
    fc_hourly = fc.T.groupby(fc.columns.str[:2]).transform("mean").T

    spec = BatterySpec()
    strategies = [
        PerfectForesight(spec),
        HourlyPerfectForesight(hourly, spec),
        PointForecast(fc, "lear_15min", spec),
        PointForecast(fc_hourly, "lear_hourly_avg", spec),
    ]
    res = {s.name: run_backtest(s, prices, start, end, spec).daily["profit_eur"] for s in strategies}
    df = pd.DataFrame(res)
    s = df.sum()
    out = pd.DataFrame({"eur_per_day": df.mean(), "share_of_pf": s / s["perfect_foresight"]})
    print(out.round(3).to_string())
    print("days LEAR 15-min beats its hourly average: %.0f %%" % (100 * (df.lear_15min > df.lear_hourly_avg).mean()))
    out.to_csv(RESULTS_DIR / f"resolution_check_{start}_{end}.csv")


if __name__ == "__main__":
    main()
