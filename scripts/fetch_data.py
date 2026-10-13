"""Download and cache all ENTSO-E series used by the project.

ENTSO-E answers a one-month query in ~25 s, so requests are issued in
parallel: every (series, year) pair is a separate job. Jobs write disjoint
monthly cache files, so they don't conflict; cached months are skipped.

Usage:
    python scripts/fetch_data.py --start 2021-01-01 --workers 8
"""

from __future__ import annotations

import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from bessopt.data import entsoe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=str(dt.date.today() + dt.timedelta(days=2)))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    series = {
        "prices SE4": entsoe.day_ahead_prices,
        "load forecast SE4": entsoe.load_forecast,
        "wind/solar forecast SE4": entsoe.wind_solar_forecast,
    }
    for name, code in entsoe.NEIGHBOURS.items():
        series[f"prices {name}"] = lambda s, e, code=code: entsoe.day_ahead_prices(s, e, zone=code)

    bounds = [start, *pd.date_range(start, end, freq="YS", inclusive="neither"), end]
    chunks = list(zip(bounds[:-1], bounds[1:]))

    jobs = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for name, fn in series.items():
            for s, e in chunks:
                jobs[pool.submit(fn, s, e)] = (name, s.year)
        failed = []
        for i, fut in enumerate(as_completed(jobs), 1):
            name, year = jobs[fut]
            try:
                n = len(fut.result())
                print(f"[{i}/{len(jobs)}] {name:<26} {year}  {n:>6} rows", flush=True)
            except Exception as ex:  # keep going; report at the end
                failed.append((name, year))
                print(f"[{i}/{len(jobs)}] {name:<26} {year}  FAILED: {type(ex).__name__}: {ex}", flush=True)

    print("Done." if not failed else f"Done with {len(failed)} failures: {failed}")


if __name__ == "__main__":
    main()
