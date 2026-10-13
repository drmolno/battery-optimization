"""Monthly parquet cache for remote time series.

Data is fetched and stored one calendar month at a time. Months that are not
yet complete (i.e. that end after today's prices could have been published)
are fetched but never written to disk, so the cache never holds partial data.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from bessopt.config import CACHE_DIR, MARKET_TZ

FetchFn = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame | pd.Series]


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    first = start.tz_convert(MARKET_TZ).normalize().replace(day=1)
    months = []
    m = first
    while m < end:
        months.append(m)
        m = (m + pd.offsets.MonthBegin(1)).tz_convert(MARKET_TZ).normalize()
    return months


def cached_monthly(
    source: str,
    key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch: FetchFn,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """Return data in [start, end), fetching uncached months via ``fetch``.

    ``fetch(month_start, month_end)`` must return a DataFrame or Series
    indexed by tz-aware period start.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize(MARKET_TZ)
    if end.tzinfo is None:
        end = end.tz_localize(MARKET_TZ)

    folder = cache_dir / source / key
    folder.mkdir(parents=True, exist_ok=True)
    # Prices for tomorrow may already be out; anything later cannot be.
    complete_before = pd.Timestamp(dt.date.today()).tz_localize(MARKET_TZ)

    frames = []
    for m0 in _month_starts(start, end):
        m1 = (m0 + pd.offsets.MonthBegin(1)).tz_convert(MARKET_TZ).normalize()
        path = folder / f"{m0:%Y-%m}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
        else:
            df = fetch(m0, min(m1, complete_before + pd.Timedelta(days=2)))
            if isinstance(df, pd.Series):
                df = df.to_frame()
            if m1 <= complete_before and not df.empty:
                df.to_parquet(path)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=MARKET_TZ))
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index = out.index.tz_convert(MARKET_TZ)
    return out.loc[(out.index >= start) & (out.index < end)]
