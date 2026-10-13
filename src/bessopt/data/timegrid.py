"""Time-grid utilities: market timing rules and resolution handling.

Conventions used throughout the project:

* Every time series is indexed by the *start* of its delivery period and is
  tz-aware in the market time zone (Europe/Stockholm).
* A *delivery day* is a local calendar day; it has 23, 24 or 25 hours
  depending on DST, i.e. 92/96/100 quarter-hours after the 15-minute switch.
* SDAC moved from 60- to 15-minute market time units on 2025-10-01
  (first delivery day). Before that, prices are hourly.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from bessopt.config import MARKET_TZ

QUARTER_HOUR_GO_LIVE = dt.date(2025, 10, 1)


def _as_date(d: dt.date | str | pd.Timestamp) -> dt.date:
    return pd.Timestamp(d).date()


def day_bounds(delivery_date: dt.date | str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Local start (inclusive) and end (exclusive) of a delivery day."""
    d = _as_date(delivery_date)
    start = pd.Timestamp(d).tz_localize(MARKET_TZ)
    end = pd.Timestamp(d + dt.timedelta(days=1)).tz_localize(MARKET_TZ)
    return start, end


def gate_closure(delivery_date: dt.date | str) -> pd.Timestamp:
    """SDAC gate closure for a delivery day: 12:00 CET/CEST on D-1.

    Any feature used to forecast prices for ``delivery_date`` must have been
    available strictly before this instant.
    """
    d = _as_date(delivery_date) - dt.timedelta(days=1)
    return pd.Timestamp(dt.datetime.combine(d, dt.time(12, 0))).tz_localize(MARKET_TZ)


def price_publication_time(delivery_date: dt.date | str) -> pd.Timestamp:
    """When day-ahead prices for a delivery day become public.

    Nominally ~12:45 CET on D-1; we use 13:00 as a conservative bound. Since
    this is after gate closure, prices for D are never known when bidding for
    D, but prices for D-1 always are.
    """
    return gate_closure(delivery_date) + pd.Timedelta(hours=1)


def native_resolution(delivery_date: dt.date | str) -> pd.Timedelta:
    """Market time unit of the day-ahead auction for a delivery day."""
    if _as_date(delivery_date) >= QUARTER_HOUR_GO_LIVE:
        return pd.Timedelta(minutes=15)
    return pd.Timedelta(hours=1)


def infer_resolution(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Most common spacing of an index (robust to DST and gaps)."""
    if len(index) < 2:
        raise ValueError("Need at least two timestamps to infer resolution.")
    diffs = pd.Series(index.tz_convert("UTC")).diff().dropna()
    return diffs.mode().iloc[0]


def _period_durations(index: pd.DatetimeIndex) -> pd.TimedeltaIndex:
    """Duration of each period, assuming each lasts until the next start.

    Durations are capped at the typical spacing within each delivery day
    (falling back to the native market resolution for single-period days), so
    gaps in the data are not interpreted as long periods and the last period
    gets the correct length. This handles hourly data after the 15-minute
    switch too, e.g. hourly aggregates.
    """
    utc = pd.Series(index.tz_convert("UTC"))
    dates = pd.Series(index.tz_convert(MARKET_TZ).date)
    within_day = utc.groupby(dates).diff()
    typical = within_day.groupby(dates).transform(lambda x: x.mode().iloc[0] if x.notna().any() else pd.NaT)
    native = dates.map(native_resolution)
    cap = typical.fillna(native)
    nxt = utc.shift(-1)
    dur = (nxt - utc).fillna(cap.iloc[-1])
    return pd.TimedeltaIndex(np.minimum(dur, cap))


def to_resolution(series: pd.Series, freq: str | pd.Timedelta) -> pd.Series:
    """Resample a (possibly mixed-resolution) price series to ``freq``.

    Prices are energy-weighted averages over their period, so:
    * upsampling (1h -> 15min) repeats the value over each sub-period;
    * downsampling (15min -> 1h) takes the arithmetic mean.

    The series is first expanded to the finest grid (15 min), which handles
    series spanning the 2025-10-01 switch.
    """
    freq = pd.Timedelta(freq)
    base = pd.Timedelta(minutes=15)
    if series.empty:
        return series.copy()
    tz = series.index.tz
    dur = _period_durations(series.index)
    counts = (dur / base).astype(int)
    starts = np.repeat(series.index.tz_convert("UTC").values, counts)
    offsets = np.concatenate([np.arange(c) for c in counts]) * base
    fine_index = pd.DatetimeIndex(starts, tz="UTC") + pd.TimedeltaIndex(offsets)
    fine = pd.Series(np.repeat(series.values, counts), index=fine_index, name=series.name)
    if freq == base:
        out = fine
    elif freq > base:
        out = fine.resample(freq).mean().dropna()
    else:
        raise ValueError(f"Resolution finer than {base} not supported.")
    return out.tz_convert(tz)


def split_by_delivery_day(series: pd.Series) -> dict[dt.date, pd.Series]:
    """Group a series by local delivery date."""
    local = series.tz_convert(MARKET_TZ)
    return {d: s for d, s in local.groupby(local.index.date)}


def period_hours(index: pd.DatetimeIndex) -> np.ndarray:
    """Length of each period in hours (for energy = power * dt)."""
    return (_period_durations(index) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
