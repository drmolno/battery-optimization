import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bessopt.config import MARKET_TZ
from bessopt.data.timegrid import (
    gate_closure,
    period_hours,
    price_publication_time,
    split_by_delivery_day,
    to_resolution,
)


def _series(start, periods, freq, values=None):
    idx = pd.date_range(start, periods=periods, freq=freq, tz=MARKET_TZ)
    v = np.arange(periods, dtype=float) if values is None else values
    return pd.Series(v, index=idx)


def test_gate_closure_precedes_own_prices_but_follows_previous_day():
    d = dt.date(2025, 6, 10)
    assert gate_closure(d) == pd.Timestamp("2025-06-09 12:00", tz=MARKET_TZ)
    assert price_publication_time(d) > gate_closure(d)
    assert price_publication_time(d - dt.timedelta(days=1)) < gate_closure(d)


def test_upsample_hourly_repeats_values():
    s = _series("2025-06-01", 3, "h")
    q = to_resolution(s, "15min")
    assert len(q) == 12
    assert (q.iloc[:4] == 0).all() and (q.iloc[-4:] == 2).all()


def test_downsample_quarter_hours_averages():
    s = _series("2025-11-01", 8, "15min")
    h = to_resolution(s, "1h")
    np.testing.assert_allclose(h.to_numpy(), [1.5, 5.5])


def test_mixed_resolution_across_switch():
    hourly = _series("2025-09-30 22:00", 2, "h", np.array([10.0, 20.0]))
    quarter = _series("2025-10-01 00:00", 4, "15min", np.array([1.0, 2.0, 3.0, 4.0]))
    s = pd.concat([hourly, quarter])
    q = to_resolution(s, "15min")
    assert len(q) == 12
    h = to_resolution(s, "1h")
    np.testing.assert_allclose(h.to_numpy(), [10.0, 20.0, 2.5])


@pytest.mark.parametrize(
    "day, expected",
    [("2025-03-30", 23), ("2025-10-26", 25), ("2025-06-01", 24)],
)
def test_dst_day_lengths(day, expected):
    start = pd.Timestamp(day).tz_localize(MARKET_TZ)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(MARKET_TZ)
    idx = pd.date_range(start, end, freq="h", inclusive="left")
    s = pd.Series(1.0, index=idx)
    days = split_by_delivery_day(s)
    assert len(days[pd.Timestamp(day).date()]) == expected
    assert period_hours(idx).sum() == expected
