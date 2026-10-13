import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bessopt.backtest import NaivePersistence, PerfectForesight, run_backtest
from bessopt.config import MARKET_TZ, BatterySpec

SPEC = BatterySpec()


def _synthetic_prices(days=10, seed=1):
    """Hourly prices before 2025-10-01, quarter-hourly after (crosses the switch)."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-09-26").tz_localize(MARKET_TZ)
    end = start + pd.Timedelta(days=days)
    switch = pd.Timestamp("2025-10-01").tz_localize(MARKET_TZ)
    idx = pd.date_range(start, switch, freq="h", inclusive="left").append(
        pd.date_range(switch, end, freq="15min", inclusive="left")
    )
    hour = idx.hour + idx.minute / 60
    shape = 50 + 40 * np.sin(2 * np.pi * (hour - 8) / 24)
    return pd.Series(shape + rng.normal(0, 10, len(idx)), index=idx)


def test_perfect_foresight_dominates_naive():
    prices = _synthetic_prices()
    a, b = dt.date(2025, 9, 27), dt.date(2025, 10, 6)
    pf = run_backtest(PerfectForesight(SPEC), prices, a, b, SPEC)
    nv = run_backtest(NaivePersistence(SPEC), prices, a, b, SPEC)
    assert len(pf.daily) == len(nv.daily) == 9
    assert pf.summary()["profit_eur"] >= nv.summary()["profit_eur"]
    assert nv.summary()["profit_eur"] > 0  # the daily shape is predictable


def test_naive_forecast_uses_only_previous_day():
    prices = _synthetic_prices()
    day = dt.date(2025, 10, 1)  # first quarter-hour day, previous day hourly
    grid = prices[prices.index.date == day].index
    known = prices[prices.index < pd.Timestamp("2025-10-01").tz_localize(MARKET_TZ)]
    fc = NaivePersistence.forecast(day, grid, known)
    assert len(fc) == 96
    prev = prices[prices.index.date == dt.date(2025, 9, 30)]
    np.testing.assert_allclose(fc.iloc[:4], prev.iloc[0])
    with pytest.raises(ValueError):
        NaivePersistence.forecast(day, grid, known.iloc[:24])
