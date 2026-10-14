"""Daily design matrices for day-ahead price forecasting.

Each row is a delivery day D. Targets are the 96 quarter-hour SE4 prices of D;
features are restricted to information available at gate closure (12:00 D-1):

* SE4 prices of D-1 at 15-min resolution, and hourly means of D-2, D-3, D-7
* hourly prices of coupled zones on D-1
* TSO load and wind forecasts for D and D-1 (hourly; published before
  gate closure)
* day-of-week dummies, a 15-min-MTU regime dummy, and a flag for days whose
  exogenous forecasts were missing and imputed

Before 2025-10-01 prices are hourly, so quarter-hour targets (and D-1
quarter-hour lags) are the hourly price repeated four times. The regime
dummy lets each per-slot model learn the mean intra-hour profile from the
15-min era alone.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from bessopt.config import MARKET_TZ
from bessopt.data.timegrid import QUARTER_HOUR_GO_LIVE, infer_resolution, to_resolution

QH_SLOTS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
H_SLOTS = [f"{h:02d}:00" for h in range(24)]
BINARY_PREFIXES = ("dow_", "regime_qh", "exog_imputed")


def daily_matrix(series: pd.Series, freq: str) -> pd.DataFrame:
    """Reshape a time series into a (day x clock-slot) matrix.

    DST is handled by clock time: on 25-hour days the repeated 02:xx slots
    are averaged; on 23-hour days the missing 02:xx slots are filled from the
    neighbouring slot.
    """
    slots = QH_SLOTS if pd.Timedelta(freq) == pd.Timedelta(minutes=15) else H_SLOTS
    s = to_resolution(series.dropna(), freq).tz_convert(MARKET_TZ)
    df = pd.DataFrame(
        {"date": pd.to_datetime(s.index.date), "slot": s.index.strftime("%H:%M"), "v": s.to_numpy()}
    )
    m = df.groupby(["date", "slot"])["v"].mean().unstack("slot").reindex(columns=slots)
    m = m.ffill(axis=1).bfill(axis=1)
    m.index.name = "date"
    return m


def slots_to_grid(values: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Map a 96-slot daily forecast onto a delivery day's native grid."""
    if len(grid) > 1 and infer_resolution(grid) == pd.Timedelta(hours=1):
        values = values.groupby(values.index.str[:2] + ":00").mean()
    out = values.reindex(grid.strftime("%H:%M")).ffill().bfill()
    return pd.Series(out.to_numpy(), index=grid, name="forecast")


def scenarios_to_grid(X: np.ndarray, grid: pd.DatetimeIndex) -> np.ndarray:
    """Map (S x 96 slot) scenarios onto a delivery day's grid -> (S x T)."""
    df = pd.DataFrame(X.T, index=QH_SLOTS)
    if len(grid) > 1 and infer_resolution(grid) == pd.Timedelta(hours=1):
        df = df.groupby(df.index.str[:2] + ":00").mean()
    return df.reindex(grid.strftime("%H:%M")).ffill().bfill().to_numpy().T


def _lag(m: pd.DataFrame, days: int, prefix: str) -> pd.DataFrame:
    out = m.copy()
    out.index = out.index + pd.Timedelta(days=days)
    return out.add_prefix(prefix)


def _impute_days(m: pd.DataFrame, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    """Reindex to all days; fill missing days with the last available day."""
    m = m.reindex(index)
    missing = m.isna().any(axis=1)
    return m.ffill(), missing


def build_dataset(
    prices: pd.Series,
    neighbours: dict[str, pd.Series],
    load_fc: pd.Series,
    wind_fc: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, Y) indexed by delivery date (naive midnight timestamps).

    X has a row for every day whose features are computable, including days
    whose prices are not yet known (the day being forecast). Y has the 96
    quarter-hour prices where known.
    """
    Y = daily_matrix(prices, "15min")
    Ph = daily_matrix(prices, "1h")

    last_exog = min(load_fc.dropna().index.max(), wind_fc.dropna().index.max())
    days = pd.date_range(Y.index.min(), pd.Timestamp(last_exog.date()), freq="D")

    L, l_miss = _impute_days(daily_matrix(load_fc, "1h"), days)
    W, w_miss = _impute_days(daily_matrix(wind_fc, "1h"), days)

    blocks = [
        _lag(Y, 1, "p_d1_"),
        _lag(Ph, 2, "p_d2_"),
        _lag(Ph, 3, "p_d3_"),
        _lag(Ph, 7, "p_d7_"),
    ]
    for name, s in neighbours.items():
        blocks.append(_lag(daily_matrix(s, "1h"), 1, f"{name}_d1_"))
    blocks += [
        L.add_prefix("load_d0_"),
        _lag(L, 1, "load_d1_"),
        W.add_prefix("wind_d0_"),
        _lag(W, 1, "wind_d1_"),
    ]

    X = pd.concat([b.reindex(days) for b in blocks], axis=1)
    dow = pd.get_dummies(days.dayofweek, prefix="dow", drop_first=True, dtype=float)
    dow.index = days
    X = X.join(dow)
    X["regime_qh"] = (days.date >= QUARTER_HOUR_GO_LIVE).astype(float)
    X["exog_imputed"] = (l_miss | w_miss).astype(float)
    X = X.dropna()
    X.index.name = "date"
    return X, Y.reindex(columns=QH_SLOTS)


def load_dataset(start: str | dt.date, end: str | dt.date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the dataset from cached ENTSO-E data."""
    from bessopt.data import entsoe

    s, e = pd.Timestamp(start), pd.Timestamp(end)
    prices = entsoe.day_ahead_prices(s, e)
    nb = {
        z: entsoe.day_ahead_prices(s, e, zone=entsoe.NEIGHBOURS[z]) for z in ("SE3", "DK2", "DE_LU")
    }
    ws = entsoe.wind_solar_forecast(s, e)
    return build_dataset(prices, nb, entsoe.load_forecast(s, e), ws["wind_onshore"])


def is_binary(columns: pd.Index) -> np.ndarray:
    return np.array([c.startswith(BINARY_PREFIXES) for c in columns])
