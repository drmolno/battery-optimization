"""Point-forecast accuracy on each day's native market resolution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bessopt.backtest.strategies import NaivePersistence
from bessopt.data.timegrid import split_by_delivery_day
from bessopt.forecast.features import slots_to_grid


def errors_by_day(forecasts: dict[str, pd.DataFrame], prices: pd.Series, start, end) -> pd.DataFrame:
    """Daily MAE and RMSE per model, plus the naive persistence forecast.

    Forecasts are mapped to the native grid (hourly before 2025-10-01), so
    pre-switch days are scored on hourly averages of the 96-slot forecast.
    """
    days = split_by_delivery_day(prices)
    rows = []
    for d, actual in days.items():
        if not (pd.Timestamp(start).date() <= d < pd.Timestamp(end).date()):
            continue
        known = prices[prices.index < actual.index[0]]
        preds = {"naive": NaivePersistence.forecast(d, actual.index, known)}
        for name, table in forecasts.items():
            if pd.Timestamp(d) in table.index:
                preds[name] = slots_to_grid(table.loc[pd.Timestamp(d)], actual.index)
        for name, p in preds.items():
            e = p.to_numpy() - actual.to_numpy()
            rows.append(
                {"date": d, "model": name, "mae": np.abs(e).mean(), "rmse": np.sqrt((e**2).mean())}
            )
    return pd.DataFrame(rows)


def summarize_errors(daily: pd.DataFrame, by_regime: bool = True) -> pd.DataFrame:
    """MAE, RMSE and rMAE (MAE relative to naive) per model and regime."""
    df = daily.copy()
    groups = ["model"]
    if by_regime:
        df["regime"] = np.where(pd.to_datetime(df["date"]) >= pd.Timestamp("2025-10-01"), "15min", "hourly")
        groups = ["regime", "model"]
    out = df.groupby(groups)[["mae", "rmse"]].mean()
    naive = out.xs("naive", level="model") if by_regime else out.loc["naive"]
    if by_regime:
        out["rmae"] = out["mae"] / out.index.get_level_values("regime").map(naive["mae"]).to_numpy()
    else:
        out["rmae"] = out["mae"] / naive["mae"]
    return out
