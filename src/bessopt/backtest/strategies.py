"""Bidding strategies.

A strategy is called once per delivery day, before gate closure, and returns a
physical schedule (charge/discharge power per period). It may only use
information available at ``gate_closure(day)``; the runner enforces this for
prices by passing only prices published before that instant.

For now schedules are submitted as price-independent (fixed-quantity) bids,
so the schedule executes regardless of the clearing price. Price-dependent bid
curves come with the stochastic strategy.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol

import pandas as pd

from bessopt.config import BatterySpec
from bessopt.data.timegrid import day_bounds, native_resolution, to_resolution
from bessopt.optim.battery import DeterministicDispatch


class Strategy(Protocol):
    name: str

    def schedule(
        self,
        day: dt.date,
        grid: pd.DatetimeIndex,
        soc0: float,
        known_prices: pd.Series,
        actual_prices: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame on ``grid`` with columns charge_mw, discharge_mw."""
        ...


class PerfectForesight:
    """Optimizes against realized prices: an upper bound, not a strategy."""

    name = "perfect_foresight"

    def __init__(self, spec: BatterySpec = BatterySpec(), solver: str | None = None):
        self.dispatch = DeterministicDispatch(spec, solver)

    def schedule(self, day, grid, soc0, known_prices, actual_prices=None):
        if actual_prices is None:
            raise ValueError("PerfectForesight needs actual prices.")
        res = self.dispatch.solve(actual_prices, soc0=soc0)
        return res.schedule[["charge_mw", "discharge_mw"]]


class NaivePersistence:
    """Assume tomorrow's prices equal today's (the last fully known day).

    The forecast is mapped onto the delivery day's grid by local clock time,
    which handles DST days and the hourly to 15-minute switch.
    """

    name = "naive_persistence"

    def __init__(self, spec: BatterySpec = BatterySpec(), solver: str | None = None):
        self.dispatch = DeterministicDispatch(spec, solver)

    @staticmethod
    def forecast(day: dt.date, grid: pd.DatetimeIndex, known_prices: pd.Series) -> pd.Series:
        prev_start, prev_end = day_bounds(day - dt.timedelta(days=1))
        prev = known_prices.loc[(known_prices.index >= prev_start) & (known_prices.index < prev_end)]
        if prev.empty:
            raise ValueError(f"No known prices for {day - dt.timedelta(days=1)}")
        prev = to_resolution(prev, native_resolution(day))
        by_clock = pd.Series(prev.to_numpy(), index=prev.index.strftime("%H:%M"))
        by_clock = by_clock[~by_clock.index.duplicated()]
        fc = by_clock.reindex(grid.strftime("%H:%M")).ffill().bfill()
        return pd.Series(fc.to_numpy(), index=grid, name="forecast")

    def schedule(self, day, grid, soc0, known_prices, actual_prices=None):
        fc = self.forecast(day, grid, known_prices)
        res = self.dispatch.solve(fc, soc0=soc0)
        return res.schedule[["charge_mw", "discharge_mw"]]


class PointForecast:
    """Deterministic dispatch against a precomputed point forecast.

    ``forecasts`` is a (day x 96 slot) table produced out-of-sample, e.g. by
    ``bessopt.forecast.rolling_forecast``, which is responsible for using
    only information available at gate closure.
    """

    def __init__(
        self,
        forecasts: pd.DataFrame,
        name: str,
        spec: BatterySpec = BatterySpec(),
        solver: str | None = None,
    ):
        self.forecasts = forecasts
        self.name = name
        self.dispatch = DeterministicDispatch(spec, solver)

    def schedule(self, day, grid, soc0, known_prices, actual_prices=None):
        from bessopt.forecast.features import slots_to_grid

        fc = slots_to_grid(self.forecasts.loc[pd.Timestamp(day)], grid)
        res = self.dispatch.solve(fc, soc0=soc0)
        return res.schedule[["charge_mw", "discharge_mw"]]
