"""Rolling day-by-day backtest with state of charge carried across days."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bessopt.backtest.strategies import Strategy
from bessopt.config import BatterySpec
from bessopt.optim.stochastic import BidCurves
from bessopt.data.timegrid import (
    day_bounds,
    gate_closure,
    period_hours,
    price_publication_time,
    split_by_delivery_day,
)


@dataclass
class BacktestResult:
    strategy: str
    daily: pd.DataFrame  # one row per delivery day
    schedule: pd.DataFrame  # full period-level schedule

    def summary(self) -> pd.Series:
        d = self.daily
        return pd.Series(
            {
                "days": len(d),
                "revenue_eur": d["revenue_eur"].sum(),
                "degradation_eur": d["degradation_eur"].sum(),
                "imbalance_eur": d["imbalance_eur"].sum(),
                "profit_eur": d["profit_eur"].sum(),
                "profit_eur_per_day": d["profit_eur"].mean(),
                "full_cycles": d["full_cycles"].sum(),
            },
            name=self.strategy,
        )


def settle(
    schedule: pd.DataFrame, prices: pd.Series, spec: BatterySpec, soc0: float
) -> tuple[pd.DataFrame, dict]:
    """Evaluate a schedule against realized prices and simulate SoC."""
    dt_h = period_hours(prices.index)
    c = schedule["charge_mw"].to_numpy()
    d = schedule["discharge_mw"].to_numpy()
    soc = soc0 + np.cumsum(dt_h * (spec.eta_charge * c - d / spec.eta_discharge))
    tol = 1e-6
    if soc.min() < spec.soc_min - tol or soc.max() > spec.soc_max + tol:
        raise ValueError("Schedule violates SoC limits.")
    out = schedule.assign(price=prices.to_numpy(), soc_mwh=soc)
    revenue = float(prices.to_numpy() * dt_h @ (d - c))
    degradation = float(spec.degradation_eur_per_mwh * dt_h @ d)
    stats = dict(
        revenue_eur=revenue,
        degradation_eur=degradation,
        imbalance_eur=0.0,
        profit_eur=revenue - degradation,
        discharged_mwh=float(dt_h @ d),
        imbalance_mwh=0.0,
        full_cycles=float(dt_h @ d) / spec.energy_mwh,
        soc_end=float(soc[-1]),
    )
    return out, stats


def settle_curves(
    curves: BidCurves,
    prices: pd.Series,
    spec: BatterySpec,
    soc0: float,
    imbalance_penalty: float = 50.0,
) -> tuple[pd.DataFrame, dict]:
    """Clear bid curves at realized prices and simulate physical delivery.

    Curves are cleared period by period, so the cleared quantities need not
    be SoC-feasible. The battery follows them as far as physically possible;
    the deviation ``delta = cleared - delivered`` is settled in the imbalance
    market, approximated as the day-ahead price worsened by
    ``imbalance_penalty`` (EUR/MWh). The net effect on revenue is

        price * delivered - imbalance_penalty * |delta|.
    """
    dt_h = period_hours(prices.index)
    p = prices.to_numpy(dtype=float)
    y_bid = curves.dispatch(p)
    T = len(p)
    c, d, soc = np.zeros(T), np.zeros(T), np.empty(T)
    e = soc0
    for t in range(T):
        if y_bid[t] >= 0:
            d[t] = min(y_bid[t], spec.power_mw, max(e - spec.soc_min, 0.0) * spec.eta_discharge / dt_h[t])
        else:
            c[t] = min(-y_bid[t], spec.power_mw, max(spec.soc_max - e, 0.0) / (spec.eta_charge * dt_h[t]))
        e = e + dt_h[t] * (spec.eta_charge * c[t] - d[t] / spec.eta_discharge)
        soc[t] = e
    delta = y_bid - (d - c)
    revenue = float(p * dt_h @ (d - c))
    imbalance = float(imbalance_penalty * dt_h @ np.abs(delta))
    degradation = float(spec.degradation_eur_per_mwh * dt_h @ d)
    out = pd.DataFrame(
        {"price": p, "cleared_mw": y_bid, "charge_mw": c, "discharge_mw": d, "soc_mwh": soc},
        index=prices.index,
    )
    stats = dict(
        revenue_eur=revenue,
        degradation_eur=degradation,
        imbalance_eur=imbalance,
        profit_eur=revenue - degradation - imbalance,
        discharged_mwh=float(dt_h @ d),
        imbalance_mwh=float(dt_h @ np.abs(delta)),
        full_cycles=float(dt_h @ d) / spec.energy_mwh,
        soc_end=float(soc[-1]),
    )
    return out, stats


def run_backtest(
    strategy: Strategy,
    prices: pd.Series,
    start: dt.date,
    end: dt.date,
    spec: BatterySpec = BatterySpec(),
    imbalance_penalty: float = 50.0,
) -> BacktestResult:
    """Run ``strategy`` for delivery days in [start, end).

    At each day only prices published before gate closure are visible to the
    strategy (``known_prices``); realized prices are used for settlement.
    Strategies return either a fixed schedule or ``BidCurves``.
    """
    days = split_by_delivery_day(prices)
    pub = pd.Series({d: price_publication_time(d) for d in days})

    soc = spec.soc_init
    rows, schedules = [], []
    day = start
    while day < end:
        actual = days.get(day)
        if actual is None or actual.empty:
            day += dt.timedelta(days=1)
            continue
        visible_days = pub.index[pub < gate_closure(day)]
        cutoff = day_bounds(max(visible_days))[1] if len(visible_days) else prices.index[0]
        known = prices.loc[prices.index < cutoff]
        bid = strategy.schedule(day, actual.index, soc, known, actual_prices=actual)
        if isinstance(bid, BidCurves):
            settled, stats = settle_curves(bid, actual, spec, soc, imbalance_penalty)
        else:
            settled, stats = settle(bid, actual, spec, soc)
        soc = stats["soc_end"]
        rows.append({"date": day, **stats})
        schedules.append(settled)
        day += dt.timedelta(days=1)

    daily = pd.DataFrame(rows).set_index("date")
    return BacktestResult(strategy.name, daily, pd.concat(schedules))
