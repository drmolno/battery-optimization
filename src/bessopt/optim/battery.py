r"""Deterministic battery dispatch against a known price vector.

For periods :math:`t = 0..T-1` with length :math:`\Delta_t` (hours) and price
:math:`\lambda_t` (EUR/MWh), solve

.. math::

    \max_{c, d, e}\; & \sum_t \lambda_t \Delta_t (d_t - c_t)
                      - \kappa \sum_t \Delta_t d_t \\
    \text{s.t.}\;\;  & e_{t+1} = e_t + \Delta_t(\eta_c c_t - d_t/\eta_d),
                       \quad e_0 = \bar e_0 \\
                     & \underline e \le e_t \le \overline e,
                       \quad e_T \ge e_{\text{term}} \\
                     & c_t, d_t \ge 0,\quad c_t + d_t \le P

where :math:`c, d` are period-average charge/discharge power (MW), :math:`e`
is stored energy (MWh) and :math:`\kappa` a throughput degradation cost.

**Power rule.** The inverter carries power one way at a time but can switch
direction within a period, so the battery may charge for part of a period and
discharge for the rest. Averaged over the period this is exactly
:math:`c + d \le P` (``power_rule="shared"``, the default). It allows
intra-period cycling, which wastes energy through real round-trip losses and
pays only at negative prices; the degradation cost decides whether it is
worth it. Alternatives, for comparison:

* ``"exclusive"``: one direction per period (binary variable); slightly
  stricter than physics.
* ``"independent"``: :math:`c \le P,\ d \le P` separately. Unphysical (full
  power in and out simultaneously), so the model "burns" energy at negative
  prices faster than a real battery can.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from bessopt.config import BatterySpec, default_solver
from bessopt.data.timegrid import period_hours


@dataclass
class DispatchResult:
    schedule: pd.DataFrame  # columns: price, charge_mw, discharge_mw, soc_mwh (end of period)
    revenue_eur: float
    degradation_eur: float
    status: str
    solve_seconds: float

    @property
    def profit_eur(self) -> float:
        return self.revenue_eur - self.degradation_eur

    @property
    def soc_end(self) -> float:
        return float(self.schedule["soc_mwh"].iloc[-1])


class DeterministicDispatch:
    """Optimal dispatch for a known (or assumed) price vector.

    Problems are compiled once per horizon length ``T`` (92/96/100 or 23/24/25)
    and re-solved with new parameter values, which matters in long backtests.
    """

    def __init__(
        self,
        spec: BatterySpec = BatterySpec(),
        solver: str | None = None,
        power_rule: str = "shared",
    ):
        if power_rule not in ("shared", "exclusive", "independent"):
            raise ValueError(power_rule)
        self.spec = spec
        self.solver = (solver or default_solver()).upper()
        self.power_rule = power_rule
        self._problems: dict[int, tuple[cp.Problem, dict]] = {}

    def _build(self, T: int) -> tuple[cp.Problem, dict]:
        s = self.spec
        price_dt = cp.Parameter(T, name="price_dt")  # lambda_t * Delta_t
        dt = cp.Parameter(T, nonneg=True, name="dt")
        soc0 = cp.Parameter(nonneg=True, name="soc0")
        soc_term = cp.Parameter(nonneg=True, name="soc_term")

        c = cp.Variable(T, nonneg=True, name="charge")
        d = cp.Variable(T, nonneg=True, name="discharge")
        e = cp.Variable(T + 1, name="soc")

        cons = [
            e[0] == soc0,
            e[1:] == e[:-1] + s.eta_charge * cp.multiply(dt, c) - cp.multiply(dt, d) / s.eta_discharge,
            e[1:] >= s.soc_min,
            e[1:] <= s.soc_max,
            e[T] >= soc_term,
        ]
        if self.power_rule == "shared":
            cons += [c + d <= s.power_mw]
        elif self.power_rule == "independent":
            cons += [c <= s.power_mw, d <= s.power_mw]
        else:
            z = cp.Variable(T, boolean=True, name="is_charging")
            cons += [c <= s.power_mw * z, d <= s.power_mw * (1 - z)]

        revenue = price_dt @ (d - c)
        degradation = s.degradation_eur_per_mwh * (dt @ d)
        prob = cp.Problem(cp.Maximize(revenue - degradation), cons)
        handles = dict(price_dt=price_dt, dt=dt, soc0=soc0, soc_term=soc_term, c=c, d=d, e=e)
        return prob, handles

    def solve(
        self,
        prices: pd.Series,
        soc0: float | None = None,
        soc_terminal: float | None = None,
    ) -> DispatchResult:
        """Optimize over the horizon spanned by ``prices`` (EUR/MWh).

        ``soc_terminal`` defaults to the initial SoC of the spec, i.e. the
        battery must end each horizon at least as full as a fresh start. This
        prevents the optimizer from selling the stored energy "for free" at
        the end of every day.
        """
        s = self.spec
        T = len(prices)
        if T not in self._problems:
            self._problems[T] = self._build(T)
        prob, h = self._problems[T]

        dt = period_hours(prices.index)
        soc0 = s.soc_init if soc0 is None else soc0
        soc_terminal = s.soc_init if soc_terminal is None else soc_terminal
        h["price_dt"].value = prices.to_numpy(dtype=float) * dt
        h["dt"].value = dt
        h["soc0"].value = float(np.clip(soc0, 0.0, s.energy_mwh))
        # Never ask for more than the battery can physically reach.
        reachable = soc0 + s.eta_charge * s.power_mw * dt.sum()
        h["soc_term"].value = float(min(soc_terminal, s.soc_max, reachable))

        t0 = time.perf_counter()
        prob.solve(solver=self.solver)
        elapsed = time.perf_counter() - t0
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Dispatch problem {prob.status}")

        c = np.maximum(h["c"].value, 0.0)
        d = np.maximum(h["d"].value, 0.0)
        schedule = pd.DataFrame(
            {"price": prices.to_numpy(), "charge_mw": c, "discharge_mw": d, "soc_mwh": h["e"].value[1:]},
            index=prices.index,
        )
        return DispatchResult(
            schedule=schedule,
            revenue_eur=float(prices.to_numpy() * dt @ (d - c)),
            degradation_eur=float(s.degradation_eur_per_mwh * dt @ d),
            status=prob.status,
            solve_seconds=elapsed,
        )
