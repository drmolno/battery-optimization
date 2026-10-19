r"""Two-stage stochastic day-ahead bidding with price-quantity curves.

First stage (before gate closure): for every period :math:`t` a
non-decreasing, piecewise-linear net-sell curve (MW; negative = buy) with
quantities :math:`x_{t,1} \le \dots \le x_{t,K}` at fixed breakpoint prices
:math:`p_{t,1} < \dots < p_{t,K}`: the 5-95 % quantiles of the period's
scenario prices. In scenario :math:`s` the cleared quantity is the curve
interpolated at the scenario price,
:math:`y_{t,s} = (1 - w_{t,s}) x_{t,k} + w_{t,s} x_{t,k+1}`, which is linear
in :math:`x` because the weights are known. Beyond the outer breakpoints the
curve is flat (when submitted it extends to the market's price limits).

Few breakpoints (K = 8) and many scenarios (S = 50) keep the policy class
small relative to the sample. With one breakpoint per scenario (Fleten &
Pettersen, 2005) the curve can tailor its dispatch to each sampled future,
and the in-sample objective is optimistic.

Second stage (after clearing, per scenario): split :math:`y = d - c` with a
binary forbidding simultaneous charge/discharge, and the SoC dynamics. The
SoC must stay feasible in every scenario.

.. math::

    \max_{x, c, d, e, z}\; \frac1S \sum_{s,t} \xi_{t,s}\Delta_t\, y_{t,s}
                          - \kappa \Delta_t d_{t,s}
    \quad \text{s.t.}\quad x_{t,k} \le x_{t,k+1},\ \ x_{t,k} = x_{t,k+1}
    \text{ if } \xi_{t,(k)} = \xi_{t,(k+1)},

plus the per-scenario battery constraints of :mod:`bessopt.optim.battery`.

**Power rule.** As in :mod:`bessopt.optim.battery`, ``power_rule="shared"``
(default) uses ``c + d <= P``: the battery may switch direction within a
period, and settlement simulates the same physics. ``"exclusive"`` forbids
switching within a period and is solved as a MILP with lazy binaries.

**Lazy binaries.** Charge/discharge exclusion with S x T binaries is
expensive (50 x 96 = 4800). Overlap is only profitable where it relaxes the
SoC upper bound (dumping energy to absorb more at low/negative prices), so
the LP relaxation is solved first and binaries are added only on the
(scenario, period) pairs where overlap occurs, iterating until none remains.
Every restricted problem relaxes the full MILP, so an overlap-free optimum is
optimal for the full MILP.

The curves are submitted per period, so a realized price path outside the
scenario set can produce an SoC-infeasible dispatch; see
:func:`bessopt.backtest.runner.settle_curves`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy import sparse

from bessopt.config import BatterySpec, default_solver
from bessopt.data.timegrid import period_hours


def price_grid(scenarios: np.ndarray, n_points: int | None) -> np.ndarray:
    """Per-period curve breakpoints (T, K), strictly increasing.

    ``n_points=K`` places K breakpoints at the 5-95 % quantiles of each
    period's scenario prices. ``None`` puts one breakpoint at every scenario
    price (maximum flexibility, prone to in-sample optimism).
    """
    S, T = scenarios.shape
    if n_points is None:
        g = np.sort(scenarios, axis=0).T
    else:
        levels = np.linspace(0.05, 0.95, n_points) if n_points > 1 else np.array([0.5])
        g = np.quantile(scenarios, levels, axis=0).T
    # Enforce strict increase so interpolation is well defined.
    for k in range(1, g.shape[1]):
        g[:, k] = np.maximum(g[:, k], g[:, k - 1] + 1e-3)
    return g


def interpolation_matrix(grid: np.ndarray, scenarios: np.ndarray) -> sparse.csr_matrix:
    """Sparse A with vec(y) = A vec(x) (row-major), y[s, t] = x_t(scenario price).

    Linear interpolation between breakpoints, flat beyond the ends, matching
    ``BidCurves.dispatch``.
    """
    T, K = grid.shape
    S = scenarios.shape[0]
    rows, cols, vals = [], [], []
    for t in range(T):
        p = grid[t]
        xi = scenarios[:, t]
        k = np.clip(np.searchsorted(p, xi, side="right") - 1, 0, K - 1)
        k_hi = np.minimum(k + 1, K - 1)
        span = np.where(k_hi > k, p[k_hi] - p[k], 1.0)
        w = np.clip((xi - p[k]) / span, 0.0, 1.0) * (k_hi > k)
        r = np.arange(S) * T + t
        rows += [r, r]
        cols += [t * K + k, t * K + k_hi]
        vals += [1.0 - w, w]
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(S * T, T * K)
    )


@dataclass
class BidCurves:
    """Piecewise-linear net-sell curves, one per period (rows of the arrays)."""

    index: pd.DatetimeIndex
    prices: np.ndarray  # (T, K) non-decreasing breakpoints
    quantities: np.ndarray  # (T, K) non-decreasing net sell (MW)

    def dispatch(self, price: np.ndarray) -> np.ndarray:
        """Cleared net quantity for realized prices.

        Linear interpolation between breakpoints; flat beyond the ends.
        """
        return np.array(
            [np.interp(p, self.prices[t], self.quantities[t]) for t, p in enumerate(price)]
        )


@dataclass
class StochasticResult:
    curves: BidCurves
    expected_profit_eur: float
    status: str
    solve_seconds: float
    n_binaries: int = 0  # binaries in the final lazy round
    lazy_rounds: int = 0


class StochasticBidding:
    def __init__(
        self,
        spec: BatterySpec = BatterySpec(),
        solver: str | None = None,
        n_points: int | None = 8,
        power_rule: str = "shared",
        mip_gap: float = 1e-4,
        max_lazy_rounds: int = 10,
        verbose: bool = False,
        by_period: bool = True,
    ):
        self.n_points = n_points
        self.verbose = verbose
        self.by_period = by_period
        self.spec = spec
        self.solver = (solver or default_solver()).upper()
        if power_rule not in ("shared", "exclusive"):
            raise ValueError(power_rule)
        self.power_rule = power_rule
        self.mip_gap = mip_gap
        self.max_lazy_rounds = max_lazy_rounds

    def _solve_kwargs(self) -> dict:
        if self.solver == "GUROBI":
            return {"MIPGap": self.mip_gap, "OutputFlag": 0}
        if self.solver == "HIGHS":
            return {"mip_rel_gap": self.mip_gap}
        return {}

    def solve(
        self,
        scenarios: np.ndarray,
        index: pd.DatetimeIndex,
        soc0: float | None = None,
        soc_terminal: float | None = None,
    ) -> StochasticResult:
        """``scenarios``: (S, T) prices on the delivery grid ``index``."""
        s = self.spec
        S, T = scenarios.shape
        dt = period_hours(index)
        soc0 = s.soc_init if soc0 is None else soc0
        soc_term = s.soc_init if soc_terminal is None else soc_terminal
        soc_term = min(soc_term, s.soc_max, soc0 + s.eta_charge * s.power_mw * dt.sum())

        grid = price_grid(scenarios, self.n_points)  # (T, K)
        K = grid.shape[1]
        x = cp.Variable((T, K), name="curve")
        A = interpolation_matrix(grid, scenarios)  # vec(y) = A vec(x), row-major
        y = cp.reshape(A @ cp.vec(x, order="C"), (S, T), order="C")

        c = cp.Variable((S, T), nonneg=True)
        d = cp.Variable((S, T), nonneg=True)
        e = cp.Variable((S, T + 1))
        cons = [
            x >= -s.power_mw,
            x <= s.power_mw,
            y == d - c,
            e[:, 0] == soc0,
            e[:, 1:] == e[:, :-1] + s.eta_charge * (c @ np.diag(dt)) - (d @ np.diag(dt)) / s.eta_discharge,
            e[:, 1:] >= s.soc_min,
            e[:, 1:] <= s.soc_max,
            e[:, T] >= soc_term,
        ]
        if K > 1:
            cons.append(x[:, 1:] >= x[:, :-1])
        # c + d <= P is the convex hull of {c <= P z, d <= P (1 - z), z binary}.
        cons += [c + d <= s.power_mw]

        W = scenarios * dt[None, :] / S
        objective = cp.sum(cp.multiply(W, y)) - s.degradation_eur_per_mwh * cp.sum(d @ np.diag(dt)) / S

        t0 = time.perf_counter()
        binary = np.zeros((S, T), dtype=bool)
        for it in range(self.max_lazy_rounds + 1):
            extra = []
            if binary.any():
                si, ti = np.nonzero(binary)
                z = cp.Variable(len(si), boolean=True)
                extra = [c[si, ti] <= s.power_mw * z, d[si, ti] <= s.power_mw * (1 - z)]
            prob = cp.Problem(cp.Maximize(objective), cons + extra)
            kwargs = self._solve_kwargs() if binary.any() else {}
            prob.solve(solver=self.solver, **kwargs)
            if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise RuntimeError(f"Stochastic bidding problem {prob.status}")
            if self.power_rule == "shared":
                break
            overlap = np.minimum(c.value, d.value) > 1e-4
            if self.verbose:
                print(f"  round {it}: {binary.sum()} binaries, overlap at {overlap.sum()}, "
                      f"obj {prob.value:.1f}, {time.perf_counter() - t0:.1f}s", flush=True)
            if not (overlap & ~binary).any():
                break
            # Overlap tends to migrate between scenarios of the same period,
            # so add binaries for every scenario of an affected period.
            binary |= overlap.any(axis=0)[None, :] if self.by_period else overlap
        else:
            raise RuntimeError("Lazy binary loop did not converge.")
        elapsed = time.perf_counter() - t0

        curves = BidCurves(index=index, prices=grid, quantities=np.asarray(x.value))
        return StochasticResult(curves, float(prob.value), prob.status, elapsed, int(binary.sum()), it)
