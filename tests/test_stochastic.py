import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from bessopt.backtest.runner import settle_curves
from bessopt.config import MARKET_TZ, BatterySpec
from bessopt.optim.battery import DeterministicDispatch
from bessopt.optim.stochastic import BidCurves, StochasticBidding, interpolation_matrix, price_grid

SPEC = BatterySpec(degradation_eur_per_mwh=2.0)
SOLVERS = [s for s in ("GUROBI", "HIGHS") if s in cp.installed_solvers()]


def _grid(T=24):
    return pd.date_range(pd.Timestamp("2025-06-01").tz_localize(MARKET_TZ), periods=T, freq="h")


def _scenarios(S, T=24, seed=0):
    rng = np.random.default_rng(seed)
    h = np.arange(T)
    base = 60 + 40 * np.sin(2 * np.pi * (h - 8) / T)
    return base[None, :] + rng.normal(0, 25, (S, T)) + rng.normal(0, 15, (S, 1))


@pytest.mark.parametrize("solver", SOLVERS)
def test_single_scenario_matches_deterministic(solver):
    X = _scenarios(1)
    grid = _grid()
    stoch = StochasticBidding(SPEC, solver).solve(X, grid)
    det = DeterministicDispatch(SPEC, solver).solve(pd.Series(X[0], index=grid))
    assert stoch.expected_profit_eur == pytest.approx(det.profit_eur, rel=1e-4, abs=1e-2)


@pytest.mark.parametrize("solver", SOLVERS)
def test_curves_are_monotone_and_feasible_in_every_scenario(solver):
    X = _scenarios(20)
    grid = _grid()
    res = StochasticBidding(SPEC, solver, power_rule="exclusive").solve(X, grid)
    assert (np.diff(res.curves.quantities, axis=1) >= -1e-6).all()
    for s in range(len(X)):
        _, stats = settle_curves(res.curves, pd.Series(X[s], index=grid), SPEC, SPEC.soc_init)
        assert stats["imbalance_mwh"] < 1e-4


@pytest.mark.parametrize("solver", SOLVERS)
def test_in_sample_value_of_stochastic_solution_is_nonnegative(solver):
    # A fixed schedule is a flat curve, hence feasible for the stochastic
    # program, so the stochastic optimum is at least the expected profit of
    # the expected-value schedule over the same scenarios.
    X = _scenarios(20, seed=3)
    grid = _grid()
    dt = np.ones(24)
    res = StochasticBidding(SPEC, solver).solve(X, grid)
    ev = DeterministicDispatch(SPEC, solver).solve(pd.Series(X.mean(axis=0), index=grid)).schedule
    y = (ev["discharge_mw"] - ev["charge_mw"]).to_numpy()
    eev = (X * dt) @ y - SPEC.degradation_eur_per_mwh * dt @ ev["discharge_mw"].to_numpy()
    assert res.expected_profit_eur >= eev.mean() - 1e-3


def test_settlement_charges_imbalance_for_infeasible_curves():
    grid = _grid(4)
    # Sell full power for 4 hours regardless of price: 40 MWh > stored energy.
    curves = BidCurves(grid, prices=np.tile([0.0, 1000.0], (4, 1)), quantities=np.full((4, 2), SPEC.power_mw))
    prices = pd.Series(100.0, index=grid)
    out, stats = settle_curves(curves, prices, SPEC, SPEC.soc_init, imbalance_penalty=50.0)
    delivered = (out["discharge_mw"] - out["charge_mw"]).sum()
    assert out["soc_mwh"].min() >= SPEC.soc_min - 1e-9
    assert stats["imbalance_mwh"] == pytest.approx(40.0 - delivered)
    assert stats["profit_eur"] == pytest.approx(
        100.0 * delivered - 50.0 * stats["imbalance_mwh"] - SPEC.degradation_eur_per_mwh * delivered
    )


def test_interpolation_matrix_matches_settlement_dispatch():
    rng = np.random.default_rng(5)
    X = _scenarios(30, seed=5)
    grid = price_grid(X, 6)
    assert (np.diff(grid, axis=1) > 0).all()
    q = np.sort(rng.uniform(-10, 10, grid.shape), axis=1)
    A = interpolation_matrix(grid, X)
    y = (A @ q.ravel()).reshape(X.shape)
    curves = BidCurves(_grid(), grid, q)
    for s in range(len(X)):
        np.testing.assert_allclose(y[s], curves.dispatch(X[s]), atol=1e-9)


def test_curve_dispatch_interpolates_and_is_flat_beyond_ends():
    grid = _grid(1)
    c = BidCurves(grid, prices=np.array([[10.0, 20.0]]), quantities=np.array([[-10.0, 10.0]]))
    np.testing.assert_allclose(c.dispatch(np.array([15.0])), [0.0])
    np.testing.assert_allclose(c.dispatch(np.array([-50.0])), [-10.0])
    np.testing.assert_allclose(c.dispatch(np.array([500.0])), [10.0])
