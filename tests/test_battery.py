import numpy as np
import pandas as pd
import pytest

from bessopt.config import MARKET_TZ, BatterySpec
from bessopt.optim.battery import DeterministicDispatch

SPEC = BatterySpec(degradation_eur_per_mwh=0.0)


def _prices(values, freq="h", start="2025-06-01"):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz=MARKET_TZ)
    return pd.Series(np.asarray(values, dtype=float), index=idx)


@pytest.fixture(params=["HIGHS", "GUROBI"])
def solver(request):
    import cvxpy as cp

    if request.param not in cp.installed_solvers():
        pytest.skip(f"{request.param} not installed")
    return request.param


def test_flat_prices_do_nothing(solver):
    res = DeterministicDispatch(SPEC, solver).solve(_prices([50.0] * 24))
    assert res.profit_eur == pytest.approx(0.0, abs=1e-6)


def test_two_level_arbitrage_matches_hand_calculation(solver):
    # Free energy for 12 h, then 100 EUR/MWh for 12 h. The optimum fills the
    # battery (SoC 10 -> 19 MWh), then discharges back to the terminal
    # requirement of 10 MWh, delivering 9 * eta_d MWh to the grid.
    p = _prices([0.0] * 12 + [100.0] * 12)
    res = DeterministicDispatch(SPEC, solver).solve(p)
    sold = 19.0 - 10.0
    expected = 100.0 * sold * SPEC.eta_discharge
    assert res.profit_eur == pytest.approx(expected, rel=1e-5)
    assert res.soc_end == pytest.approx(SPEC.soc_init, abs=1e-5)


def test_soc_limits_and_power_limits_hold(solver):
    rng = np.random.default_rng(0)
    p = _prices(rng.normal(60, 40, 96), freq="15min", start="2025-11-01")
    res = DeterministicDispatch(SPEC, solver).solve(p)
    s = res.schedule
    assert s["soc_mwh"].min() >= SPEC.soc_min - 1e-6
    assert s["soc_mwh"].max() <= SPEC.soc_max + 1e-6
    assert s[["charge_mw", "discharge_mw"]].max().max() <= SPEC.power_mw + 1e-6


def test_negative_prices_relaxation_burns_energy_milp_does_not(solver):
    # With strongly negative prices, being paid to consume makes simultaneous
    # charge+discharge profitable in the LP relaxation (round-trip losses
    # "consume" energy). The MILP forbids it.
    p = _prices([-100.0] * 24)
    lp = DeterministicDispatch(SPEC, solver, relax_binaries=True).solve(p)
    milp = DeterministicDispatch(SPEC, solver).solve(p)
    overlap = lambda r: np.minimum(r.schedule["charge_mw"], r.schedule["discharge_mw"]).sum()
    assert overlap(lp) > 1.0
    assert overlap(milp) == pytest.approx(0.0, abs=1e-6)
    assert lp.profit_eur > milp.profit_eur


def test_problem_is_reused_across_solves(solver):
    disp = DeterministicDispatch(SPEC, solver)
    disp.solve(_prices(np.arange(24)))
    disp.solve(_prices(np.arange(24)[::-1]))
    assert list(disp._problems) == [24]
