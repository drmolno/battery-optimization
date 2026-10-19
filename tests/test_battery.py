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


def test_power_rules_at_negative_prices(solver):
    # Paid to consume, every rule wastes energy, at different rates:
    # "independent" (full power in and out simultaneously, unphysical) >
    # "shared" (switching within a period, c + d <= P) >= "exclusive"
    # (one direction per period).
    p = _prices([-100.0] * 24)
    res = {r: DeterministicDispatch(SPEC, solver, power_rule=r).solve(p) for r in ("independent", "shared", "exclusive")}
    s = {r: x.schedule for r, x in res.items()}
    overlap = lambda sch: np.minimum(sch["charge_mw"], sch["discharge_mw"]).sum()
    assert overlap(s["exclusive"]) == pytest.approx(0.0, abs=1e-6)
    assert ((s["shared"]["charge_mw"] + s["shared"]["discharge_mw"]) <= SPEC.power_mw + 1e-6).all()
    assert res["independent"].profit_eur > 1.4 * res["shared"].profit_eur
    assert res["shared"].profit_eur >= res["exclusive"].profit_eur - 1e-6
    assert res["shared"].profit_eur < 1.05 * res["exclusive"].profit_eur


def test_wear_cost_makes_negative_price_cycling_unprofitable(solver):
    # Each 10 MWh round trip at -100 EUR/MWh earns about 100 EUR via losses;
    # with a wear cost of 20 EUR per discharged MWh, cycling no longer pays,
    # and the battery only fills up.
    p = _prices([-100.0] * 24)
    worn = DeterministicDispatch(BatterySpec(degradation_eur_per_mwh=20.0), solver).solve(p)
    assert worn.schedule["discharge_mw"].sum() == pytest.approx(0.0, abs=1e-6)
    assert worn.soc_end == pytest.approx(SPEC.soc_max, abs=1e-6)


def test_problem_is_reused_across_solves(solver):
    disp = DeterministicDispatch(SPEC, solver)
    disp.solve(_prices(np.arange(24)))
    disp.solve(_prices(np.arange(24)[::-1]))
    assert list(disp._problems) == [24]
