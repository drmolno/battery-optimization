import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from bessopt.forecast.probabilistic import (
    TAUS,
    copula_correlation,
    energy_score,
    pinball,
    quantile_function,
    rolling_quantiles,
    sample_scenarios,
)


def _synthetic(days=300, slots=8, seed=0):
    """Point forecast F and outcomes Y = F + correlated noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=days, freq="D")
    cols = [f"s{j}" for j in range(slots)]
    F = pd.DataFrame(50 + 20 * rng.standard_normal((days, slots)), index=idx, columns=cols)
    common = rng.standard_normal((days, 1))
    noise = 10 * (0.8 * common + 0.6 * rng.standard_normal((days, slots)))
    return F, F + noise


def test_quantile_function_is_monotone_and_interpolates():
    q = np.column_stack([norm.ppf(TAUS), 10 + 2 * norm.ppf(TAUS)])
    u = np.linspace(0.001, 0.999, 500)[:, None].repeat(2, axis=1)
    v = quantile_function(q, u)
    assert (np.diff(v, axis=0) >= 0).all()
    np.testing.assert_allclose(quantile_function(q, np.full((1, 2), 0.5))[0], [0.0, 10.0], atol=1e-9)


def test_scenarios_reproduce_marginals_and_correlation():
    q = np.column_stack([norm.ppf(TAUS)] * 3)
    corr = np.array([[1, 0.9, 0], [0.9, 1, 0], [0, 0, 1.0]])
    X = sample_scenarios(q, 20000, corr, np.random.default_rng(0))
    np.testing.assert_allclose(np.quantile(X, [0.1, 0.5, 0.9], axis=0), q[[TAUS.tolist().index(t) for t in (0.1, 0.5, 0.9)]], atol=0.05)
    c = np.corrcoef(X.T)
    assert c[0, 1] == pytest.approx(0.9, abs=0.03) and abs(c[0, 2]) < 0.03


@pytest.mark.parametrize("method", ["qra", "er"])
def test_rolling_quantiles_are_calibrated_and_out_of_sample(method):
    F, Y = _synthetic(slots=4)
    kw = dict(method=method, calib_days=150, refit_every=7, n_jobs=1)
    Q = rolling_quantiles(F, Y, "2025-08-01", "2025-09-15", **kw)
    days = Q.index.get_level_values("date").unique()
    assert len(days) == 45
    lo = Q.xs(0.05, level="tau").to_numpy()
    hi = Q.xs(0.95, level="tau").to_numpy()
    y = Y.loc[days].to_numpy()
    assert 0.83 < ((y >= lo) & (y <= hi)).mean() < 0.97
    # Altering day D's outcome must leave all forecasts up to D unchanged.
    Y2 = Y.copy()
    Y2.loc["2025-08-20"] += 1e3
    Q2 = rolling_quantiles(F, Y2, "2025-08-01", "2025-09-15", **kw)
    upto = Q.index.get_level_values("date") <= "2025-08-20"
    pd.testing.assert_frame_equal(Q[upto], Q2[upto])
    assert not np.allclose(Q[~upto].to_numpy(), Q2[~upto].to_numpy())


def test_copula_correlation_recovers_common_factor():
    F, Y = _synthetic()
    R = copula_correlation(F, Y, pd.Timestamp("2025-10-01"))
    off = R[~np.eye(len(R), dtype=bool)]
    assert 0.45 < off.mean() < 0.8  # true correlation 0.64, shrunk towards 0


def test_scores_prefer_correct_dependence():
    rng = np.random.default_rng(1)
    q = np.column_stack([norm.ppf(TAUS)] * 24)
    corr = np.full((24, 24), 0.9) + 0.1 * np.eye(24)
    good, bad = [], []
    for _ in range(30):
        y = sample_scenarios(q, 1, corr, rng)[0]
        good.append(energy_score(sample_scenarios(q, 200, corr, rng), y))
        bad.append(energy_score(sample_scenarios(q, 200, None, rng), y))
    assert np.mean(good) < np.mean(bad)
    assert pinball(q, np.zeros(24)).shape == (len(TAUS),)
