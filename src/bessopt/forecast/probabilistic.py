r"""Probabilistic day-ahead price forecasts and joint price scenarios.

Marginals (per quarter-hour slot), from out-of-sample point forecasts:

* **QRA**, Quantile Regression Averaging (Nowotarski & Weron, 2015): for each
  slot and quantile level :math:`\tau`, a linear quantile regression of the
  price on the LEAR and naive forecasts, fit on a rolling calibration window.
* **ER**, empirical residuals: the LEAR forecast plus empirical quantiles of
  its recent errors.

Both work in the asinh-transformed space of the calibration window. Quantiles
are equivariant under monotone transforms, so they map back exactly; the
transform makes errors closer to homoscedastic.

Joint scenarios use a Gaussian copula. Its correlation matrix is estimated
from rank-based normal scores of recent LEAR errors (semiparametric, so no
PIT of the calibration days is needed), with Ledoit-Wolf shrinkage.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm, rankdata
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import QuantileRegressor

from bessopt.forecast.lear import AsinhScaler

TAUS = np.round(
    np.concatenate([[0.01, 0.025], np.arange(0.05, 0.951, 0.05), [0.975, 0.99]]), 3
)


def naive_table(Y: pd.DataFrame) -> pd.DataFrame:
    """Day-ahead persistence forecast: yesterday's 96 slots."""
    N = Y.copy()
    N.index = N.index + pd.Timedelta(days=1)
    return N


def _calibration_days(F: pd.DataFrame, Y: pd.DataFrame, day: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """The ``n`` most recent days before ``day`` with forecast and outcome."""
    ok = F.dropna().index.intersection(Y.dropna().index)
    return ok[ok < day][-n:]


def _qra_slot(y: np.ndarray, X: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Fit quantile regressions for one slot; returns (taus, 1 + n_regressors)."""
    out = np.empty((len(taus), X.shape[1] + 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, t in enumerate(taus):
            m = QuantileRegressor(quantile=t, alpha=0.0, solver="highs").fit(X, y)
            out[k] = np.r_[m.intercept_, m.coef_]
    return out


def _to_long(blocks: list[tuple[pd.Timestamp, np.ndarray]], slots: pd.Index) -> pd.DataFrame:
    """[(day, (taus x slots) array)] -> DataFrame indexed by (date, tau)."""
    idx = pd.MultiIndex.from_product([[d for d, _ in blocks], TAUS], names=["date", "tau"])
    return pd.DataFrame(np.vstack([q for _, q in blocks]), index=idx, columns=slots)


def rolling_quantiles(
    F: pd.DataFrame,
    Y: pd.DataFrame,
    start,
    end,
    method: str = "qra",
    calib_days: int = 182,
    refit_every: int = 7,
    n_jobs: int = -1,
    pool_slots: int = 0,
    center: bool = False,
) -> pd.DataFrame:
    """Out-of-sample quantile forecasts for days in [start, end).

    ``F`` holds out-of-sample LEAR point forecasts (day x slot), ``Y`` the
    realized prices. For day D, calibration uses only days < D, whose prices
    are known at gate closure.

    For ``method="er"``: ``pool_slots`` pools residuals of +-k neighbouring
    slots; ``center`` shifts residual quantiles to median zero, so the
    predictive median equals the LEAR forecast.
    """
    N = naive_table(Y)
    days = F.index[(F.index >= pd.Timestamp(start)) & (F.index < pd.Timestamp(end))]
    slots = F.columns
    blocks = []
    model = None
    for i, d in enumerate(days):
        if model is None or i % refit_every == 0:
            cal = _calibration_days(F, Y, d, calib_days)
            if len(cal) < calib_days // 2:
                raise ValueError(f"Only {len(cal)} calibration days before {d:%Y-%m-%d}")
            scalers = [AsinhScaler().fit(Y.loc[cal, s].to_numpy()[:, None]) for s in slots]
            tf = lambda j, v, scalers=scalers: scalers[j].transform(np.asarray(v, float)[:, None])[:, 0]
            if method == "qra":
                coefs = Parallel(n_jobs=n_jobs)(
                    delayed(_qra_slot)(
                        tf(j, Y.loc[cal, s]),
                        np.column_stack([tf(j, F.loc[cal, s]), tf(j, N.loc[cal, s])]),
                        TAUS,
                    )
                    for j, s in enumerate(slots)
                )
                model = ("qra", scalers, coefs)
            elif method == "er":
                E = np.column_stack([tf(j, Y.loc[cal, s]) - tf(j, F.loc[cal, s]) for j, s in enumerate(slots)])
                resid = []
                for j in range(len(slots)):
                    # Pool errors of neighbouring slots: per-slot estimates
                    # vary from one quarter-hour to the next, which creates
                    # spurious intra-hour spreads in the forecast distribution.
                    lo, hi = max(0, j - pool_slots), min(len(slots), j + pool_slots + 1)
                    r = np.quantile(E[:, lo:hi].ravel(), TAUS)
                    resid.append(r - np.interp(0.5, TAUS, r) if center else r)
                model = ("er", scalers, resid)
            else:
                raise ValueError(method)

        _, scalers, params = model
        q = np.empty((len(TAUS), len(slots)))
        for j, s in enumerate(slots):
            f = scalers[j].transform(np.array([[F.at[d, s]]]))[0, 0]
            if method == "qra":
                n = scalers[j].transform(np.array([[N.at[d, s]]]))[0, 0]
                qt = params[j] @ np.array([1.0, f, n])
            else:
                qt = f + params[j]
            q[:, j] = scalers[j].inverse(np.sort(qt)[:, None])[:, 0]
        blocks.append((d, q))
    return _to_long(blocks, slots)


# --- quantile functions ---------------------------------------------------


def quantile_function(q: np.ndarray, u: np.ndarray, taus: np.ndarray = TAUS) -> np.ndarray:
    """Evaluate piecewise-linear quantile functions.

    ``q``: (taus, slots) quantiles; ``u``: (..., slots) levels in (0, 1).
    Beyond the outermost levels the end segments are extended linearly.
    """
    out = np.empty(u.shape)
    lo_slope = (q[1] - q[0]) / (taus[1] - taus[0])
    hi_slope = (q[-1] - q[-2]) / (taus[-1] - taus[-2])
    for j in range(q.shape[1]):
        uj = u[..., j]
        v = np.interp(uj, taus, q[:, j])
        v = np.where(uj < taus[0], q[0, j] - lo_slope[j] * (taus[0] - uj), v)
        v = np.where(uj > taus[-1], q[-1, j] + hi_slope[j] * (uj - taus[-1]), v)
        out[..., j] = v
    return out


def distribution_mean(q: np.ndarray, taus: np.ndarray = TAUS, n: int = 2000) -> np.ndarray:
    """Mean of each slot's predictive distribution, E[X] = int_0^1 Q(u) du.

    Midpoint rule on a fine grid of levels; tails use the linear extension
    of :func:`quantile_function`.
    """
    u = (np.arange(n) + 0.5) / n
    return quantile_function(q, np.repeat(u[:, None], q.shape[1], axis=1), taus).mean(axis=0)


# --- copula ---------------------------------------------------------------


def copula_correlation(F: pd.DataFrame, Y: pd.DataFrame, day: pd.Timestamp, calib_days: int = 182) -> np.ndarray:
    """Correlation of rank-normal scores of recent LEAR errors (asinh space)."""
    cal = _calibration_days(F, Y, day, calib_days)
    E = np.empty((len(cal), F.shape[1]))
    for j, s in enumerate(F.columns):
        sc = AsinhScaler().fit(Y.loc[cal, s].to_numpy()[:, None])
        e = sc.transform(Y.loc[cal, s].to_numpy()[:, None]) - sc.transform(F.loc[cal, s].to_numpy()[:, None])
        E[:, j] = norm.ppf(rankdata(e[:, 0]) / (len(cal) + 1))
    C = LedoitWolf().fit(E).covariance_
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def sample_scenarios(
    q: np.ndarray,
    n: int,
    corr: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw ``n`` joint price scenarios (n x slots).

    ``corr=None`` samples slots independently (for comparison only).
    """
    rng = rng or np.random.default_rng()
    k = q.shape[1]
    if corr is None:
        z = rng.standard_normal((n, k))
    else:
        w, V = np.linalg.eigh(corr)
        L = V * np.sqrt(np.clip(w, 0, None))
        z = rng.standard_normal((n, k)) @ L.T
    return quantile_function(q, norm.cdf(z))


# --- scoring --------------------------------------------------------------


def pinball(q: np.ndarray, y: np.ndarray, taus: np.ndarray = TAUS) -> np.ndarray:
    """Pinball loss per quantile level, averaged over slots. q: (taus, slots)."""
    diff = y[None, :] - q
    return np.mean(np.maximum(taus[:, None] * diff, (taus[:, None] - 1) * diff), axis=1)


def energy_score(X: np.ndarray, y: np.ndarray) -> float:
    """Energy score of scenarios X (n x d) for outcome y (d,). Lower is better."""
    t1 = np.linalg.norm(X - y, axis=1).mean()
    n = len(X)
    diffs = X[:, None, :] - X[None, :, :]
    t2 = np.linalg.norm(diffs, axis=2).sum() / (n * (n - 1))
    return float(t1 - 0.5 * t2)


def variogram_score(X: np.ndarray, y: np.ndarray, p: float = 0.5) -> float:
    """Variogram score of order p (Scheuerer & Hamill, 2015), unit weights.

    Sensitive to misspecified dependence between slots, which is what
    matters for storage arbitrage.
    """
    vy = np.abs(y[:, None] - y[None, :]) ** p
    vx = (np.abs(X[:, :, None] - X[:, None, :]) ** p).mean(axis=0)
    return float(((vy - vx) ** 2).sum() / len(y) ** 2)
