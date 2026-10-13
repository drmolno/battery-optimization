r"""LASSO-Estimated AutoRegressive (LEAR) day-ahead price model.

One linear model per quarter-hour slot :math:`h`,

.. math::  \tilde y_{d,h} = \beta_h^\top \tilde x_d + \varepsilon_{d,h},

with the L1 penalty chosen per slot by an information criterion along the
LARS path (Lago et al., 2021). Prices and continuous features are variance-
stabilized with the area hyperbolic sine after median/MAD normalization
(Uniejewski et al., 2018):

.. math::  \tilde z = \operatorname{asinh}\big((z - \operatorname{med}(z)) / \operatorname{MAD}(z)\big).

Normalization statistics are computed on the training window only.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoLarsIC

from bessopt.forecast.features import is_binary

MAD_TO_STD = 1.4826  # MAD of a standard normal is 1/1.4826


class AsinhScaler:
    """Column-wise median/MAD normalization followed by asinh."""

    def fit(self, Z: np.ndarray) -> "AsinhScaler":
        self.med = np.median(Z, axis=0)
        mad = MAD_TO_STD * np.median(np.abs(Z - self.med), axis=0)
        self.mad = np.where(mad > 1e-9, mad, 1.0)
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        return np.arcsinh((Z - self.med) / self.mad)

    def inverse(self, Zt: np.ndarray) -> np.ndarray:
        return np.sinh(Zt) * self.mad + self.med


def _fit_one(X: np.ndarray, y: np.ndarray, criterion: str) -> tuple[np.ndarray, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = LassoLarsIC(criterion=criterion, fit_intercept=True).fit(X, y)
    return m.coef_.copy(), float(m.intercept_)


class LEAR:
    def __init__(self, criterion: str = "aic", n_jobs: int = -1):
        self.criterion = criterion
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "LEAR":
        # LARS breaks on exactly tied columns. In training windows within the
        # hourly era, the four D-1 quarter-hour lags of each hour are identical
        # and regime_qh is constant, so drop constant and duplicate columns.
        varying = X.columns[X.std() > 0]
        X = X[varying].loc[:, ~X[varying].T.duplicated()]
        self.columns = X.columns
        self.slots = Y.columns
        self._binary = is_binary(X.columns)
        Xv = X.to_numpy(dtype=float)
        self.x_scaler = AsinhScaler().fit(Xv[:, ~self._binary])
        Xt = self._transform_x(Xv)
        self.y_scaler = AsinhScaler().fit(Y.to_numpy(dtype=float))
        Yt = self.y_scaler.transform(Y.to_numpy(dtype=float))

        res = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(Xt, Yt[:, j], self.criterion) for j in range(Yt.shape[1])
        )
        self.coef_ = np.column_stack([r[0] for r in res])  # (features, slots)
        self.intercept_ = np.array([r[1] for r in res])
        return self

    def _transform_x(self, Xv: np.ndarray) -> np.ndarray:
        Xt = Xv.copy()
        Xt[:, ~self._binary] = self.x_scaler.transform(Xv[:, ~self._binary])
        return Xt

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = self._transform_x(X[self.columns].to_numpy(dtype=float))
        Yt = Xt @ self.coef_ + self.intercept_
        return pd.DataFrame(self.y_scaler.inverse(Yt), index=X.index, columns=self.slots)

    def coefficients(self, all_columns: pd.Index | None = None) -> pd.DataFrame:
        """Fitted coefficients in transformed space, (features x slots).

        Columns dropped as constant/duplicate in the training window get 0.
        """
        c = pd.DataFrame(self.coef_, index=self.columns, columns=self.slots)
        return c if all_columns is None else c.reindex(all_columns, fill_value=0.0)


def rolling_forecast(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    start,
    end,
    window_days: int = 730,
    refit_every: int = 1,
    model_factory=LEAR,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Out-of-sample forecasts for delivery days in [start, end).

    For day D the model is trained on days in [D - window_days, D - 1] whose
    prices are fully known. D-1 prices are published at ~12:45 on D-2, so
    they are known at gate closure for D; D's are not. Returns the forecast
    table (day x slot) and the coefficient tables of each refit.
    """
    days = X.index[(X.index >= pd.Timestamp(start)) & (X.index < pd.Timestamp(end))]
    Yk = Y.dropna()
    preds, coefs = [], []
    model = None
    t0 = time.perf_counter()
    for i, d in enumerate(days):
        if model is None or i % refit_every == 0:
            train = Yk.index[(Yk.index < d) & (Yk.index >= d - pd.Timedelta(days=window_days))]
            train = train.intersection(X.index)
            model = model_factory().fit(X.loc[train], Yk.loc[train])
            coefs.append(model.coefficients(X.columns).assign(fit_date=d))
        preds.append(model.predict(X.loc[[d]]))
        if verbose and (i + 1) % 30 == 0:
            el = time.perf_counter() - t0
            print(f"  {d:%Y-%m-%d}  {i + 1}/{len(days)} days  {el / (i + 1):.1f} s/day", flush=True)
    return pd.concat(preds), coefs
