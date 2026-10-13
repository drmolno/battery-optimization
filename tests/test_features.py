import numpy as np
import pandas as pd
import pytest

from bessopt.config import MARKET_TZ
from bessopt.forecast.features import QH_SLOTS, build_dataset, daily_matrix, slots_to_grid
from bessopt.forecast.lear import rolling_forecast


def _series(start, end, freq, seed):
    idx = pd.date_range(pd.Timestamp(start).tz_localize(MARKET_TZ), pd.Timestamp(end).tz_localize(MARKET_TZ), freq=freq, inclusive="left")
    rng = np.random.default_rng(seed)
    hour = idx.hour + idx.minute / 60
    return pd.Series(50 + 30 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 5, len(idx)), index=idx)


def _inputs(prices):
    s, e = "2025-08-01", "2025-11-15"
    return dict(
        prices=prices,
        neighbours={"DE": _series(s, e, "h", 1)},
        load_fc=_series(s, e, "h", 2) * 40,
        wind_fc=_series(s, e, "h", 3) * 10,
    )


@pytest.fixture
def prices():
    hourly = _series("2025-08-01", "2025-10-01", "h", 0)
    quarter = _series("2025-10-01", "2025-11-14", "15min", 0)
    return pd.concat([hourly, quarter])


def test_features_of_day_d_do_not_depend_on_prices_of_day_d(prices):
    X, _ = build_dataset(**_inputs(prices))
    d = pd.Timestamp("2025-10-20")
    bumped = prices.copy()
    bumped[bumped.index.date == d.date()] += 1000.0
    X2, Y2 = build_dataset(**_inputs(bumped))
    pd.testing.assert_series_equal(X.loc[d], X2.loc[d])
    assert (X2.loc[d + pd.Timedelta(days=1)] != X.loc[d + pd.Timedelta(days=1)]).any()
    assert Y2.loc[d].mean() > 500


def test_daily_matrix_handles_dst_and_hourly_era(prices):
    Y = daily_matrix(prices, "15min")
    assert list(Y.columns) == QH_SLOTS
    assert not Y.isna().any().any()
    # Fall-back day 2025-10-26 has 100 periods; averaged into 96 slots.
    assert Y.loc["2025-10-26"].shape == (96,)
    # Hourly era: four identical quarter-hours per hour.
    row = Y.loc["2025-09-15"].to_numpy().reshape(24, 4)
    assert np.allclose(row, row[:, :1])


def test_slots_to_grid_averages_for_hourly_grid():
    vals = pd.Series(np.arange(96, dtype=float), index=QH_SLOTS)
    grid = pd.date_range(pd.Timestamp("2025-06-01").tz_localize(MARKET_TZ), periods=24, freq="h")
    out = slots_to_grid(vals, grid)
    assert out.iloc[0] == pytest.approx(1.5)
    grid100 = pd.date_range(pd.Timestamp("2025-10-26").tz_localize(MARKET_TZ), pd.Timestamp("2025-10-27").tz_localize(MARKET_TZ), freq="15min", inclusive="left")
    assert len(slots_to_grid(vals, grid100)) == 100


def test_rolling_forecast_trains_only_on_past(prices):
    X, Y = build_dataset(**_inputs(prices))
    seen = []

    class Spy:
        """Climatology model that records the last training day."""

        def fit(self, X_, Y_):
            seen.append(Y_.index.max())
            self.mean = Y_.mean()
            return self

        def predict(self, X_):
            return pd.DataFrame([self.mean] * len(X_), index=X_.index)

        def coefficients(self, all_columns=None):
            return pd.DataFrame({"c": [0.0]})

    fc, _ = rolling_forecast(X, Y, "2025-11-01", "2025-11-08", window_days=60, refit_every=3,
                             model_factory=Spy, verbose=False)
    assert len(fc) == 7
    fit_days = pd.date_range("2025-11-01", periods=7)[::3]
    assert all(last < d for last, d in zip(seen, fit_days))
