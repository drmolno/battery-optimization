"""ENTSO-E Transparency Platform data (requires ENTSOE_API_KEY).

All functions return data indexed by tz-aware delivery-period start in market
time and cache it per month under ``data/cache/entsoe/``.

Availability at gate closure (12:00 D-1), which the forecaster must respect:

* day-ahead prices for D-1 and earlier (published ~12:45 on D-2);
* TSO day-ahead load forecast and wind/solar forecasts for D (published
  before gate closure by regulation, typically in the morning of D-1);
* prices of neighbouring zones follow the same rule as SE4 prices.
"""

from __future__ import annotations

from functools import cache

import pandas as pd

from bessopt.config import ENTSOE_ZONE, MARKET_TZ, entsoe_api_key
from bessopt.data.cache import cached_monthly

# Zones coupled to SE4, useful as price features.
NEIGHBOURS = {"SE3": "SE_3", "DK2": "DK_2", "DE_LU": "DE_LU", "PL": "PL", "LT": "LT"}


@cache
def _client():
    from entsoe import EntsoePandasClient

    return EntsoePandasClient(api_key=entsoe_api_key())


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize(MARKET_TZ) if t.tzinfo is None else t


def day_ahead_prices(start, end, zone: str = ENTSOE_ZONE) -> pd.Series:
    """Day-ahead prices in EUR/MWh on native resolution (60 or 15 min)."""

    def fetch(s, e):
        return _client().query_day_ahead_prices(zone, start=s, end=e).rename("price")

    return cached_monthly("entsoe", f"prices_{zone}", _ts(start), _ts(end), fetch)["price"]


def load_forecast(start, end, zone: str = ENTSOE_ZONE) -> pd.Series:
    """TSO day-ahead total load forecast (MW)."""

    def fetch(s, e):
        df = _client().query_load_forecast(zone, start=s, end=e)
        return df.iloc[:, 0].rename("load_forecast")

    return cached_monthly("entsoe", f"loadfc_{zone}", _ts(start), _ts(end), fetch)["load_forecast"]


def wind_solar_forecast(start, end, zone: str = ENTSOE_ZONE) -> pd.DataFrame:
    """Day-ahead wind (onshore/offshore) and solar generation forecasts (MW)."""

    def fetch(s, e):
        df = _client().query_wind_and_solar_forecast(zone, start=s, end=e)
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        return df

    return cached_monthly("entsoe", f"windsolarfc_{zone}", _ts(start), _ts(end), fetch)
