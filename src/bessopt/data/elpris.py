"""Day-ahead prices from elprisetjustnu.se (no API key required).

Used as a fallback/prototyping source until the ENTSO-E token is available.
The API serves one JSON file per zone and delivery day, from late 2022.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

from bessopt.config import MARKET_TZ, ZONE
from bessopt.data.cache import cached_monthly

URL = "https://www.elprisetjustnu.se/api/v1/prices/{d:%Y}/{d:%m-%d}_{zone}.json"


def fetch_day(day: dt.date, zone: str = ZONE, session: requests.Session | None = None) -> pd.Series:
    """Prices for one delivery day in EUR/MWh, empty if not (yet) published."""
    s = session or requests
    r = s.get(URL.format(d=day, zone=zone), timeout=30)
    if r.status_code == 404:
        return pd.Series(dtype=float, name="price")
    r.raise_for_status()
    rows = r.json()
    # Parse via UTC: DST days mix +01:00 and +02:00 offsets.
    idx = pd.to_datetime([x["time_start"] for x in rows], utc=True).tz_convert(MARKET_TZ)
    return pd.Series([1000.0 * x["EUR_per_kWh"] for x in rows], index=idx, name="price")


def _fetch_range(zone: str):
    def fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        days = pd.date_range(start.date(), end.date(), freq="D", inclusive="left")
        with requests.Session() as session:
            parts = [fetch_day(d.date(), zone, session) for d in days]
        parts = [p for p in parts if not p.empty]
        return pd.concat(parts) if parts else pd.Series(dtype=float, name="price")

    return fetch


def day_ahead_prices(start, end, zone: str = ZONE) -> pd.Series:
    """SE day-ahead prices (EUR/MWh) on the native market resolution."""
    df = cached_monthly("elpris", zone, pd.Timestamp(start), pd.Timestamp(end), _fetch_range(zone))
    return df["price"]
