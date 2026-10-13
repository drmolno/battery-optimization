import os

import numpy as np
import pandas as pd
import pytest

from bessopt.data.timegrid import split_by_delivery_day

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not os.environ.get("ENTSOE_API_KEY"), reason="no ENTSOE_API_KEY"),
]


def test_entsoe_prices_match_elpris_across_switch():
    from bessopt.data import elpris, entsoe

    a = entsoe.day_ahead_prices("2025-09-29", "2025-10-03")
    b = elpris.day_ahead_prices("2025-09-29", "2025-10-03")
    pd.testing.assert_index_equal(a.index, b.index)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=0.01)


def test_forecasts_cover_dst_day():
    from bessopt.data import entsoe

    load = entsoe.load_forecast("2025-10-26", "2025-10-27")
    assert len(split_by_delivery_day(load)[pd.Timestamp("2025-10-26").date()]) == 25
