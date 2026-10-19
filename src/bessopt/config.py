"""Project-wide configuration: paths, market constants, battery specification."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"

load_dotenv(PROJECT_ROOT / ".env")

ZONE = "SE4"
ENTSOE_ZONE = "SE_4"
# Nord Pool / SDAC operate in CET/CEST, which coincides with Swedish local time.
MARKET_TZ = "Europe/Stockholm"


def entsoe_api_key() -> str:
    key = os.environ.get("ENTSOE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "ENTSOE_API_KEY not set. Copy .env.example to .env and add your token."
        )
    return key


def default_solver() -> str:
    return os.environ.get("BESSOPT_SOLVER", "HIGHS").upper()


@dataclass(frozen=True)
class BatterySpec:
    """Physical and economic parameters of the battery.

    Defaults: 10 MW / 20 MWh, 90 % round-trip efficiency split symmetrically
    between charging and discharging, usable SoC window 5-95 %.
    """

    power_mw: float = 10.0
    energy_mwh: float = 20.0
    round_trip_eff: float = 0.90
    soc_min_frac: float = 0.05
    soc_max_frac: float = 0.95
    soc_init_frac: float = 0.50
    # Minimum state of charge at the end of each day (fraction of capacity).
    # 5 % = the normal lower limit, i.e. no extra end-of-day rule: a 50 %
    # minimum forces refilling before midnight at late-evening prices instead
    # of after midnight at the cheapest night prices, and costs about 11 %
    # profit (scripts/run_end_level.py).
    soc_end_min_frac: float = 0.05
    # Wear cost per MWh discharged (EUR/MWh), a throughput model:
    #   cell replacement cost / lifetime discharged energy
    #   = 20 MWh x 80 kEUR/MWh / (7,000 cycles x 18 MWh usable) ~ 13,
    # rounded up to 15. Assumptions: LFP stationary packs ~70 USD/kWh (BNEF
    # 2025) plus installation; 6,000-10,000 cycles to 70-80 % capacity.
    # Plausible range about 8-22. The optimizer plans with this value as is;
    # no safety margin is built in.
    degradation_eur_per_mwh: float = 15.0

    @property
    def eta_charge(self) -> float:
        return math.sqrt(self.round_trip_eff)

    @property
    def eta_discharge(self) -> float:
        return math.sqrt(self.round_trip_eff)

    @property
    def soc_min(self) -> float:
        return self.soc_min_frac * self.energy_mwh

    @property
    def soc_max(self) -> float:
        return self.soc_max_frac * self.energy_mwh

    @property
    def soc_init(self) -> float:
        return self.soc_init_frac * self.energy_mwh

    @property
    def soc_end_min(self) -> float:
        return max(self.soc_end_min_frac * self.energy_mwh, self.soc_min)
