# Battery bidding in the Swedish day-ahead market (SE4)

Optimal day-ahead bidding and dispatch of a grid-scale battery (10 MW / 20 MWh)
in bidding zone SE4, with honest out-of-sample backtesting.

**Status:** work in progress. Data layer, deterministic MILP dispatch, a
15-minute LEAR price forecaster and backtests are done; probabilistic
forecasting and stochastic bidding are next.

## Problem

Each day before 12:00 CET, the battery operator submits bids for each of the
next day's 96 quarter-hours (24 hours before 2025-10-01) without knowing the
prices. Profit depends on the realized prices, the battery's physics and how
much uncertainty the bidding strategy accounts for.

## Design principles

- **Information discipline.** Strategies only see data published before gate
  closure (`bessopt.data.timegrid.gate_closure`). Day D-1's prices are known
  when bidding for D; day D's are not.
- **Settlement at realized prices.** Strategies output a schedule; the backtest
  settles it against actual prices, with state of charge carried across days.
- **Native resolution.** The series spans the hourly to 15-minute switch
  (2025-10-01) and DST days (23/25 h); the time grid is handled explicitly.
- **Binary charge/discharge exclusion.** SE4 prices go negative, and then the
  LP relaxation "burns" energy by charging and discharging simultaneously
  (see `tests/test_battery.py`).

## Results (SE4, out-of-sample 2025-01-01 to 2026-10-12, 650 days)

Each strategy forecasts prices, optimizes the next day's schedule with the
same MILP, and is settled at realized prices.

| Strategy | EUR/day | Share of perfect foresight<br>hourly era / 15-min era / overall |
|---|---:|---:|
| Perfect foresight (upper bound) | 2,007 | 100 % |
| Naive: tomorrow = today | 1,356 | 71 % / 65 % / 68 % |
| LEAR point forecast | 1,622 | 85 % / 78 % / 81 % |

Forecast accuracy (MAE in EUR/MWh at native resolution; rMAE relative to naive):

| Era | Naive MAE | LEAR MAE | LEAR rMAE |
|---|---:|---:|---:|
| Hourly (Jan to Sep 2025) | 27.4 | 18.5 | 0.67 |
| 15-min (Oct 2025 onwards) | 36.3 | 21.9 | 0.60 |

Observations:

- The daily price *shape* is so persistent that the naive forecast already
  captures two-thirds of the attainable value.
- A 40 % lower MAE buys only 13 percentage points of profit capture: arbitrage
  value depends on ranking cheap and expensive periods, not on absolute error.
  The remaining gap to perfect foresight motivates scenario-based bidding.
- In the 15-minute era, both forecasts capture a smaller *share* of perfect
  foresight, but LEAR earns more in absolute terms (about 1,660 vs 1,570
  EUR/day). The benchmark rose because intra-hour variation became
  tradable (see below). The eras also differ in season and volatility, so
  this is not a controlled comparison.
- Degradation is a 5 EUR/MWh throughput cost, giving about 1.7 cycles per
  day. That is probably optimistic and will be revisited.

### Forecaster: LEAR at 15-minute resolution

96 per-slot LASSO-ARX models (Lago et al., 2021) on asinh-transformed prices.
They are refit weekly on a rolling 2-year window, with the penalty chosen by
AIC along the LARS path. The 344 candidate features are all known at gate
closure:

- SE4 prices of D-1 (15 min) and D-2, D-3, D-7 (hourly);
- D-1 prices in SE3, DK2 and DE-LU;
- TSO load and wind forecasts for D and D-1;
- day-of-week dummies;
- a 15-minute-era dummy.

Before 2025-10-01 the targets are hourly prices repeated over four
quarter-hours. Because the per-slot models are separate, the era dummy learns
the mean intra-hour profile. LASSO selects it in 0 % of slots in January 2025,
41 % in November 2025 and 75 % in October 2026, as 15-minute data fills the
window.

**Does the 15-minute forecast pay off?** The table below covers the 15-minute
era (2025-10-01 to 2026-10-12, same days). All schedules are settled at
actual 15-minute prices.

| Strategy | EUR/day | Share of perfect foresight |
|---|---:|---:|
| Perfect foresight, 15-min | 2,127 | 100 % |
| Perfect foresight, hourly-constant schedules | 1,827 | 86 % |
| LEAR, 15-min forecast | 1,658 | 78 % |
| LEAR, forecast averaged to hourly | 1,570 | 74 % |

About 14 % of the attainable value lies within the hour. The 15-minute
forecast captures part of it: it beats its own hourly average by about
90 EUR/day and wins on 71 % of days.

## Roadmap

1. ~~Data layer, time grid, deterministic MILP, baseline backtests~~
2. ~~ENTSO-E features (load and wind forecasts, neighbouring zones), 15-minute LEAR~~
3. Probabilistic forecasts (quantile models) and scenario generation (copula)
4. Two-stage stochastic bidding with price-quantity curves; VSS, EVPI
5. Degradation modelling, terminal SoC valuation
6. Joint day-ahead + FCR-D capacity allocation

## Usage

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"          # add ,gurobi for Gurobi
cp .env.example .env              # add ENTSOE_API_KEY, choose solver
pytest
python scripts/fetch_data.py --start 2021-01-01      # cache ENTSO-E data
python scripts/run_lear.py --start 2025-01-01 --end 2026-10-13
```

HiGHS is the default solver so the project runs without a commercial
license; set `BESSOPT_SOLVER=GUROBI` to use Gurobi.

## Layout

```
src/bessopt/
  config.py             battery spec, paths, solver selection
  data/timegrid.py      gate closure, resolution handling, DST
  data/cache.py         monthly parquet cache
  data/elpris.py        SE prices from elprisetjustnu.se (no key)
  data/entsoe.py        ENTSO-E prices, load and RES forecasts
  forecast/features.py  gate-closure-safe daily design matrices (96 slots)
  forecast/lear.py      LEAR model and rolling out-of-sample loop
  forecast/evaluate.py  MAE / rMAE at native resolution
  optim/battery.py      deterministic dispatch MILP (cvxpy, DPP-parametrized)
  backtest/             strategies, rolling backtest, settlement
scripts/                runnable experiments
tests/
```

## Data sources

ENTSO-E Transparency Platform, elprisetjustnu.se (Nord Pool day-ahead prices),
Svenska kraftnät Mimer (balancing markets, planned).
