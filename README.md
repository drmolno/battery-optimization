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

### Probabilistic forecasts and stochastic bidding

**Forecast distribution.** Per-slot predictive quantiles are LEAR plus
empirical quantiles of recent LEAR errors (asinh space, 182-day window,
pooled over +-2 neighbouring quarter-hours, centred so the median equals
LEAR). Quantile Regression Averaging (QRA) was evaluated as an alternative
and was no better. Over the test period the pinball loss is 6.10 (hourly
era) and 7.28 (15-min era). The 90 % intervals cover 90 % and 87 %, so they
are slightly narrow in the 15-minute era.

**Scenarios.** 50 joint daily price paths per day from a Gaussian copula,
whose correlation is estimated from rank-normal scores of recent errors with
Ledoit-Wolf shrinkage. Compared with sampling each quarter-hour independently,
the copula scenarios score 3 % better on the energy score and 15-23 % better
on the variogram score, which measures dependence between slots. Scenario
means are moment-matched to the exact distribution mean.

![Scenarios](docs/scenarios_2026-05-28.png)

**Bidding.** Two-stage stochastic program:

- *First stage:* monotone price-quantity curves with K = 8 breakpoints per
  period, placed at quantiles of the scenario prices.
- *Second stage:* per-scenario dispatch and SoC, which must be feasible in
  every scenario.

It is solved as an LP using the convex hull `c + d <= P` of the
charge/discharge exclusion. Against the exact MILP (lazy binaries) the
in-sample gap is 0.5 % at about 60x the cost. Curves are cleared at realized
prices; any dispatch that is physically infeasible is settled as imbalance
at a 50 EUR/MWh penalty.

| Strategy (650 days) | EUR/day | Share of PF<br>hourly / 15-min | vs expected value |
|---|---:|---:|---:|
| LEAR point forecast (≈ predictive median) | 1,623 | 85.4 % / 78.1 % | +16.4 |
| Stochastic curves, K = 8, S = 50 | 1,609 | 84.7 % / 77.3 % | +2.1 ± 0.9 |
| Expected value: schedule on predictive mean | 1,607 | 84.7 % / 77.2 % | 0 |

Findings:

- **Value of the stochastic solution (VSS) is about 0.1 %.** Price-dependent
  curves beat the expected-value schedule by 2.1 ± 0.9 EUR/day (out of
  sample). The curves were imbalanced for only 0.75 EUR/day on average.
- **Median beats mean.** The schedule on the predictive median (LEAR) beats
  the schedule on the predictive mean by 16 ± 5 EUR/day. Theory says the mean
  should win for a fixed schedule, since profit is linear in price. On
  average the mean is in fact better calibrated (bias −1.0 vs −2.9 EUR/MWh),
  so the difference likely comes from how errors are placed within the day,
  not from overall bias. This is open.
- **Interpretation.** For a price-taking 10 MW battery in day-ahead alone,
  almost all of the attainable value comes from forecasting the daily price
  shape. Reacting to realized prices through bid curves adds little, because
  the battery's profit depends on spreads across the day, and a per-period
  curve can't condition on the other periods' prices. The case for
  probabilistic models is therefore stronger for risk management and for
  co-optimization with reserves (FCR-D), where the trade-off is explicitly
  about uncertainty.

## Roadmap

1. ~~Data layer, time grid, deterministic MILP, baseline backtests~~
2. ~~ENTSO-E features (load and wind forecasts, neighbouring zones), 15-minute LEAR~~
3. ~~Probabilistic forecasts (empirical residuals, QRA) and copula scenarios~~
4. ~~Two-stage stochastic bidding with price-quantity curves; VSS~~
5. Degradation modelling, terminal SoC valuation
6. Joint day-ahead + FCR-D capacity allocation
7. Project economics: NPV/IRR with capex, opex, grid fees, capacity fade and
   cell replacement; sensitivity to capex, degradation cost and spread
   compression

## Usage

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"          # add ,gurobi for Gurobi
cp .env.example .env              # add ENTSOE_API_KEY, choose solver
pytest
python scripts/fetch_data.py --start 2021-01-01      # cache ENTSO-E data
python scripts/run_lear.py --start 2025-01-01 --end 2026-10-13
python scripts/run_probabilistic.py --start 2025-01-01 --end 2026-10-13
python scripts/run_stochastic.py --start 2025-01-01 --end 2026-10-13 --moment-match --tag full
```

`run_probabilistic.py` needs out-of-sample LEAR forecasts for a calibration
period before `--start`. Produce them with `run_lear.py` from 2024-06-01.

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
  forecast/probabilistic.py  quantiles (ER, QRA), Gaussian copula, scoring
  optim/battery.py      deterministic dispatch MILP (cvxpy, DPP-parametrized)
  optim/stochastic.py   two-stage stochastic bid curves (K-point grid)
  backtest/             strategies, rolling backtest, settlement
scripts/                runnable experiments
tests/
```

## Data sources

ENTSO-E Transparency Platform, elprisetjustnu.se (Nord Pool day-ahead prices),
Svenska kraftnät Mimer (balancing markets, planned).
