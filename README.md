# Battery bidding in the Swedish day-ahead market (SE4)

Optimal day-ahead bidding and dispatch of a grid-scale battery (10 MW / 20 MWh)
in bidding zone SE4, with honest out-of-sample backtesting.

**Status:** work in progress. Done: data layer, deterministic dispatch optimization,
a 15-minute LEAR price forecaster, probabilistic forecasts with copula
scenarios, two-stage stochastic bidding, and out-of-sample backtests of all
strategies. Next: degradation modelling, FCR-D co-optimization and project
economics (see Roadmap).

## Problem

Each day before 12:00 CET (gate closure), the battery operator submits bids
to the Nord Pool day-ahead auction for each of the next day's 96 quarter-hours
(24 hours before 2025-10-01).

**What a bid is.** For one delivery period, a bid is a list of price-quantity
pairs: how much energy the operator is willing to sell (discharge) or buy
(charge) at each price. For example:

| Price (EUR/MWh) | ≤ 20 | 70 | 120 | ≥ 200 |
|---|---:|---:|---:|---:|
| Net quantity (MW) | buy 10 | 0 | sell 8 | sell 10 |

Quantities between the listed prices are interpolated linearly: at 95,
halfway between 70 and 120, the bid is to sell 4 MW. A sell
quantity at 120 means "willing to sell at any price of 120 or more".

**What happens next.** At about 12:45 the auction publishes one clearing price
per period for the whole zone. Each bid is then a binding commitment: if the
18:00-18:15 price clears at 120, the operator *must* sell 8 MW during that
quarter-hour (2 MWh), and is paid the clearing price for every MWh. All
participants get the same clearing price, not the price they bid
(uniform pricing). Delivering less than committed has to be settled in the
imbalance market, typically at a worse price.

A fixed schedule ("sell 8 MW at 18:00 whatever the price") is the special
case of a flat curve.

**The battery.** All results use one hypothetical battery:

| Parameter | Symbol | Value |
|---|---|---|
| Power rating (max charge or discharge power) | P | 10 MW |
| Energy capacity | E | 20 MWh |
| Usable state of charge (SoC) | | 5-95 % of E, i.e. 1-19 MWh |
| Round-trip efficiency (charge and discharge losses split evenly) | η_c, η_d | 90 % (√0.9 ≈ 0.95 each way) |
| SoC at the start of the backtest; minimum SoC at the end of each day | | 50 % (10 MWh) |
| Wear cost per MWh discharged | κ | 15 EUR/MWh (see *Battery wear*) |

The battery is assumed small enough that its bids do not move the market
price (price-taker).

**The problem.** Bids must be fixed before the prices are known. Profit
depends on the realized prices, the battery's physical limits (power,
energy, efficiency) and how the bids handle the uncertainty about prices.

## What is being optimized

**The real objective** is the expected discounted cash flow over the
battery's life: daily market revenue, minus the upfront investment and
any cell replacement. Two states link the days:

- *State of charge* (fast): energy carried from one day into the next.
- *State of health* (slow): every MWh discharged uses up part of a finite
  cycle budget; when it is spent, or calendar aging ends the life, the cells
  must be replaced.

Prices are uncertain and revealed one day at a time, so in principle this is
a stochastic dynamic program over years.

**What the code solves** is one day at a time, a receding-horizon scheme.
Each day maximizes that day's revenue plus an *approximate* value of the
battery's state at the end of the day. The approximation has two parts, and
both are engineering estimates rather than derived quantities:

1. **Wear: a linear cost κ per MWh discharged.** Formally this is the
   Lagrange multiplier of the lifetime cycle budget: dualizing "total
   throughput ≤ budget" splits the lifetime problem into independent daily
   problems "revenue − κ · throughput". With the right κ these daily
   decisions are optimal over the lifetime (for a convex problem). The right
   κ is the marginal value of a cycle; we *estimate* it as replacement cost
   divided by lifetime throughput (see *Battery wear*). If calendar aging
   would end the life before the cycle budget runs out, the true κ is lower.
2. **Stored energy: a floor instead of a value.** Each day must end at
   least 50 % full. This stands in for the value of energy carried
   overnight and is the more heuristic of the two. A proper version would
   value end-of-day energy by the expected next-day spread.

Both are on the roadmap: estimate κ adaptively so cumulative throughput
tracks the cycle budget, and replace the 50 % floor by a value of stored
energy. All reported profits are for this daily scheme, settled at real
prices.

## Design principles

- **Information discipline.** Strategies only see data published before gate
  closure (`bessopt.data.timegrid.gate_closure`). Day D-1's prices are known
  when bidding for D; day D's are not.
- **Settlement at realized prices.** Strategies output a schedule; the backtest
  settles it against actual prices, with state of charge carried across days.
- **Native resolution.** The series spans the hourly to 15-minute switch
  (2025-10-01) and the daylight-saving-time changeover days, which have 23
  or 25 hours (92 or 100 quarter-hours); the time grid is handled
  explicitly.
- **Charging plus discharging within a period cannot exceed the inverter
  rating.** The model tracks charging power c and discharging power d
  separately, because conversion losses differ by direction. Power flows one
  way at a time, but the battery can switch direction within a period:
  charge for part of the 15 minutes, discharge for the rest. Since c and d
  are averages over the period, this is a time budget: c + d ≤ P, where P is
  the power rating (10 MW). For example, 5 minutes charging and 10 minutes
  discharging at full power gives c = 3.3 and d = 6.7 MW.

  Switching within a period wastes energy through real round-trip losses
  (about 10 %), which pays only at negative prices, when you are paid to
  consume. In SE4 negative prices occurred on 104 of 651 days since January
  2025 (2 % of all periods). Whether wasting energy is worth it is then
  decided by the wear cost (see *Battery wear*). Without the shared limit
  (c ≤ P and d ≤ P separately) the model would run full power in and out
  at once, which is unphysical, and on a day at −100 EUR/MWh would report
  1.5x the attainable profit (`tests/test_battery.py`).

## Results (SE4, out-of-sample 2025-01-01 to 2026-10-12, 650 days)

Each strategy forecasts prices, optimizes the next day's schedule with the
same battery model (wear cost 15 EUR/MWh), and is settled at realized
prices.

Profit is net of wear. All strategies cycle about 1.4 times per day. The
wear charge (see *Battery wear*) is a decision cost representing used-up
cycle life, not a cash expense; payback and NPV will be computed from cash
flows (revenue before wear, capex, cell replacement) in the economics step.

| Strategy | EUR/day | Share of perfect foresight<br>hourly era / 15-min era / overall |
|---|---:|---:|
| Perfect foresight (upper bound) | 1,701 | 100 % |
| Naive: tomorrow = today | 1,059 | 65 % / 61 % / 62 % |
| LEAR point forecast | 1,343 | 83 % / 77 % / 79 % |

Forecast accuracy (MAE in EUR/MWh at native resolution; rMAE relative to naive):

| Era | Naive MAE | LEAR MAE | LEAR rMAE |
|---|---:|---:|---:|
| Hourly (Jan to Sep 2025) | 27.4 | 18.5 | 0.67 |
| 15-min (Oct 2025 onwards) | 36.3 | 21.9 | 0.60 |

Observations:

- The daily price *shape* is so persistent that the naive forecast already
  captures about 60 % of the attainable value.
- A 40 % lower MAE buys 17 percentage points of profit capture: arbitrage
  value depends on ranking cheap and expensive periods, not on absolute error.
  The remaining gap to perfect foresight motivates scenario-based bidding.
- In the 15-minute era, both forecasts capture a smaller *share* of perfect
  foresight, but LEAR earns more in absolute terms (about 1,385 vs 1,285
  EUR/day). The benchmark rose because intra-hour variation became
  tradable (see below). The eras also differ in season and volatility, so
  this is not a controlled comparison.

### Battery wear

Each MWh discharged costs a wear charge κ, a throughput model:

$$
\kappa = \frac{\text{cell replacement cost}}{\text{lifetime discharged energy}}
\approx \frac{20\ \text{MWh} \times 80\ \text{kEUR/MWh}}{7{,}000\ \text{cycles} \times 18\ \text{MWh}}
\approx 13 \to 15\ \text{EUR/MWh}.
$$

Assumptions:

- *Cell cost:* stationary LFP packs at about 70 USD/kWh (BloombergNEF 2025
  survey), plus installation.
- *Cycle life:* 6,000-10,000 full cycles before capacity falls to 70-80 %.

These are simplifying, non-expert assumptions from public sources, so the
plausible range is wide, about 8-22 EUR/MWh. Calendar aging does not depend
on operation and belongs in the project economics. Cycle-depth effects are
ignored. The optimizer plans with κ = 15, with no extra safety margin.

To check the choice, the optimizer was run with a range of assumed wear
costs, and every run was charged the true cost (LEAR forecast, 650 days):

| Planning wear cost (EUR/MWh) | 0 | 5 | 10 | **15** | 20 | 30 | 50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cycles per day | 2.2 | 1.8 | 1.5 | 1.4 | 1.3 | 1.1 | 0.8 |
| Net profit at true 15 (EUR/day) | 1,116 | 1,271 | 1,323 | 1,343 | 1,346 | 1,304 | 1,135 |
| Net profit at true 8 (EUR/day) | 1,430 | 1,517 | 1,537 | 1,538 | 1,527 | 1,459 | 1,243 |
| Net profit at true 22 (EUR/day) | 803 | 1,025 | 1,109 | 1,148 | 1,165 | 1,149 | 1,028 |

- **Ignoring wear is expensive.** Planning with κ = 0 cycles 2.2 times a day
  and loses about 230 EUR/day against planning with the true cost.
- **Under perfect foresight the best planning cost equals the true cost**,
  as theory says. For the forecast-based strategy the optimum is slightly
  above it (by about 3 EUR/day; flat around the optimum). Cycling on
  forecast spreads carries risk the strategy does not account for. That is
  reported here but deliberately *not* built in by inflating κ; a principled
  treatment is risk-aware bidding (see Roadmap).
- **Wasting energy at negative prices never paid in SE4.** A round trip that
  discharges 1 MWh loses about 0.11 MWh, so at price p it earns about
  0.11·|p| against a wear cost of κ. That pays only below about −135 EUR/MWh
  at κ = 15. SE4's lowest price since 2021 is −60 EUR/MWh. In the backtests
  the optimizer never switches direction within a period (0 of 42,743
  periods, for both perfect foresight and LEAR). Switching can still occur
  when bid curves are settled, where it lets a nearly full battery honour a
  committed purchase instead of paying for imbalance.

### Forecaster: LEAR at 15-minute resolution

**LEAR = LASSO-Estimated AutoRegressive model.** It is a linear regression
of tomorrow's price on many past prices (the *autoregressive* part) and on
other information such as load and wind forecasts, fitted with LASSO (least
absolute shrinkage and selection operator). LASSO adds a penalty on the absolute size of the coefficients, which
sets most of them exactly to zero, so the data picks the few inputs that
matter. LEAR was proposed by Uniejewski, Nowotarski and Weron (2016). Lago,
Marcjasz, De Schutter and Weron (2021) made it the reference benchmark for
day-ahead price forecasting, with an open implementation (`epftoolbox`).
The design here follows their setup; differences are listed below.

**One model per delivery period.** Each of the 96 quarter-hours of day D
gets its own linear model,

$$
\tilde y_{D,h} = \beta_h^\top \tilde x_D + \varepsilon_{D,h}, \qquad
\hat\beta_h = \arg\min_\beta \sum_{d} \big(\tilde y_{d,h} - \beta^\top \tilde x_d\big)^2 + \lambda_h \lVert\beta\rVert_1 ,
$$

so the 18:00 price can depend on the inputs differently from the 03:00
price. All 96 models see the same input vector x̃_D. The penalty λ_h is
chosen per model by AIC (Akaike information criterion) along the LARS (least
angle regression) path: the LASSO solution changes piecewise linearly in λ,
and LARS computes all of it in about the cost of one least-squares fit, so
the AIC can be evaluated at every breakpoint.

**Inputs** (344 candidates, all known at gate closure, 12:00 on D-1):

- SE4 prices of D-1 (all 96 quarter-hours) and D-2, D-3, D-7 (hourly);
- D-1 prices in the coupled zones SE3, DK2 and DE-LU (hourly);
- the grid operator's load and wind forecasts for D and D-1 (hourly);
- day-of-week dummies;
- a 15-minute-era dummy (1 from 2025-10-01), and a flag for the 9 days whose
  load or wind forecast was missing and filled from the previous day.

A median of about 48 inputs survive per model.

**The asinh transformation.** Electricity prices are heavy-tailed: mostly
moderate, with occasional spikes to several hundred EUR/MWh and sometimes
negative values. Least squares is dominated by the spikes, so a few extreme
hours would drive the coefficients. A log transform would tame them, but it
is undefined for zero and negative prices. The area hyperbolic sine,

$$
\operatorname{asinh}(z) = \ln\!\big(z + \sqrt{z^2 + 1}\big)
\;\approx\; \begin{cases} z & |z| \text{ small}, \\ \operatorname{sign}(z)\,\ln(2|z|) & |z| \text{ large}, \end{cases}
$$

behaves like the identity near zero and like a log for large values, on
both sides. It compresses spikes and keeps negative prices.

Each price series is first centred and scaled with robust statistics from
the training window, z = (price − median) / MAD (median absolute deviation),
so that "near zero" means "a typical price". The models are fitted on
asinh(z), and forecasts are mapped back with sinh. This transformation for
electricity prices was studied by Uniejewski, Weron and Ziel (2018), *IEEE
Transactions on Power Systems*. Lago et al. (2021) use it as the default
in LEAR. Inputs are transformed the same way, except dummies.

A side effect: back-transforming the fitted mean of asinh(z) gives
something close to the *median* of the price, not its mean, because sinh is
nonlinear. This is why the LEAR forecast is treated as the predictive
median in the probabilistic section.

**Training.** Rolling 2-year window, refit weekly, forecasts every day. Only
days whose prices are known at gate closure enter the training set.
Constant and duplicate input columns are dropped per window: before the
15-minute switch the four quarter-hour lags of an hour are identical copies,
which breaks the LARS algorithm.

**Differences from Lago et al. (2021):**

| | Lago et al. | Here |
|---|---|---|
| Periods per day | 24 hourly models | 96 quarter-hour models |
| Refitting | daily | weekly (daily is about 7x slower) |
| Training window | several windows (56 days to 4 years), forecasts averaged | one 2-year window |
| Exogenous inputs | 2 market-specific forecasts for D, D-1, D-7 | load and wind for D, D-1, plus neighbouring-zone prices |
| Penalty | AIC on the LARS path, then refit with coordinate descent | AIC on the LARS path, solution taken directly from the path (same estimate in principle) |
| Extra | | 15-minute-era dummy, imputation flag |

**Limitation: one set of coefficients for all weekdays.** Weekdays enter
only through the dummies, which shift the level: "Sundays are lower by this
much". The *weights* on the inputs are shared. So the model cannot express,
for example, that for a Monday yesterday (a Sunday) is a weaker guide than
for a Wednesday, and last week's Monday a stronger one. The D-7 lag
("same weekday last week") and the Monday dummy partly compensate.
Weekday-specific weights, via interaction inputs such as "D-1 average price
× Monday" that LASSO may select, are left as further work. Separate models
per weekday are not viable: each would have only about 104 training days,
fewer than inputs.

**Tried, no gain: holiday flags.** Swedish public holidays plus Midsummer
Eve, Christmas Eve and New Year's Eve behave like Sundays, so flags for "D
is a holiday" and "D-1 was a holiday" were added as inputs (`run_lear.py
--holidays`). Forecast error did not change (MAE 18.1 → 18.0 on the 27
holidays, 20.6 → 20.6 on other days) and profit changed by −1 ± 2 EUR/day.
LEAR was already *more* accurate on holidays than on other days, probably
because low holiday demand shows up in the load forecast. The flags are
off by default.

**The 15-minute-era dummy.** Before 2025-10-01 the targets are hourly prices
repeated over four quarter-hours. Because the per-slot models are separate,
the era dummy learns the mean intra-hour profile. LASSO selects it in 0 % of
slots in January 2025, 41 % in November 2025 and 75 % in October 2026, as
15-minute data fills the window.

**Does the 15-minute forecast pay off?** The table below covers the 15-minute
era (2025-10-01 to 2026-10-12, same days). All schedules are settled at
actual 15-minute prices.

| Strategy | EUR/day | Share of perfect foresight |
|---|---:|---:|
| Perfect foresight, 15-min | 1,807 | 100 % |
| Perfect foresight, hourly-constant schedules | 1,561 | 86 % |
| LEAR, 15-min forecast | 1,385 | 77 % |
| LEAR, forecast averaged to hourly | 1,311 | 73 % |

About 14 % of the attainable value lies within the hour. The 15-minute
forecast captures part of it: it beats its own hourly average by about
75 EUR/day and wins on 71 % of days.

### Beyond a point forecast: bidding under uncertainty

So far every strategy turns a single price forecast into a fixed schedule:
"charge 10 MW at 03:00, discharge at 18:00", whatever the prices turn out to
be. But the forecast is uncertain, and a bid does not have to be fixed. As
described under *Problem*, a bid can be a curve: sell more if the price
turns out high, buy more if it turns out low. That raises the question this
section answers: **is it worth knowing how uncertain the forecast is, and
bidding curves that react to the actual price?**

Answering it takes three steps:

1. **Forecast a distribution, not just a number:** for each quarter-hour, a
   range of likely prices around the LEAR forecast.
2. **Generate scenarios:** plausible full-day price paths (here 50 per day)
   consistent with those ranges and with how prices move together over the
   day.
3. **Optimize the bid curves against all scenarios at once,** so that the
   curves do well on average over the plausible futures.

The benchmark is the *expected-value* strategy: a fixed schedule optimized
for the average of the same scenarios. The difference between the two is
the value of the stochastic solution (VSS; Birge & Louveaux, 2011): what reacting to the actual
price is worth.

**1. Forecast distribution.** For each quarter-hour, the price range comes
from LEAR's own recent errors. If over the last 182 days LEAR was within
±20 EUR/MWh 90 % of the time, the forecast range for tomorrow is LEAR ±
about 20 at 90 %. Some details:

- Errors are measured on the asinh scale, so ranges are wider at high prices
  and around spikes.
- Each quarter-hour pools errors from its neighbours (±2 quarter-hours) for
  more stable estimates.
- The ranges are centred so that the middle of the range is the LEAR
  forecast.

A more elaborate method, Quantile Regression Averaging (QRA; Nowotarski &
Weron, 2015), was no better.
Over the test period the 90 % ranges contain the actual price 90 % of the
time in the hourly era and 87 % in the 15-minute era, so they are slightly
too narrow lately. Average pinball loss, the standard accuracy score for
ranges: 6.10 and 7.28 EUR/MWh.

**2. Scenarios.** Sampling each quarter-hour separately from its range would
give jagged, unrealistic days: price jumps of 100 EUR between neighbouring
quarter-hours. A battery optimizer would see arbitrage everywhere that never
happens. Real forecast errors are correlated: if the midday solar dip is
deeper than forecast at 12:00, it is probably deeper at 14:00 too. So the
scenarios are sampled with that correlation, estimated from the past 182
days of LEAR errors.

The tool for this is a *copula*. Sklar's theorem (Sklar, 1959; Nelsen,
2006) says any joint distribution can be split into its marginals (the
distribution of each quarter-hour on its own, from step 1) and a copula,
which describes only how the quarter-hours move together. The two can then
be modelled separately. A *Gaussian* copula takes the dependence from a
multivariate normal distribution and nothing else. This is the standard
way to turn per-period probabilistic forecasts into scenarios in energy
forecasting, introduced for wind power by Pinson et al. (2009). The recipe:

1. *Estimate the dependence.* For each quarter-hour, rank LEAR's errors
   over the past 182 days and map the ranks to standard-normal values
   (rank-based normal scores). Compute the 96 × 96 correlation matrix R of
   these values. Because 182 days is little data for 96 × 96 correlations,
   R is shrunk slightly towards zero with the Ledoit-Wolf (2004) estimator.
2. *Sample.* Draw a 96-dimensional normal vector z with correlation R and
   convert each component to a probability level u = Φ(z) (Φ the
   standard-normal CDF). Each u lies between 0 and 1, and neighbouring
   quarter-hours get similar levels.
3. *Map to prices.* Read each quarter-hour's price off its own distribution
   from step 1 at level u: if 12:00 lands at its 10 % level, its price is
   the 10 % quantile of the 12:00 forecast.

Each scenario thus has exactly the forecast spread in every quarter-hour,
and correlation between them taken from history. The random draws are
seeded by date, so every strategy sees the same scenarios. The figure shows
one day:

![Scenarios](docs/scenarios_2026-05-28.png)

Correlated scenarios (panel B) look like real days; independent sampling
(panel C) does not. On scores that check whole-day paths rather than single
quarter-hours, the correlated scenarios are 3 % better on the energy score
(Gneiting & Raftery, 2007) and 15-23 % better on the variogram score
(Scheuerer & Hamill, 2015), which specifically checks how well the relation
between quarter-hours is captured.

One correction: with only 50 scenarios, their average differs from the true
forecast mean by random noise of several EUR/MWh in each quarter-hour, which
the optimizer would mistake for real price differences. The scenarios are
shifted so their average matches the forecast mean exactly. This is a simple
form of moment matching, a standard scenario-generation technique (Høyland
& Wallace, 2001).

Limitations: a Gaussian copula has no tail dependence, so it under-represents
several neighbouring quarter-hours spiking together; a Student-t copula
would capture that. The correlation is also the same every day, regardless
of the weather situation.

**3. Bid curves.** One optimization per day chooses:

- *before the auction:* a bid curve for each quarter-hour, as in the
  *Problem* example, with 8 price points (placed at quantiles of that
  quarter-hour's scenario prices) and non-decreasing quantities;
- *for each scenario:* how the battery would then operate, given the
  quantities the curves clear at that scenario's prices. The state of charge
  must stay within limits in every scenario.

The objective is the average profit over the 50 scenarios. The battery model
is the same as above, including c + d ≤ P and the wear cost. In technical
terms this is a two-stage stochastic linear program (Birge & Louveaux,
2011); placing curve points at scenario prices follows Fleten & Pettersen
(2005). The curves are then
settled at the actual prices. If the actual prices clear an amount the
battery cannot physically deliver (for example, selling when nearly empty),
the shortfall is settled as imbalance at a 50 EUR/MWh penalty.

| Strategy (650 days, net of wear) | EUR/day | Share of perfect foresight<br>hourly / 15-min | vs expected value (EUR/day) |
|---|---:|---:|---:|
| LEAR point forecast (= predictive median) | 1,343 | 82.6 % / 76.6 % | +9.2 ± 4.6 |
| Stochastic curves, K = 8, S = 50 | 1,334 | 82.5 % / 75.9 % | +0.4 ± 1.0 |
| Expected value: schedule on predictive mean | 1,334 | 82.5 % / 75.9 % | 0 |

(± is one standard error over days.)

Findings:

- **The value of the stochastic solution (VSS) is zero within noise.**
  Price-dependent curves beat the expected-value schedule by 0.4 ± 1.0
  EUR/day out of sample. Imbalance from infeasible dispatch cost only
  0.6 EUR/day. In an earlier run with a lower wear cost (5 EUR/MWh) the gain
  was 2 EUR/day (0.1 %), so it is small either way.
- **Median beats mean, marginally.** The schedule on the predictive median
  (LEAR) beats the schedule on the predictive mean by 9 ± 5 EUR/day, about
  two standard errors. Theory says the mean should win for a fixed schedule,
  since profit is linear in price. On average the mean is in fact better
  calibrated (bias −1.0 vs −2.9 EUR/MWh), so the difference likely comes from
  how errors are placed within the day, not from overall bias. This is open.
- **Interpretation.** For a price-taking 10 MW battery in day-ahead alone,
  almost all of the attainable value comes from forecasting the daily price
  shape. Reacting to realized prices through bid curves adds little, because
  the battery's profit depends on spreads across the day, and a per-period
  curve can't condition on the other periods' prices. The case for
  probabilistic models is therefore stronger for risk management and for
  co-optimization with reserves (FCR-D), where the trade-off is explicitly
  about uncertainty.

## Roadmap

1. ~~Data layer, time grid, deterministic dispatch, baseline backtests~~
2. ~~ENTSO-E features (load and wind forecasts, neighbouring zones), 15-minute LEAR~~
3. ~~Probabilistic forecasts (empirical residuals, QRA) and copula scenarios~~
4. ~~Two-stage stochastic bidding with price-quantity curves; VSS~~
5. ~~Wear cost (throughput model) and intra-period switching~~; still open:
   value of end-of-day stored energy instead of the 50 % floor, κ estimated
   adaptively as a shadow price (cumulative throughput tracking the cycle
   budget), cycle-depth-dependent wear
6. Risk-aware bidding: account for forecast risk explicitly (e.g. a CVaR
   term in the stochastic program) instead of implicitly through cost
   parameters
7. Joint day-ahead + FCR-D capacity allocation
8. Project economics: NPV/IRR with capex, opex, grid fees, capacity fade and
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
  data/timegrid.py      gate closure, resolution handling, daylight saving
  data/cache.py         monthly parquet cache
  data/elpris.py        SE prices from elprisetjustnu.se (no key)
  data/entsoe.py        ENTSO-E prices, load and RES forecasts
  forecast/features.py  gate-closure-safe daily design matrices (96 slots)
  forecast/lear.py      LEAR model and rolling out-of-sample loop
  forecast/evaluate.py  MAE / rMAE at native resolution
  forecast/probabilistic.py  quantiles (ER, QRA), Gaussian copula, scoring
  optim/battery.py      deterministic dispatch LP (cvxpy, DPP-parametrized)
  optim/stochastic.py   two-stage stochastic bid curves (K-point grid)
  backtest/             strategies, rolling backtest, settlement
scripts/                runnable experiments
tests/
```

## Data sources

ENTSO-E Transparency Platform, elprisetjustnu.se (Nord Pool day-ahead prices),
Svenska kraftnät Mimer (balancing markets, planned).

## References

*Price forecasting*

- Uniejewski, B., Nowotarski, J., Weron, R. (2016). Automated variable
  selection and shrinkage for day-ahead electricity price forecasting.
  *Energies* 9(8), 621. (The LEAR model.)
- Uniejewski, B., Weron, R., Ziel, F. (2018). Variance stabilizing
  transformations for electricity spot price forecasting. *IEEE Transactions
  on Power Systems* 33(2), 2219-2229. (The asinh transform.)
- Lago, J., Marcjasz, G., De Schutter, B., Weron, R. (2021). Forecasting
  day-ahead electricity prices: A review of state-of-the-art algorithms,
  best practices and an open-access benchmark. *Applied Energy* 293, 116983.
  (LEAR as benchmark; `epftoolbox`.)
- Nowotarski, J., Weron, R. (2015). Computing electricity spot price
  prediction intervals using quantile regression and forecast averaging.
  *Computational Statistics* 30(3), 791-803. (QRA.)

*Statistical methods*

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
  *Journal of the Royal Statistical Society B* 58(1), 267-288.
- Efron, B., Hastie, T., Johnstone, I., Tibshirani, R. (2004). Least angle
  regression. *Annals of Statistics* 32(2), 407-499. (LARS.)
- Zou, H., Hastie, T., Tibshirani, R. (2007). On the "degrees of freedom" of
  the lasso. *Annals of Statistics* 35(5), 2173-2192. (AIC for LASSO.)
- Akaike, H. (1974). A new look at the statistical model identification.
  *IEEE Transactions on Automatic Control* 19(6), 716-723. (AIC.)
- Ledoit, O., Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices. *Journal of Multivariate Analysis*
  88(2), 365-411.

*Copulas, scenarios and scoring*

- Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges.
  *Publications de l'Institut de Statistique de l'Université de Paris* 8,
  229-231.
- Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer.
- Pinson, P., Madsen, H., Nielsen, H. A., Papaefthymiou, G., Klöckl, B.
  (2009). From probabilistic forecasts to statistical scenarios of
  short-term wind power production. *Wind Energy* 12(1), 51-62. (Gaussian
  copula scenarios from marginal forecasts; the method used here.)
- Høyland, K., Wallace, S. W. (2001). Generating scenario trees for
  multistage decision problems. *Management Science* 47(2), 295-307.
  (Moment matching.)
- Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. *Journal of the American Statistical
  Association* 102(477), 359-378. (Pinball loss, energy score.)
- Scheuerer, M., Hamill, T. M. (2015). Variogram-based proper scoring rules
  for probabilistic forecasts of multivariate quantities. *Monthly Weather
  Review* 143(4), 1321-1334.

*Stochastic programming and bidding*

- Birge, J. R., Louveaux, F. (2011). *Introduction to Stochastic
  Programming*, 2nd ed. Springer. (Two-stage programs, VSS, EVPI.)
- Fleten, S.-E., Pettersen, E. (2005). Constructing bidding curves for a
  price-taking retailer in the Norwegian electricity market. *IEEE
  Transactions on Power Systems* 20(2), 701-708.
