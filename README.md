# Studio North team forecast: sBG with shrunk cohort offsets + log-linear unit economics

Entry for the Studio North LTV holdout challenge (FAQ 8). It forecasts expected retained members, net revenue and contribution for the 24 existing acquisition cohorts × 4 channel/need cells for January–December 2026 (1,152 grid rows), using only observations through December 31, 2025.

This target is **remaining contribution over the next twelve calendar months**, not the report's contribution LTV over each cohort's first twelve receipts. Every cohort gets twelve future months; their ages differ (1–12 for the December 2025 cohort, 24–35 for January 2024).

## Run

```text
uv sync --frozen
uv run --frozen python main.py --data-dir /path/to/released/data --prediction-grid prediction_grid.csv --output outputs/predictions.csv --seed 20261116
```

Python 3.12; dependencies are pandas, numpy and scipy (locked in `uv.lock`). Everything is fitted from the eight supplied CSVs at run time. There are no network calls, no cached fits and no randomness, so the seed is accepted but the output is deterministic. Fit plus prediction takes about 1 second and about 270 MB of memory on a laptop CPU.

## Model

**Observed panel.** `build_panel()` is a pandas port of the lab's `sql/01_views.sql` `cohort_age_summary`. It covers first paid payment, channel attribution, retained = a paid subscription period starting in the month with renewal due before the data cutoff, and monthly net revenue and variable cost. On the public data it reproduces the lab report's table on all 1,200 rows. Differences are below $10⁻¹².

**Retention.** Shifted-Beta-Geometric per channel × need cell. The churn probability at age *t* for a member alive at *t−1* is α/(α+β+t−1). Each acquisition cohort gets its own offset in log β, shrunk toward zero with a ridge penalty of 20. It is fitted by maximum likelihood on all observed monthly survivor transitions (binomial likelihood, L-BFGS-B, deterministic multistart). A cell's forecast is anchored to its observed origin survivors: R(a) = R_origin · S(a)/S(a₀), where S(t) = B(α, β+t)/B(α, β). Expected membership is therefore monotone, never exceeds `origin_retained`, and is exactly zero for extinct cells.

Why these choices:
- The hazard declines with age within cohorts, which is the sBG selection pattern: survivors are increasingly low-churn members.
- Later cohorts churn much less at the same age. First-month churn is about 0.21 for 2024Q1 cohorts and about 0.10 for 2025Q3. Part of this is a shift in channel/need mix, but it also holds within cells.
- A linear cohort trend extrapolated poorly in the 12-month backtest. Shrunk per-cohort offsets let each *existing* cohort follow its own observed churn level instead.

**Unit economics.** Two weighted least-squares regressions on observed cell-months (age ≥ 1), weighted by retained members, with Duan smearing to return to dollar scale:
- log(net revenue per retained member) ~ cell + log(age) + calendar years
- log(contribution ÷ net revenue) ~ cell + log(age) + calendar years, with predicted margin capped at 1

Net revenue = retained × rate; contribution = net revenue × margin, so contribution ≤ net revenue. Acquisition cost is not subtracted. Per-member contribution rises with age (selection toward heavier users) and with calendar time, by about 9% a year at the same age.

## Chronological validation (public data only)

`backtest.py` truncates the **raw tables** at each cutoff C: transactions before C, subscription periods starting before C, spend before C. Renewals are due on the day their period starts, so the truncated panel is identical to the full-history panel for all pre-cutoff months (checked: 0 differing rows). Each model is refit from scratch, forecasts 12 months, and is scored where outcomes exist (through December 2025) using the challenge's cohort × calendar-month contribution WAPE. "Tail" rows are ages older than any age observed in that fold's training data. This is the closest available check of extrapolation beyond the observed age range, which the oldest 2026 cohorts require (ages 24–35, maximum observed 23).

| Cutoff | Horizon | Model | Contribution WAPE | Retention WAPE | Tail contribution WAPE |
|---|---:|---|---:|---:|---:|
| 2025-01 | 12 | template constant hazard | 0.379 | 0.239 | 0.526 |
| 2025-01 | 12 | sBG, no cohort/calendar terms | 0.084 | 0.088 | 0.097 |
| 2025-01 | 12 | sBG + calendar economics | 0.052 | 0.088 | 0.061 |
| 2025-01 | 12 | **final** | **0.050** | **0.086** | **0.059** |
| 2025-07 | 6 | template constant hazard | 0.270 | 0.141 | 0.422 |
| 2025-07 | 6 | sBG + calendar economics | 0.049 | 0.072 | 0.048 |
| 2025-07 | 6 | **final** | **0.042** | **0.065** | **0.048** |
| 2025-10 | 3 | template constant hazard | 0.207 | 0.085 | 0.338 |
| 2025-10 | 3 | sBG + calendar economics | 0.043 | 0.053 | 0.055 |
| 2025-10 | 3 | **final** | **0.040** | **0.050** | **0.056** |

Full results for all 14 candidates are in `validation/backtest_results.csv`. Rejected alternatives:
- **Linear cohort trend in sBG parameters:** 0.089 on the 12-month fold.
- **Calendar trend damped over the forecast year (τ = 12 or 6 months):** slightly better on the 12-month fold, worse on the other two.
- **No calendar trend:** about twice the error.
- **Ridge 2 and 5 on the cohort offsets:** both did slightly worse than 20.

The folds overlap and only the first covers a full 12-month horizon. These are historical validation results, not private-holdout scores.

## 2026 forecast and sensitivity

`validation/sensitivity_2026.csv`:

| Scenario | 2026 contribution | vs final |
|---|---:|---:|
| **Final** | **$1,050,104** | — |
| Calendar trend damped, τ = 12 | $1,033,864 | −1.5% |
| No calendar trend | $946,465 | −9.9% |
| Linear cohort trend instead of offsets | $1,089,984 | +3.8% |
| Hazard ×0.8 at ages 24–35 | $1,052,774 | +0.3% |
| Hazard ×1.2 at ages 24–35 | $1,047,485 | −0.2% |

For comparison, the released hierarchical sBG + Gamma reference forecasts $1,008,152; the cohort-month distance between the two forecasts is 5.8%. Forecast January 2026 contribution ($97.2k) continues smoothly from observed December 2025 ($99.5k).

**Main assumption and risk:** per-member economics keep rising on their 2024–2025 calendar trend through 2026. Removing that trend would lower the forecast by about 10%. The extrapolation for ages 24–35 has little effect on the total.

## Final configuration

`CONFIG` in `main.py`: `cohort_effects=True, effect_ridge=20, cohort_trend=False, calendar_trend=True, calendar_damping_tau=None`. Seed 20261116.

Reproducibility: two runs in a clean `uv sync --frozen` environment produced byte-identical predictions (SHA-256 `5e0639ead34cafd85e736f18c8db1ed157514464bb6d7d8c81a8b19cbb57d7c7`). The output passed the instructor's public `cohort_lab.challenge.validate_predictions` against the grid rebuilt from the report: 1,152 rows, no future outcomes read. Passing format validation says nothing about private-holdout accuracy.

`reference/template_main.py` is the challenge's unmodified constant-hazard example. It is used only as a backtest comparator. Model design and review used UNC Sol (gpt-6.1-sol) through the course's UNC API bridge; the submitted predictor makes no model or network calls.
