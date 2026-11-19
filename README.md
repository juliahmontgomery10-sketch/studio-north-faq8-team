# Studio North team forecast: sBG retention × structural customer economics (ensemble)

## Summary

This model predicts how much profit (contribution) Studio North's existing customers will generate in 2026: **$1,050,473** in total. It does this by forecasting, month by month, how many members stay subscribed, how many move up to the premium plan and how many add-ons they buy.

On past data it was tested by hiding the most recent months and checking its predictions against what actually happened. It missed by about **4–5%** on average, compared with 28–31% for the challenge's example model.

The best customers are people who joined to **build a skill** and came through a **partner**. About 64% of build_skill members are still active, and they have generated $237 each so far. Members who joined to **save time** cancel at twice the rate. Age, household type and region make no difference.

Entry for the Studio North LTV holdout challenge (FAQ 8). It forecasts expected retained members, net revenue and contribution for the 24 existing acquisition cohorts × 4 channel/need cells for January–December 2026 (1,152 grid rows), using only observations through December 31, 2025.

This target is **remaining contribution over the next twelve calendar months**, not the report's contribution LTV over each cohort's first twelve receipts. Every cohort gets twelve future months; their ages differ (1–12 for the December 2025 cohort, 24–35 for January 2024).

## Run

```text
uv sync --frozen
uv run --frozen python main.py --data-dir /path/to/released/data --prediction-grid prediction_grid.csv --output outputs/predictions.csv --seed 20261116
```

Python 3.12; dependencies are pandas, numpy and scipy (locked in `uv.lock`). Everything is fitted from the eight supplied CSVs at run time. There are no network calls, no cached fits and no randomness, so the seed is accepted but the output is deterministic. Fit plus prediction takes a few seconds and well under 1 GB of memory.

Analysis scripts that are not part of the predictor:
- `backtest.py`: chronological validation
- `sensitivity.py`: 2026 scenarios
- `segments.py`: customer-type report

## What the public data shows

These findings shaped the model. All of them come from the eight observed tables.

- **Prices are fixed.** Standard is $50 and premium $75 in every month. Add-ons are $9 for standard members and $12 for premium. Average revenue per member rises only because of **plan mix** and **add-ons**, not price increases.
- **Upgrades only go one way.** There were 1,407 standard → premium switches and no downgrades. About 10% of standard members upgrade each month. The premium share of active members rises from 20% at signup to about 78% by month 18. Later cohorts also join as premium more often.
- **Subscription variable cost is fixed per acquisition cohort × tier.** It has zero variation within a cohort/tier and falls exactly $0.15 per join month (premium $40.50 for January 2024, $40.35 for February 2024, …).
- **Add-on purchases rise roughly linearly with tenure.** About 0.32 per member-month at signup, about 0.94 at months 18–23. A member's own early add-on rate does not predict their later rate (correlation 0.03).
- **Churn falls with tenure and is much lower for later cohorts.** The odds fall about 29% per cohort-year. Save_time members churn at **2.0×** the odds of build_skill members, and search-acquired members at **1.2×** partner-acquired. Once those are accounted for, plan tier, age band, household type, region, product engagement, support contacts and feedback ratings have **no significant effect** on churn (see `validation/segment_insights.md`).

## Model

The final prediction is the **equal-weight average of two members** that share the same structure (sBG retention per channel × need cell, anchored to each cell's observed origin survivors) but use different churn priors and different revenue logic. Averaging two forecasts that each satisfy the contract keeps the result within the contract: membership stays monotone, never exceeds `origin_retained`, and contribution stays ≤ net revenue.

**Retention (both members).** Shifted-Beta-Geometric: the churn probability at age *t* for a member alive at *t−1* is α/(α+β+t−1), with separate α and β for each channel × need cell. Each acquisition cohort gets an offset in log β. It is fitted by maximum likelihood on all observed monthly survivor transitions (binomial likelihood, L-BFGS-B, deterministic multistart). A cell's forecast is R(a) = R_origin · S(a)/S(a₀), where S(t) = B(α, β+t)/B(α, β). The two members differ only in how cohort offsets are shrunk:
- *Member A:* ridge toward zero (λ = 20).
- *Member B:* random-walk smoothing across neighbouring cohorts (λ = 50). A newly acquired cohort with no renewals yet inherits its neighbours' level, rather than the average of all cohorts.

**Economics, member A (log-linear).** Retained-member-weighted least squares, with Duan smearing to return to dollar scale:
- log(net revenue per retained member) ~ cell + log(age) + calendar years
- log(contribution ÷ net revenue) ~ cell + log(age) + calendar years

**Economics, member B (structural, built from the mechanics above).**
- Each cell-cohort's **observed premium share at December 2025** is rolled forward month by month with a fitted standard → premium upgrade hazard (logistic in log tenure and calendar time).
- **Subscription price and variable cost per tier** are the cohort's observed values. Any cohort/tier combination not yet observed uses a per-tier linear fit in cohort index.
- **Add-on demand per member-month** is linear in tenure and calendar time, with need and channel effects. It is priced at the tier's observed unit price and unit cost.
- A recent-12-month refund ratio is deducted.
- Expected contribution = members × (premium share × premium margin + standard share × standard margin).

Acquisition cost is not subtracted in either member.

## Chronological validation (public data only)

`backtest.py` truncates the **raw tables** at each cutoff C: transactions before C, subscription periods starting before C, spend before C. Renewals are due on the day their period starts, so the truncated panel is identical to the full-history panel for all pre-cutoff months (checked: 0 differing rows). Each model is refit from scratch, forecasts 12 months, and is scored where outcomes exist (through December 2025) using the challenge's cohort × calendar-month contribution WAPE. "Tail" rows are ages older than any age observed in that fold's training data. This is the closest available check of extrapolation beyond the observed age range, which the oldest 2026 cohorts require (ages 24–35, maximum observed 23).

| Model | 2025-01 (12 mo) | 2025-07 (6 mo) | 2025-10 (3 mo) | Mean | Mean, 12-mo fold ×2 | Tail WAPE, 12-mo fold |
|---|---:|---:|---:|---:|---:|---:|
| template constant hazard | 0.379 | 0.270 | 0.207 | 0.285 | 0.309 | 0.526 |
| sBG, no cohort/calendar terms | 0.084 | 0.107 | 0.098 | 0.097 | 0.093 | 0.097 |
| sBG + log-linear calendar economics | 0.052 | 0.049 | 0.043 | 0.048 | 0.049 | 0.061 |
| member A (previous submission) | 0.050 | 0.042 | 0.040 | 0.044 | 0.045 | 0.059 |
| member B (structural) | 0.055 | 0.039 | 0.038 | 0.044 | 0.047 | 0.070 |
| **final ensemble (A + B)** | **0.051** | **0.039** | **0.037** | **0.042** | **0.045** | **0.063** |

The ensemble has the lowest mean under both weightings, and its signed contribution bias is −0.03% (double-weighted). The margin over member A is small: 0.0014 unweighted, 0.0006 double-weighted.

The main gain is robustness. The two members reach nearly the same 2026 total by different routes, and the ensemble depends much less on extrapolating a calendar trend (see below).

Full results for all candidates are in `validation/backtest_results.csv` and `validation/backtest_log.txt`. Rejected alternatives:
- **Linear cohort trend in sBG parameters:** 0.089 on the 12-month fold.
- **Calendar trend damped over the forecast year (τ = 12 or 6 months):** helped the 12-month fold, hurt the others.
- **No calendar trend:** about twice the error.
- **Log-link add-on models:** overshoot at long tenure.
- **Structural economics with 12-month tier average costs instead of cohort-tier costs:** biased low.
- **Three-member ensembles.**

The folds overlap and only the first covers a full 12-month horizon. These are historical validation results, not private-holdout scores.

## 2026 forecast and sensitivity

`validation/sensitivity_2026.csv`:

| Scenario | 2026 contribution | vs final |
|---|---:|---:|
| **Final ensemble** | **$1,050,473** | — |
| Member A alone (previous submission) | $1,050,104 | −0.0% |
| Member B alone (structural) | $1,050,842 | +0.0% |
| Ensemble, no calendar trend in either member | $989,213 | −5.8% |
| Ensemble, calendar trend damped, τ = 12 | $1,042,353 | −0.8% |
| Ensemble, cohort-tier costs off | $1,048,156 | −0.2% |
| Hazard ×0.8 at ages 24–35 | $1,053,111 | +0.3% |
| Hazard ×1.2 at ages 24–35 | $1,047,885 | −0.2% |

For comparison, the released hierarchical sBG + Gamma reference forecasts $1,008,152; the cohort-month distance between the two forecasts is 5.5%.

**Main remaining assumption:** add-on demand and upgrade rates keep their 2024–2025 calendar drift through 2026. Removing every calendar term would lower the total by 5.8%; for member A alone, that figure was 9.9%. The extrapolation for ages 24–35 has little effect on the total.

## Which kinds of customers do better (`validation/segment_insights.md`)

The submission grid is fixed by the contract (cohort × channel × need), so segments cannot be added as extra rows. Instead they are used inside the model where the backtests support it (need, channel and plan tier) and reported separately here. Ratios are adjusted for tenure, cohort/calendar and all other segments at once. They are associations, not causal effects.

| Customer type | Still active Dec 2025 | Contribution to date per acquired member | Churn odds | Add-on rate | 2026 forecast per current member |
|---|---:|---:|---:|---:|---:|
| build_skill | 64.0% | $237 | reference | reference | $383 |
| save_time | 42.3% | $158 | **2.01×** | **0.78×** | $302 |
| partner | 58.5% | $211 | reference | reference | $365 |
| search | 48.3% | $184 | **1.20×** | 1.01× | $335 |
| joined premium | 59.0% | $252 | 0.96× (tier last month) | 1.00× | — |
| joined standard | 51.2% | $182 | reference | reference | — |
| age 18–34 / 35+, solo / shared, west / east | 51.5–53.9% | $194–$197 | no significant difference | no significant difference | $348–$351 |

- **Who lasts longest and is worth most:** build_skill members acquired through partners.
- **Who costs the business:** save_time members churn at twice the odds and buy fewer add-ons. Search-acquired members churn more and upgrade less (0.88×).
- **Premium vs standard:** premium is worth more because of price and margin (contribution $39.71 vs $22.23 per active member-month). Premium members are *not* less likely to churn once tenure is accounted for.
- **Demographics** (age band, household type, region) don't distinguish customers on any outcome.

## Final configuration

`CONFIG['ensemble'] = [MEMBER_LOGLINEAR, MEMBER_STRUCTURAL]` in `main.py`. Seed 20261116.

Reproducibility, format validation and the prediction hash are recorded in `validation/final_run.json`. Passing format validation says nothing about private-holdout accuracy.

`reference/template_main.py` is the challenge's unmodified constant-hazard example. It is used only as a backtest comparator. Model design and review used UNC Sol (gpt-6.1-sol) through the course's UNC API bridge (three project calls, admitted by Jev); the submitted predictor makes no model or network calls.
