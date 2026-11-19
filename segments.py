"""Which kinds of customers last longer and earn more? (public data only)

Not part of the scored predictor. Fits person-level models on the observed
history and writes validation/segment_insights.csv and segment_insights.md:

* churn: logistic hazard per member-month, adjusted for tenure and cohort
* upgrades: logistic hazard standard -> premium
* add-ons: Poisson rate of add-on purchases per paid member-month
* value: observed contribution per member-month, current members and the
  final model's 2026 contribution per current member, by segment

    uv run --frozen python segments.py --data-dir /path/to/released/data
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import main

SEGMENTS = {
    'primary_need': ('save_time', 'build_skill'),
    'channel': ('search', 'partner'),
    'plan_tier': ('premium', 'standard'),
    'age_band': ('18_34', '35_plus'),
    'household_type': ('solo', 'shared'),
    'region': ('west', 'east'),
}


def _fit(X, y, kind):
    A = X.to_numpy(float); b = np.zeros(A.shape[1])
    if kind == 'poisson': b[0] = np.log(y.mean())
    for _ in range(60):
        eta = A @ b
        if kind == 'logit':
            mu = 1/(1+np.exp(-eta)); W = mu*(1-mu)
        else:
            mu = np.exp(eta); W = mu
        H = A.T @ (A*W[:, None]) + 1e-8*np.eye(len(b)); s = np.linalg.solve(H, A.T @ (y-mu)); b += s
        if np.abs(s).max() < 1e-10: break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return pd.DataFrame({'coef': b, 'se': se}, index=X.columns)


def member_months(tables):
    pm, _ = main.person_months(tables, pd.Timestamp('2026-01-01'))
    sp = tables['subscription_periods'].copy()
    sp = sp[sp.renewal_outcome.isin(['paid', 'cancelled', 'failed'])]
    sp['month'] = pd.to_datetime(sp.period_start).dt.to_period('M')
    acq = main.acquisitions(tables)
    r = sp.merge(acq, on='person_id')
    r['age_month'] = main._ordinal(r.month) - main._ordinal(r.cohort)
    r = r.sort_values(['person_id', 'month'])
    r['plan_tier_prev'] = r.groupby('person_id').plan_tier.shift()
    r = r[r.age_month >= 1].copy()
    r['churn'] = (r.renewal_outcome != 'paid').astype(float)
    r['cohort_index'] = main._ordinal(r.cohort) - main.BASE
    ev = tables['product_events'].copy(); ev['month'] = pd.to_datetime(ev.occurred_at).dt.to_period('M')
    core = ev[ev.event_type == 'core_action'].groupby(['person_id', 'month']).size()
    idx = pd.MultiIndex.from_arrays([r.person_id, r.month - 1])
    r['core_prev'] = np.log1p(core.reindex(idx).fillna(0).to_numpy())
    return pm, r


def dummies(d, tier_col):
    X = pd.DataFrame({'const': 1.0}, index=d.index)
    for seg, (level, _) in SEGMENTS.items():
        col = tier_col if seg == 'plan_tier' else seg
        X[f'{seg}={level}'] = (d[col] == level).astype(float)
    return X


def run(data_dir: Path):
    contract = json.loads(Path(__file__).with_name('submission-contract.json').read_text())
    tables = main.load_tables(data_dir, contract)
    pm, r = member_months(tables)

    X = dummies(r, 'plan_tier_prev')
    X['log_tenure'] = np.log(r.age_month); X['cohort_years'] = (r.cohort_index-12)/12; X['core_actions_prev'] = r.core_prev
    churn = _fit(X, r.churn.to_numpy(), 'logit')
    up = pm[pm.prev_tier == 'standard']
    Xu = dummies(up, 'plan_tier').drop(columns=['plan_tier=premium'])
    Xu['log_tenure'] = np.log(up.age_month.clip(lower=1)); Xu['calendar_years'] = up.cal - 1
    upgrade = _fit(Xu, (up.plan_tier == 'premium').to_numpy(float), 'logit')
    Xa = dummies(pm, 'plan_tier'); Xa['tenure_years'] = pm.age_month/12; Xa['calendar_years'] = pm.cal - 1
    addon = _fit(Xa, pm.add_qty.to_numpy(float), 'poisson')

    grid = pd.read_csv(Path(__file__).with_name('prediction_grid.csv'))
    pred = main.predict(tables, grid, seed=contract['execution']['seed'])
    per_member = (pred.groupby(['cohort_month', 'channel', 'primary_need']).predicted_contribution.sum()
                  / pred.groupby(['cohort_month', 'channel', 'primary_need']).origin_retained.first()).replace([np.inf], 0).fillna(0)
    current = pm[pm.month == pm.month.max()].copy()
    current['forecast_2026'] = [per_member.get((c.to_timestamp().strftime('%Y-%m-%d'), ch, nd), 0.0)
                                for c, ch, nd in zip(current.cohort, current.channel, current.primary_need)]
    acq = main.acquisitions(tables)
    pm['contribution'] = pm.sub_rev + pm.add_rev - pm.refund_amount - pm.sub_cost - pm.add_cost

    rows = []
    for seg, levels in SEGMENTS.items():
        for level in levels:
            m, c = pm[pm[seg] == level], current[current[seg] == level]
            if seg == 'plan_tier':   # tier at acquisition (age 0) for the per-acquired-member view
                ids = pm[(pm.age_month == 0) & (pm.plan_tier == level)].person_id
            else:
                ids = acq[acq[seg] == level].person_id
            a_ = pm[pm.person_id.isin(ids)]
            row = {'segment': seg, 'level': level, 'acquired': len(ids),
                   'retained_share_dec2025_of_acquired': current.person_id.isin(ids).sum()/max(len(ids), 1),
                   'observed_contribution_per_acquired_member': a_.contribution.sum()/max(len(ids), 1),
                   'member_months': len(m), 'current_members_dec2025': len(c),
                   'observed_contribution_per_member_month': m.contribution.mean(),
                   'forecast_2026_contribution_per_current_member': c.forecast_2026.mean(),
                   'premium_share_of_member_months': (m.plan_tier == 'premium').mean(),
                   'addons_per_member_month': m.add_qty.mean()}
            key = f'{seg}={level}'
            for name, fit in (('churn_odds_ratio', churn), ('upgrade_odds_ratio', upgrade), ('addon_rate_ratio', addon)):
                if key in fit.index:
                    b, se = fit.loc[key]
                    row[name] = np.exp(b); row[name+'_lo'] = np.exp(b-1.96*se); row[name+'_hi'] = np.exp(b+1.96*se)
            rows.append(row)
    out = pd.DataFrame(rows)
    extra = pd.concat({'churn': churn, 'upgrade': upgrade, 'addon': addon}).reset_index().rename(columns={'level_0': 'model', 'level_1': 'term'})
    return out, extra


def markdown(out: pd.DataFrame) -> str:
    def ratio(r, name):
        if pd.isna(r.get(name)): return 'reference' if r['level'] == SEGMENTS[r['segment']][1] else '—'
        sig = '' if r[name+'_lo'] <= 1 <= r[name+'_hi'] else ' *'
        return f"{r[name]:.2f} ({r[name+'_lo']:.2f}–{r[name+'_hi']:.2f}){sig}"
    lines = ['| Segment | Level | Acquired | Still active Dec 2025 | Contribution to date / acquired member | Churn odds | Upgrade odds | Add-on rate | Contribution / active member-month | 2026 forecast / current member | Members Dec 2025 |',
             '|---|---|---:|---:|---:|---|---|---|---:|---:|---:|']
    for r in out.to_dict('records'):
        lines.append(f"| {r['segment']} | {r['level']} | {r['acquired']:,} | {r['retained_share_dec2025_of_acquired']:.1%} | ${r['observed_contribution_per_acquired_member']:.0f} | {ratio(r,'churn_odds_ratio')} | {ratio(r,'upgrade_odds_ratio')} | "
                     f"{ratio(r,'addon_rate_ratio')} | ${r['observed_contribution_per_member_month']:.2f} | "
                     f"${r['forecast_2026_contribution_per_current_member']:.0f} | {r['current_members_dec2025']:,} |")
    return '\n'.join(lines) + '\n\nRatios are adjusted (tenure, cohort/calendar, all segments together) with 95% intervals; * = interval excludes 1. The second level of each segment is the reference. Plan tier is the tier last month (churn), the current tier (add-ons) or the tier at acquisition (acquired-member columns). Observed columns run through December 2025; the 2026 forecast is the final model cohort x channel x need forecast per current member, so for tier and demographic rows it reflects their cell/cohort mix only. Associations are adjusted, not causal.\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    a = ap.parse_args()
    out, coefs = run(a.data_dir)
    Path('validation').mkdir(exist_ok=True)
    out.to_csv('validation/segment_insights.csv', index=False, float_format='%.4f')
    coefs.to_csv('validation/segment_model_coefficients.csv', index=False, float_format='%.5f')
    md = markdown(out); Path('validation/segment_insights.md').write_text(md); print(md)
    print(coefs.assign(ratio=np.exp(coefs.coef)).round(3).to_string(index=False))
