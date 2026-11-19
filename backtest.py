"""Chronological backtest of candidate models on the public data only.

For each cutoff C the raw tables are truncated to what was known before C
(transactions before C, subscription periods starting before C, renewals due
before C, spend before C). predict() refits from scratch on those tables, and
its forecasts for months C .. min(C+11, 2025-12) are scored against the panel
built from the full public data. No 2026 outcomes exist here or are used.

    uv run --frozen python backtest.py --data-dir /path/to/released/data
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

import main

CUTOFFS = ['2025-01-01', '2025-07-01', '2025-10-01']
LAST_OBSERVED = pd.Period('2025-12', 'M')
CANDIDATES = {
    'sbg_plain':             {'cohort_effects': False, 'cohort_trend': False, 'calendar_trend': False, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'sbg_cohort':            {'cohort_effects': False, 'cohort_trend': True,  'calendar_trend': False, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'sbg_calendar':          {'cohort_effects': False, 'cohort_trend': False, 'calendar_trend': True, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'sbg_cohort_calendar':   {'cohort_effects': False, 'cohort_trend': True,  'calendar_trend': True, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'sbg_cohort_calendar_r10': {'cohort_effects': False, 'cohort_trend': True, 'calendar_trend': True, 'trend_ridge': 10.0, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'sbg_effects2_calendar':  {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 2.0, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'sbg_effects5_calendar':  {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 5.0, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'sbg_effects20_calendar': {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'sbg_effects20_calendar_damp12': {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'calendar_damping_tau': 12.0, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'sbg_effects20_calendar_damp6': {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'calendar_damping_tau': 6.0, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'sbg_effects20_nocalendar': {'cohort_trend': False, 'calendar_trend': False, 'cohort_effects': True, 'effect_ridge': 20.0, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'struct_addon_age_cal':  {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'economics': 'structural', 'addon_terms': ('logage','age','cal')},
    'struct_addon_age':      {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'economics': 'structural', 'addon_terms': ('logage','age')},
    'struct_addon_logage':   {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'economics': 'structural', 'addon_terms': ('logage',)},
    'struct_upgrade_cal':    {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'economics': 'structural', 'addon_terms': ('logage','age','cal'), 'upgrade_terms': ('logage','cal')},
    'blend50_struct_age_cal': {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'economics': 'blend', 'blend_weight': 0.5, 'addon_terms': ('logage','age','cal')},
    'blend50_struct_age':    {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'economics': 'blend', 'blend_weight': 0.5, 'addon_terms': ('logage','age')},
    'struct2_lin_age':        {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'economics': 'structural', 'addon_link': 'linear', 'addon_terms': ('age',)},
    'struct2_lin_age_cal':    {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'economics': 'structural', 'addon_link': 'linear', 'addon_terms': ('age','cal')},
    'struct2_lin_logage_age': {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'economics': 'structural', 'addon_link': 'linear', 'addon_terms': ('logage','age')},
    'struct2_log_logage_age_cal': {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'economics': 'structural', 'addon_link': 'log', 'addon_terms': ('logage','age','cal')},
    'struct2_lin_age_cal_upcal': {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'economics': 'structural', 'addon_link': 'linear', 'addon_terms': ('age','cal'), 'upgrade_terms': ('logage','cal')},
    'rw5_loglinear':   {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_prior': 'rw', 'effect_ridge': 5.0, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'rw20_loglinear':  {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_prior': 'rw', 'effect_ridge': 20.0, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'rw50_loglinear':  {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_prior': 'rw', 'effect_ridge': 50.0, 'economics': 'loglinear', 'effect_prior': 'zero'},
    'rw5_struct':      {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_prior': 'rw', 'effect_ridge': 5.0, 'economics': 'structural', 'addon_link': 'linear', 'addon_terms': ('age','cal'), 'upgrade_terms': ('logage','cal')},
    'rw20_struct':     {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_prior': 'rw', 'effect_ridge': 20.0, 'economics': 'structural', 'addon_link': 'linear', 'addon_terms': ('age','cal'), 'upgrade_terms': ('logage','cal')},
    'rw50_struct':     {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_prior': 'rw', 'effect_ridge': 50.0, 'economics': 'structural', 'addon_link': 'linear', 'addon_terms': ('age','cal'), 'upgrade_terms': ('logage','cal')},
    'rw20_blend50':    {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_prior': 'rw', 'effect_ridge': 20.0, 'economics': 'blend', 'addon_link': 'linear', 'addon_terms': ('age','cal'), 'upgrade_terms': ('logage','cal'), 'blend_weight': 0.5},
    'ENS_orig+rw50struct': {'ensemble': [{'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'effect_prior': 'zero', 'economics': 'loglinear'}, {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 50.0, 'effect_prior': 'rw', 'economics': 'structural', 'addon_link': 'linear', 'addon_terms': ('age','cal'), 'upgrade_terms': ('logage','cal')}]},
    'ENS_orig+zero_struct': {'ensemble': [{'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'effect_prior': 'zero', 'economics': 'loglinear'}, {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'effect_prior': 'zero', 'economics': 'structural', 'addon_link': 'linear', 'addon_terms': ('age','cal'), 'upgrade_terms': ('logage','cal')}]},
    'ENS_orig+rw50struct+zero_struct': {'ensemble': [{'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'effect_prior': 'zero', 'economics': 'loglinear'}, {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 50.0, 'effect_prior': 'rw', 'economics': 'structural', 'addon_link': 'linear', 'addon_terms': ('age','cal'), 'upgrade_terms': ('logage','cal')}, {'cohort_trend': False, 'calendar_trend': True, 'cohort_effects': True, 'effect_ridge': 20.0, 'effect_prior': 'zero', 'economics': 'structural', 'addon_link': 'linear', 'addon_terms': ('age','cal'), 'upgrade_terms': ('logage','cal')}]},
    'sbg_trend_effects5_calendar': {'cohort_trend': True, 'calendar_trend': True, 'trend_ridge': 10.0, 'cohort_effects': True, 'effect_ridge': 5.0, 'economics': 'loglinear', 'effect_prior': 'zero'},
}


def truncate(tables, cutoff: pd.Timestamp):
    c = cutoff.tz_localize('UTC')
    t = {k: v.copy() for k, v in tables.items()}
    t['transactions'] = t['transactions'][pd.to_datetime(t['transactions'].occurred_at, utc=True) < c]
    sp = t['subscription_periods']
    t['subscription_periods'] = sp[pd.to_datetime(sp.period_start, utc=True) < c]
    cs = t['campaign_spend']
    t['campaign_spend'] = cs[pd.to_datetime(cs.spend_date, utc=True) < c]
    return t


def make_grid(panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    origin = cutoff.to_period('M') - 1
    last = panel[panel.calendar_month == origin.to_timestamp().strftime('%Y-%m-%d')]
    rows = []
    for r in last.itertuples(index=False):
        for step in range(1, 13):
            rows.append({'cohort_month': r.cohort_month, 'channel': r.channel, 'primary_need': r.primary_need,
                         'calendar_month': (origin + step).to_timestamp().strftime('%Y-%m-%d'),
                         'age_month': int(r.age_month) + step, 'n_acquired': int(r.n_acquired),
                         'origin_retained': int(r.n_retained)})
    return pd.DataFrame(rows).sort_values(main.KEYS).reset_index(drop=True)


def score(pred: pd.DataFrame, truth: pd.DataFrame, max_age: int) -> dict:
    j = pred.merge(truth, on=main.KEYS, how='inner', validate='one_to_one')
    tail = j[j.age_month > max_age]  # ages never observed in this fold's training data
    tail_cm = tail.groupby(['cohort_month', 'calendar_month'])[['predicted_contribution', 'contribution']].sum()
    cm = j.groupby(['cohort_month', 'calendar_month'])[['predicted_contribution', 'contribution']].sum()
    denom = cm.contribution.abs().sum()
    return {'contribution_wape': float((cm.predicted_contribution - cm.contribution).abs().sum()/denom),
            'retention_wape': float((j.predicted_retained - j.n_retained).abs().sum()/j.n_retained.sum()),
            'net_revenue_wape': float((j.predicted_net_revenue - j.net_revenue).abs().sum()/j.net_revenue.abs().sum()),
            'contribution_bias': float((cm.predicted_contribution.sum() - cm.contribution.sum())/denom),
            'tail_rows': int(len(tail)),
            'tail_contribution_wape': (float((tail_cm.predicted_contribution - tail_cm.contribution).abs().sum()/tail_cm.contribution.abs().sum())
                                       if len(tail) else float('nan')),
            'points': int(len(cm))}


def template_predict():
    path = Path(__file__).with_name('reference') / 'template_main.py'
    spec = importlib.util.spec_from_file_location('template_main', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.predict


def run(data_dir: Path) -> pd.DataFrame:
    contract = json.loads(Path(__file__).with_name('submission-contract.json').read_text())
    tables = main.load_tables(data_dir, contract)
    full = main.build_panel(tables, pd.Timestamp('2026-01-01'))
    truth = full.rename(columns={})[main.KEYS + ['n_retained', 'net_revenue', 'contribution']]
    models = {'template_constant_hazard': lambda t, g: template_predict()(t, g, seed=0)}
    models |= {name: (lambda cfg: lambda t, g: main.predict(t, g, seed=0, config={'ensemble': None} | cfg))(cfg) for name, cfg in CANDIDATES.items()}
    rows = []
    for cutoff in map(pd.Timestamp, CUTOFFS):
        t = truncate(tables, cutoff)
        tp = main.build_panel(t, cutoff)
        grid = make_grid(tp, cutoff)
        max_age = int(tp.age_month.max())
        horizon = min(12, LAST_OBSERVED.ordinal - (cutoff.to_period('M') - 1).ordinal)
        for name, fn in models.items():
            pred = main.validate_output(fn(t, grid.copy(deep=True)), grid)
            rows.append({'cutoff': cutoff.strftime('%Y-%m'), 'horizon_months': horizon, 'model': name} | score(pred, truth, max_age))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('validation/backtest_results.csv'))
    a = ap.parse_args()
    res = run(a.data_dir)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.output, index=False, float_format='%.6f')
    pd.set_option('display.width', 200)
    print(res.drop(columns=['net_revenue_wape']).round(4).to_string(index=False))
    print('\nUnweighted mean over cutoffs:')
    print(res.groupby('model')[['contribution_wape', 'retention_wape', 'contribution_bias']].mean().sort_values('contribution_wape').round(4).to_string())
    print('\nMean over cutoffs (12-month fold weighted 2x):')
    w = res.assign(w=np.where(res.cutoff == '2025-01', 2.0, 1.0))
    print(w.groupby('model').apply(lambda g: pd.Series({c: np.average(g[c], weights=g.w) for c in
          ['contribution_wape', 'retention_wape', 'net_revenue_wape', 'contribution_bias']}), include_groups=False)
          .sort_values('contribution_wape').round(4).to_string())
