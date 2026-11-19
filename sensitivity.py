"""2026 forecast sensitivity of the final ensemble (public data only; not part of the predictor).

    uv run --frozen python sensitivity.py --data-dir /path/to/released/data
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import main


def tail_hazard(pred: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Scale the implied monthly hazard at ages 24-35 and rescale receipts per member accordingly."""
    out = []
    for _, g in pred.sort_values(main.KEYS).groupby(main.KEYS[:3]):
        r = g.predicted_retained.to_numpy(); r0 = float(g.origin_retained.iloc[0]); prev = np.r_[r0, r[:-1]]
        h = np.where(prev > 0, 1 - r/np.where(prev > 0, prev, 1), 0)
        h = np.where(g.age_month >= 24, np.minimum(h*factor, 1), h)
        nr = r0*np.cumprod(1-h); s = np.where(r > 0, nr/np.where(r > 0, r, 1), 0)
        g = g.copy(); g['predicted_retained'] = nr
        g['predicted_net_revenue'] *= s; g['predicted_contribution'] *= s
        out.append(g)
    return pd.concat(out)


def main_(data_dir: Path):
    contract = json.loads(Path(__file__).with_name('submission-contract.json').read_text())
    tables = main.load_tables(data_dir, contract)
    grid = pd.read_csv(Path(__file__).with_name('prediction_grid.csv'))
    L, S = main.MEMBER_LOGLINEAR, main.MEMBER_STRUCTURAL
    scenarios = {
        'final ensemble': {},
        'member: log-linear economics (previous submission)': {'ensemble': [L]},
        'member: structural economics': {'ensemble': [S]},
        'ensemble, no calendar trend in either member': {'ensemble': [L | {'calendar_trend': False},
                                                                      S | {'addon_terms': ('age',), 'upgrade_terms': ('logage',)}]},
        'ensemble, calendar trend damped (tau=12)': {'ensemble': [L | {'calendar_damping_tau': 12.0}, S]},
        'ensemble, cohort-tier costs off (12-month tier averages)': {'ensemble': [L, S | {'cohort_costs': False}]},
    }
    preds = {n: main.validate_output(main.predict(tables, grid, seed=contract['execution']['seed'], config=c), grid)
             for n, c in scenarios.items()}
    preds['final ensemble, hazard x0.8 at ages 24-35'] = tail_hazard(preds['final ensemble'], 0.8)
    preds['final ensemble, hazard x1.2 at ages 24-35'] = tail_hazard(preds['final ensemble'], 1.2)
    ref = Path(__file__).parent/'..'/'customer-economics-lab'/'outputs'/'instructor-batch-2026-11-05-run2'/'challenge'/'baseline_predictions.csv'
    rows = [{'scenario': n, 'contribution_2026': p.predicted_contribution.sum(), 'net_revenue_2026': p.predicted_net_revenue.sum(),
             'retained_dec2026': p[p.calendar_month == '2026-12-01'].predicted_retained.sum(),
             'contribution_age24plus': p[p.age_month >= 24].predicted_contribution.sum()} for n, p in preds.items()]
    out = pd.DataFrame(rows)
    out['vs_final_pct'] = 100*(out.contribution_2026/out.contribution_2026.iloc[0] - 1)
    if ref.exists():
        b = pd.read_csv(ref); f = preds['final ensemble']
        cm = lambda d: d.groupby(['cohort_month', 'calendar_month']).predicted_contribution.sum()
        print('released sBG+Gamma reference 2026 contribution:', round(b.predicted_contribution.sum()),
              '| cohort-month distance from final:', round(float((cm(f)-cm(b)).abs().sum()/cm(b).abs().sum()), 4))
    Path('validation').mkdir(exist_ok=True)
    out.to_csv('validation/sensitivity_2026.csv', index=False, float_format='%.2f')
    print(out.round(1).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    main_(ap.parse_args().data_dir)
