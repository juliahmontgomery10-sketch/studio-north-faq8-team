"""Studio North team forecast: cohort-trend sBG retention + log-linear unit economics.

Retention: shifted-Beta-Geometric (sBG) per channel x need cell, plus a
ridge-shrunk offset in log(beta) for each acquisition cohort so each existing
cohort's churn level follows its own observed history. Fitted by maximum likelihood on observed monthly survivor
transitions. Each cell's forecast is anchored to its observed origin survivors.

Unit economics: expected net revenue and contribution per retained member, each
log-linear in log(age), segment and (optionally) calendar time, fitted by
retained-member-weighted least squares on observed cell-months.

Everything is fitted from the supplied observed tables at run time with numpy,
pandas and scipy. No network, cached fits or future outcomes are used. The
model is deterministic; `seed` is accepted for the common interface.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln

KEYS = ['cohort_month','channel','primary_need','calendar_month','age_month']
EXPOSURES = ['n_acquired','origin_retained']
PREDICTIONS = ['predicted_retained','predicted_net_revenue','predicted_contribution']
CELL = ['channel','primary_need']

# Final configuration (chosen by chronological backtest; see README).
CONFIG = {'cohort_trend': False, 'calendar_trend': True, 'trend_ridge': 1.0, 'cohort_effects': True, 'effect_ridge': 20.0,
          'calendar_damping_tau': None, 'economics': 'loglinear', 'blend_weight': 0.5,
          'addon_terms': ('logage', 'age', 'cal'), 'upgrade_terms': ('logage',),
          'addon_link': 'log', 'cohort_costs': True, 'effect_prior': 'zero', 'ensemble': None}
# Final submission: equal-weight ensemble of the two backtest-selected specifications.
MEMBER_LOGLINEAR = {'effect_prior': 'zero', 'effect_ridge': 20.0, 'economics': 'loglinear'}
MEMBER_STRUCTURAL = {'effect_prior': 'rw', 'effect_ridge': 50.0, 'economics': 'structural', 'addon_link': 'linear',
                     'addon_terms': ('age', 'cal'), 'upgrade_terms': ('logage', 'cal')}
CONFIG['ensemble'] = [MEMBER_LOGLINEAR, MEMBER_STRUCTURAL]


# ---------------------------------------------------------------- observed panel
def build_panel(tables: dict[str,pd.DataFrame], data_as_of: pd.Timestamp) -> pd.DataFrame:
    """pandas port of the lab's sql/01_views.sql cohort_age_summary.

    One row per cohort_month x channel x primary_need x observed age, with
    n_acquired, n_retained, net_revenue, variable_cost and contribution.
    """
    tx = tables['transactions'].copy()
    tx['occurred_at'] = pd.to_datetime(tx.occurred_at, utc=True)
    for c in ['gross_amount','discount_amount','refund_amount','variable_cost']:
        tx[c] = tx[c].astype(float).fillna(0.0)
    pay = tx[(tx.transaction_type=='payment') & (tx.gross_amount-tx.discount_amount>0)]
    first = (pay.sort_values(['person_id','occurred_at','transaction_id'])
                .drop_duplicates('person_id')[['person_id','occurred_at']])
    camp = (tables['campaign_spend'].groupby('campaign_id', as_index=False).channel.min())
    acq = (first.merge(tables['people'][['person_id','primary_need']], on='person_id')
                .merge(tables['acquisition_attribution'][['person_id','credited_campaign_id','source_type']], on='person_id')
                .merge(camp, left_on='credited_campaign_id', right_on='campaign_id', how='left'))
    acq['channel'] = acq.channel.fillna(acq.source_type)
    acq['cohort'] = acq.occurred_at.dt.tz_localize(None).dt.to_period('M')
    acq = acq[['person_id','cohort','channel','primary_need']]

    tx['month'] = tx.occurred_at.dt.tz_localize(None).dt.to_period('M')
    tx['net'] = tx.gross_amount - tx.discount_amount - tx.refund_amount
    fin = tx.groupby(['person_id','month'], as_index=False)[['net','variable_cost']].sum()

    sp = tables['subscription_periods'].copy()
    due = pd.to_datetime(sp.renewal_due_at, utc=True)
    sp = sp[due < data_as_of.tz_localize('UTC')]
    sp = sp[sp.renewal_outcome=='paid']
    sp['month'] = pd.to_datetime(sp.period_start).dt.to_period('M')
    ret = sp[['person_id','month']].drop_duplicates().assign(retained=1)

    last = (data_as_of.to_period('M') - 1)
    acq = acq[acq.cohort <= last]
    n_months = (last.ordinal - acq.cohort.map(lambda p: p.ordinal)).to_numpy() + 1
    rep = acq.loc[acq.index.repeat(n_months)].copy()
    rep['age_month'] = rep.groupby('person_id').cumcount()
    rep['month'] = [c + a for c, a in zip(rep.cohort, rep.age_month)]
    rep = rep.merge(ret, on=['person_id','month'], how='left').merge(fin, on=['person_id','month'], how='left')
    rep[['retained','net','variable_cost']] = rep[['retained','net','variable_cost']].fillna(0)
    out = (rep.groupby(['cohort','channel','primary_need','age_month','month'], as_index=False)
              .agg(n_acquired=('person_id','size'), n_retained=('retained','sum'),
                   net_revenue=('net','sum'), variable_cost=('variable_cost','sum')))
    out['contribution'] = out.net_revenue - out.variable_cost
    out['cohort_month'] = out.cohort.dt.to_timestamp().dt.strftime('%Y-%m-%d')
    out['calendar_month'] = out.month.dt.to_timestamp().dt.strftime('%Y-%m-%d')
    out['cohort_index'] = out.cohort.map(lambda p: p.ordinal) - pd.Period('2024-01', 'M').ordinal
    return out.drop(columns=['cohort','month']).sort_values(['cohort_month','channel','primary_need','age_month']).reset_index(drop=True)


# ---------------------------------------------------------------- retention
def _log_surv(la, lb, t):
    a, b = np.exp(la), np.exp(lb)
    return betaln(a, b + t) - betaln(a, b)


def fit_retention(panel: pd.DataFrame, cohort_trend: bool, ridge: float,
                  cohort_effects: bool = False, effect_ridge: float = 5.0, effect_prior: str = 'zero') -> dict:
    p = panel.sort_values(['cohort_month','channel','primary_need','age_month']).copy()
    p['prev'] = p.groupby(['cohort_month','channel','primary_need']).n_retained.shift()
    tr = p[p.prev.notna() & (p.prev > 0)]
    cells = sorted(map(tuple, panel[CELL].drop_duplicates().to_numpy()))
    cidx = {c: i for i, c in enumerate(cells)}
    ci = np.array([cidx[(a, b)] for a, b in zip(tr.channel, tr.primary_need)])
    k = tr.cohort_index.to_numpy(float)
    kc = float(panel.cohort_index.mean())
    t = tr.age_month.to_numpy(float)
    prev, lost = tr.prev.to_numpy(float), (tr.prev - tr.n_retained).to_numpy(float)
    nc = len(cells)
    cohorts = sorted(panel.cohort_index.unique().tolist())
    kidx = {c: i for i, c in enumerate(cohorts)}
    ki = np.array([kidx[c] for c in tr.cohort_index])
    nt = 2 if cohort_trend else 0
    ne = len(cohorts) if cohort_effects else 0

    def unpack(x):
        la = x[:nc][ci]; lb = x[nc:2*nc][ci]
        if cohort_trend:
            la = la + x[2*nc]*(k-kc)/12; lb = lb + x[2*nc+1]*(k-kc)/12
        if cohort_effects:  # shrunk per-cohort shift in log(beta): churn level of each cohort
            lb = lb + x[2*nc+nt:][ki]
        return la, lb

    def nll(x):
        la, lb = unpack(x)
        a, b = np.exp(la), np.exp(lb)
        h = np.clip(a/(a+b+t-1), 1e-10, 1-1e-10)  # P(churn at age t | alive at t-1)
        u = x[2*nc+nt:]
        if effect_prior == 'rw' and len(u) > 1:   # smooth neighbouring cohorts; newest inherits its neighbours
            pen_u = effect_ridge*np.sum(np.diff(u)**2) + 0.1*np.sum(u**2)
        else:
            pen_u = effect_ridge*np.sum(u**2)
        pen = ridge*np.sum(x[2*nc:2*nc+nt]**2) + pen_u
        return -np.sum(lost*np.log(h) + (prev-lost)*np.log1p(-h)) + pen

    x0 = np.r_[np.zeros(nc), np.log(4.0)*np.ones(nc), np.zeros(nt + ne)]
    best = None
    for shift in (0.0, -1.0, 1.0):  # deterministic multistart
        r = minimize(nll, x0 + np.r_[np.full(2*nc, shift), np.zeros(len(x0)-2*nc)], method='L-BFGS-B',
                     bounds=[(-8, 8)]*len(x0))
        if best is None or r.fun < best.fun:
            best = r
    return {'x': best.x, 'cells': cidx, 'nc': nc, 'kc': kc, 'cohort_trend': cohort_trend,
            'nt': nt, 'cohorts': kidx if cohort_effects else None}


def retention_params(fit, cell, cohort_index):
    x, nc, i = fit['x'], fit['nc'], fit['cells'][cell]
    la, lb = x[i], x[nc+i]
    if fit['cohort_trend']:
        la += x[2*nc]*(cohort_index-fit['kc'])/12; lb += x[2*nc+1]*(cohort_index-fit['kc'])/12
    if fit['cohorts'] is not None:
        lb += x[2*nc+fit['nt']+fit['cohorts'][cohort_index]]
    return la, lb


# ---------------------------------------------------------------- unit economics
def fit_economics(panel: pd.DataFrame, calendar_trend: bool) -> dict:
    d = panel[(panel.age_month >= 1) & (panel.n_retained > 0)].copy()
    d['cal'] = (pd.to_datetime(d.calendar_month).dt.to_period('M').map(lambda p: p.ordinal)
                - pd.Period('2024-01', 'M').ordinal)/12
    cells = sorted(map(tuple, panel[CELL].drop_duplicates().to_numpy()))

    def design(frame):
        cols = [np.array([(a, b) == c for a, b in zip(frame.channel, frame.primary_need)], float) for c in cells]
        cols.append(np.log(frame.age_month.to_numpy(float)))
        if calendar_trend:
            cols.append(frame.cal.to_numpy(float))
        return np.column_stack(cols)

    X, w = design(d), np.sqrt(d.n_retained.to_numpy(float))
    coefs = {}
    for target, col in (('net', 'net_revenue'), ('margin', None)):
        if target == 'net':
            y = np.log(np.maximum(d.net_revenue/d.n_retained, 1e-3))
        else:  # contribution share of net revenue, logit-free: log of (contribution/net)
            y = np.log(np.clip(d.contribution/np.maximum(d.net_revenue, 1e-6), 1e-3, 1.0))
        coefs[target] = np.linalg.lstsq(X*w[:, None], y*w, rcond=None)[0]
        resid = y - X @ coefs[target]
        coefs[target+'_smear'] = float(np.average(np.exp(resid), weights=d.n_retained))  # Duan smearing
    return {'coefs': coefs, 'design': design}


# ---------------------------------------------------------------- structural economics
BASE = pd.Period('2024-01', 'M').ordinal


def _ordinal(periods) -> np.ndarray:
    return np.array([p.ordinal for p in periods], dtype=float)


def acquisitions(tables: dict[str,pd.DataFrame]) -> pd.DataFrame:
    """Same acquisition rule as build_panel/sql: first positive payment, credited channel."""
    tx = tables['transactions']
    pay = tx[(tx.transaction_type=='payment') & (tx.gross_amount.astype(float)-tx.discount_amount.astype(float).fillna(0)>0)].copy()
    pay['t'] = pd.to_datetime(pay.occurred_at, utc=True)
    first = pay.sort_values(['person_id','t','transaction_id']).drop_duplicates('person_id')[['person_id','t']]
    camp = tables['campaign_spend'].groupby('campaign_id', as_index=False).channel.min()
    acq = (first.merge(tables['people'][['person_id','primary_need','age_band','household_type','region']], on='person_id')
                .merge(tables['acquisition_attribution'][['person_id','credited_campaign_id','source_type']], on='person_id')
                .merge(camp, left_on='credited_campaign_id', right_on='campaign_id', how='left'))
    acq['channel'] = acq.channel.fillna(acq.source_type)
    acq['cohort'] = acq.t.dt.tz_localize(None).dt.to_period('M')
    return acq[['person_id','cohort','channel','primary_need','age_band','household_type','region']]


def person_months(tables: dict[str,pd.DataFrame], data_as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per member-month with a paid subscription period (= retained), with plan tier,
    previous paid tier and that month's subscription / add-on money."""
    acq = acquisitions(tables)
    sp = tables['subscription_periods']
    sp = sp[(pd.to_datetime(sp.renewal_due_at, utc=True) < data_as_of.tz_localize('UTC')) & (sp.renewal_outcome=='paid')].copy()
    sp['month'] = pd.to_datetime(sp.period_start).dt.to_period('M')
    pm = sp[['person_id','period_id','month','plan_tier']].merge(acq, on='person_id')
    pm = pm[pm.month >= pm.cohort].sort_values(['person_id','month'])
    pm['age_month'] = _ordinal(pm.month) - _ordinal(pm.cohort)
    pm['cal'] = (_ordinal(pm.month) - BASE)/12
    pm['cohort_index'] = _ordinal(pm.cohort) - BASE
    pm['prev_tier'] = pm.groupby('person_id').plan_tier.shift()
    tx = tables['transactions'].copy()
    tx = tx[pd.to_datetime(tx.occurred_at, utc=True) < data_as_of.tz_localize('UTC')]
    tx['month'] = pd.to_datetime(tx.occurred_at, utc=True).dt.tz_localize(None).dt.to_period('M')
    for c in ['gross_amount','discount_amount','refund_amount','variable_cost']:
        tx[c] = tx[c].astype(float).fillna(0.0)
    tx['rev'] = tx.gross_amount - tx.discount_amount
    pay = tx[tx.transaction_type=='payment']
    agg = lambda d, p: d.groupby(['person_id','month']).agg(**{p+'rev':('rev','sum'), p+'cost':('variable_cost','sum'),
                                                               p+'qty':('quantity','sum')})
    money = agg(pay[pay.charge_type=='subscription'], 'sub_').join(agg(pay[pay.charge_type=='add_on'], 'add_'), how='outer')
    money = money.join(tx.groupby(['person_id','month']).refund_amount.sum(), how='outer').fillna(0).reset_index()
    pm = pm.merge(money, on=['person_id','month'], how='left').fillna({c: 0.0 for c in money.columns if c not in ('person_id','month')})
    return pm, money


def _logit_fit(X, y, ridge=1e-3):
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1/(1+np.exp(-X@b)); H = X.T@(X*(p*(1-p))[:,None]) + ridge*np.eye(len(b))
        s = np.linalg.solve(H, X.T@(y-p) - ridge*b); b += s
        if np.abs(s).max() < 1e-10: break
    return b


def _poisson_fit(X, y, ridge=1e-3):
    b = np.zeros(X.shape[1]); b[0] = np.log(max(y.mean(), 1e-6))
    for _ in range(60):
        mu = np.exp(X@b); H = X.T@(X*mu[:,None]) + ridge*np.eye(len(b))
        s = np.linalg.solve(H, X.T@(y-mu) - ridge*b); b += s
        if np.abs(s).max() < 1e-10: break
    return b


def _segment_design(frame, terms):
    cols = [np.ones(len(frame)), (frame.primary_need=='save_time').to_numpy(float), (frame.channel=='search').to_numpy(float)]
    age = np.maximum(frame.age_month.to_numpy(float), 1.0)
    if 'logage' in terms: cols.append(np.log(age))
    if 'age' in terms: cols.append(age/12)
    if 'cal' in terms: cols.append(frame.cal.to_numpy(float) - 1.0)
    return np.column_stack(cols)


def fit_structural(tables, origin: pd.Period, cfg: dict) -> dict:
    pm, money = person_months(tables, (origin+1).to_timestamp())
    window = pm[pm.month > origin - 12]          # recent 12 months for prices and unit costs
    rates = {}
    for tier in ('standard', 'premium'):
        w = window[window.plan_tier==tier]
        q = max(w.add_qty.sum(), 1.0)
        rates[tier] = {'sub_rev': w.sub_rev.mean(), 'sub_cost': w.sub_cost.mean(),
                       'add_rev_unit': w.add_rev.sum()/q, 'add_cost_unit': w.add_cost.sum()/q}
    recent = money[money.month > origin - 12]
    # subscription price and variable cost are fixed per acquisition cohort x tier in the data:
    # use each cohort's observed value, else a per-tier linear fit in cohort index
    sub = pm[pm.sub_rev > 0]
    rates['by_cohort'] = {}
    for tier in ('standard', 'premium'):
        d = sub[sub.plan_tier==tier]
        cm = d.groupby('cohort_index')[['sub_rev','sub_cost']].mean()
        X = np.column_stack([np.ones(len(cm)), cm.index.to_numpy(float)])
        fit = {c: np.linalg.lstsq(X, cm[c].to_numpy(), rcond=None)[0] if len(cm) >= 2 else np.array([rates[tier][c], 0.0])
               for c in ('sub_rev','sub_cost')}
        rates['by_cohort'][tier] = {'obs': cm.to_dict('index'), 'fit': fit}
    rates['refund_ratio'] = float(recent.refund_amount.sum()/max((recent.sub_rev+recent.add_rev).sum(), 1.0))
    # add-on purchases per retained member-month (Poisson)
    terms = cfg['addon_terms']
    rates['addon_terms'] = terms
    rates['addon_link'] = cfg['addon_link']
    Xa, ya = _segment_design(pm, terms), pm.add_qty.to_numpy(float)
    rates['addon_b'] = (_poisson_fit(Xa, ya) if cfg['addon_link'] == 'log'
                        else np.linalg.lstsq(Xa, ya, rcond=None)[0])
    # monthly upgrade hazard for members who were standard last paid month
    up = pm[(pm.prev_tier=='standard')]
    uterms = cfg['upgrade_terms']
    rates['upgrade_terms'] = uterms
    rates['upgrade_b'] = _logit_fit(_segment_design(up, uterms), (up.plan_tier=='premium').to_numpy(float))
    # premium share among each cell-cohort's survivors at the origin
    last = pm[pm.month == origin]
    rates['origin_premium'] = (last.assign(p=(last.plan_tier=='premium').astype(float))
                                   .groupby([last.cohort.dt.to_timestamp().dt.strftime('%Y-%m-%d'),'channel','primary_need']).p.mean().to_dict())
    rates['cohort_costs'] = cfg['cohort_costs']
    return rates


def structural_unit_economics(g: pd.DataFrame, rates: dict) -> tuple[np.ndarray, np.ndarray]:
    """Expected net revenue and contribution per retained member for each grid row."""
    g = g.sort_values(KEYS).copy()
    eta = _segment_design(g, rates['addon_terms']) @ rates['addon_b']
    lam = np.exp(eta) if rates['addon_link'] == 'log' else np.maximum(eta, 0.0)
    u = 1/(1+np.exp(-_segment_design(g, rates['upgrade_terms']) @ rates['upgrade_b']))
    g['u'] = u
    pi = np.empty(len(g))
    for key, idx in g.groupby(['cohort_month','channel','primary_need']).indices.items():
        p = rates['origin_premium'].get(key, 0.0)
        for j in idx:                        # rows are sorted by calendar month within the cell
            p = 1 - (1-p)*(1-g.u.iat[j]); pi[j] = p
    S, P = rates['standard'], rates['premium']
    k = (_ordinal(pd.to_datetime(g.cohort_month).dt.to_period('M')) - BASE)
    def sub(tier, col):
        bc = rates['by_cohort'][tier]
        if not rates.get('cohort_costs', True):
            return np.full(len(g), rates[tier][col])
        return np.array([bc['obs'][ki][col] if ki in bc['obs'] else bc['fit'][col][0] + bc['fit'][col][1]*ki for ki in k])
    rev = pi*(sub('premium','sub_rev') + lam*P['add_rev_unit']) + (1-pi)*(sub('standard','sub_rev') + lam*S['add_rev_unit'])
    cost = pi*(sub('premium','sub_cost') + lam*P['add_cost_unit']) + (1-pi)*(sub('standard','sub_cost') + lam*S['add_cost_unit'])
    net = rev*(1 - rates['refund_ratio'])
    out = pd.DataFrame({'net': net, 'contrib': np.minimum(net - cost, net), 'premium_share': pi, 'addon_rate': lam}, index=g.index)
    return out


# ---------------------------------------------------------------- predict
def predict(tables: dict[str,pd.DataFrame], prediction_grid: pd.DataFrame, *, seed: int,
            config: dict | None = None) -> pd.DataFrame:
    cfg = CONFIG | (config or {})
    if cfg.get('ensemble'):  # equal-weight average of member models (keeps monotonicity and contribution <= net)
        members = [predict(tables, prediction_grid, seed=seed, config=CONFIG | m | {'ensemble': None}).sort_values(KEYS).reset_index(drop=True)
                   for m in cfg['ensemble']]
        out = members[0].copy()
        for c in PREDICTIONS:
            out[c] = np.mean([m[c].to_numpy() for m in members], axis=0)
        return out
    grid = prediction_grid.copy(deep=True)
    origin = pd.Period(grid.calendar_month.min(), freq='M') - 1
    panel = build_panel(tables, (origin + 1).to_timestamp())
    rfit = fit_retention(panel, cfg['cohort_trend'], cfg['trend_ridge'], cfg['cohort_effects'], cfg['effect_ridge'],
                         cfg['effect_prior'])
    efit = fit_economics(panel, cfg['calendar_trend'])

    g = grid.copy()
    g['cohort_index'] = pd.to_datetime(g.cohort_month).dt.to_period('M').map(lambda p: p.ordinal) - pd.Period('2024-01','M').ordinal
    g['cal'] = (pd.to_datetime(g.calendar_month).dt.to_period('M').map(lambda p: p.ordinal) - pd.Period('2024-01','M').ordinal)/12
    if cfg['calendar_damping_tau']:  # damp only the future calendar increment beyond the origin
        tau = float(cfg['calendar_damping_tau'])
        cal0 = (origin.ordinal - pd.Period('2024-01','M').ordinal)/12
        h = (g.cal - cal0)*12
        g['cal'] = cal0 + tau*(-np.expm1(-h/tau))/12
    origin_age = g.age_month - (pd.to_datetime(g.calendar_month).dt.to_period('M').map(lambda p: p.ordinal) - origin.ordinal)
    retained = np.empty(len(g))
    for i, (ch, nd, k, a, a0, r0) in enumerate(zip(g.channel, g.primary_need, g.cohort_index, g.age_month, origin_age, g.origin_retained)):
        la, lb = retention_params(rfit, (ch, nd), k)
        retained[i] = float(r0)*np.exp(_log_surv(la, lb, a) - _log_surv(la, lb, a0)) if r0 > 0 else 0.0
    X = efit['design'](g)
    c = efit['coefs']
    npr = np.exp(X @ c['net'])*c['net_smear']
    contrib = npr*np.minimum(np.exp(X @ c['margin'])*c['margin_smear'], 1.0)
    if cfg['economics'] in ('structural', 'blend'):
        se = structural_unit_economics(g, fit_structural(tables, origin, cfg)).reindex(g.index)
        w = 1.0 if cfg['economics'] == 'structural' else cfg['blend_weight']
        npr = w*se.net.to_numpy() + (1-w)*npr
        contrib = w*se.contrib.to_numpy() + (1-w)*contrib
    contrib = np.clip(contrib, 0.0, npr)
    grid['predicted_retained'] = retained
    grid['predicted_net_revenue'] = np.where(retained > 0, retained*npr, 0.0)
    grid['predicted_contribution'] = np.where(retained > 0, retained*contrib, 0.0)
    return grid[KEYS+EXPOSURES+PREDICTIONS]


def load_tables(data_dir: Path, contract: dict) -> dict[str,pd.DataFrame]:
    expected = contract['input_tables']
    if {p.name for p in data_dir.glob('*.csv')} != {name+'.csv' for name in expected}:
        raise ValueError('Supply exactly the eight permitted observed CSVs')
    tables = {}
    for name, fields in expected.items():
        dtypes = {field:('string' if kind in {'VARCHAR','DATE','TIMESTAMP'} else
                         'Int64' if kind=='INTEGER' else 'float64') for field,kind in fields.items()}
        table = pd.read_csv(data_dir/(name+'.csv'),dtype=dtypes)
        if list(table.columns)!=list(fields):
            raise ValueError('Unexpected source schema: '+name)
        tables[name] = table
    return tables


def validate_output(predictions: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(predictions,pd.DataFrame) or set(predictions)!=set(KEYS+EXPOSURES+PREDICTIONS):
        raise ValueError('Return exactly the required grid and prediction columns')
    if predictions.duplicated(KEYS).any():
        raise ValueError('Duplicate prediction keys')
    ordered = predictions.sort_values(KEYS).reset_index(drop=True)
    expected = grid.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered[KEYS+EXPOSURES],expected[KEYS+EXPOSURES],check_dtype=False)
    values = ordered[PREDICTIONS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Predictions must be finite and nonnegative')
    if (ordered.predicted_retained>ordered.origin_retained+1e-8).any():
        raise ValueError('Predicted retained count exceeds origin survivors')
    if (ordered.groupby(KEYS[:3]).predicted_retained.diff().dropna()>1e-8).any():
        raise ValueError('Expected membership must not increase')
    if ((ordered.predicted_retained==0) & ((ordered.predicted_net_revenue>0)|(ordered.predicted_contribution>0))).any():
        raise ValueError('Zero expected members must have zero expected receipts')
    if (ordered.predicted_contribution>ordered.predicted_net_revenue+1e-8).any():
        raise ValueError('Contribution exceeds net revenue')
    return ordered[KEYS+EXPOSURES+PREDICTIONS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--prediction-grid',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261116)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.receipt.json').exists():
        parser.error('Use a new output path; existing forecasts are preserved')
    contract_path = Path(__file__).with_name('submission-contract.json')
    contract = json.loads(contract_path.read_text())
    tables = load_tables(args.data_dir,contract)
    grid = pd.read_csv(args.prediction_grid)
    started = time.monotonic()
    predictions = validate_output(predict(tables,grid.copy(deep=True),seed=args.seed),grid)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(args.output,index=False,float_format='%.10f',lineterminator='\n')
    sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt = {'model':'sBG retention with shrunk cohort offsets + log-linear unit economics','config':CONFIG,
               'rows':len(predictions),'seed':args.seed,'elapsed_seconds':time.monotonic()-started,
               'source_sha256':sha(__file__),'uv_lock_sha256':sha(Path(__file__).with_name('uv.lock')),
               'contract_sha256':sha(contract_path),'prediction_grid_sha256':sha(args.prediction_grid),
               'input_sha256':{name:sha(args.data_dir/(name+'.csv')) for name in tables},
               'prediction_sha256':sha(args.output),'future_outcomes_read':False}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'validated','rows':len(predictions),'output':str(args.output)}))


if __name__ == '__main__':
    main()
