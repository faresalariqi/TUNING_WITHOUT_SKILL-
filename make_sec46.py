"""Summaries for Section 4.6 / Table 6: seeds and search-budget runs."""
import sys, json, os
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, '/home/claude/r2/code')
from stats_rev import per_asset_metrics, MODELS
R = '/home/claude/r2/runs'

def summarise(run, label, cands):
    P = pd.read_csv(f'{R}/{run}/predictions.csv', parse_dates=['date'])
    C = pd.read_csv(f'{R}/{run}/chosen_params.csv')
    M = per_asset_metrics(P)
    up = M[M.method == 'AlwaysUp'].set_index('asset').acc
    row = dict(run=label, cands=cands)
    ps = []
    for mm in MODELS:
        d = (M[M.method == f'{mm}_tuned'].set_index('asset').bacc - M[M.method == f'{mm}_def'].set_index('asset').bacc) * 100
        row[f'd_{mm}'] = d.mean(); p = stats.wilcoxon(d.values).pvalue; ps.append(p); row[f'p_{mm}'] = p
    row['min_p'] = min(ps)
    accs = {m: M[M.method == m].acc.mean() for m in M.method.unique()}
    row['acc_tuned_min'] = min(accs[f'{m}_tuned'] for m in MODELS) * 100
    row['acc_tuned_max'] = max(accs[f'{m}_tuned'] for m in MODELS) * 100
    row['acc_up'] = accs['AlwaysUp'] * 100
    row['n_rules_above_up'] = sum(accs[f'{m}_{k}'] > accs['AlwaysUp'] for m in MODELS for k in ('def', 'tuned'))
    g = C.groupby('model').apply(lambda x: 100 * ((x.inner_bacc - x.outer_bacc_tuned).mean() - (x.default_inner_bacc - x.outer_bacc_def).mean()))
    row['selopt_min'] = g.min(); row['selopt_max'] = g.max()
    gap = C.groupby('model').apply(lambda x: 100 * (x.inner_bacc - x.outer_bacc_tuned).mean())
    row['gap_min'] = gap.min(); row['gap_max'] = gap.max()
    row['picked_default'] = {m: float(C[C.model == m].picked_default.mean()) for m in MODELS}
    row['mcc_tuned_max'] = max(M[M.method == f'{m}_tuned'].mcc.mean() for m in MODELS)
    row['mcc_all_min'] = min(M[M.method == f'{m}_{k}'].mcc.mean() for m in MODELS for k in ('def', 'tuned'))
    row['mcc_all_max'] = max(M[M.method == f'{m}_{k}'].mcc.mean() for m in MODELS for k in ('def', 'tuned'))
    return row

rows = []
for run, lab, c in [('repro_s42', 'Seed 42 (re-run)', 12), ('seed_1', 'Seed 1', 12), ('seed_2', 'Seed 2', 12),
                    ('seed_3', 'Seed 3', 12), ('seed_4', 'Seed 4', 12), ('budget48_def', '48 candidates + default', 49)]:
    if os.path.exists(f'{R}/{run}/predictions.csv'):
        rows.append(summarise(run, lab, c))
T = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(T.drop(columns=['picked_default']).round(3).to_string())
for r in rows: print(r['run'], r['picked_default'])
seeds = T[T.run.str.startswith('Seed')]
print('seed summary: mean d per model', seeds[[f'd_{m}' for m in MODELS]].mean().round(2).to_dict(),
      'sd', seeds[[f'd_{m}' for m in MODELS]].std().round(2).to_dict())
print('all seed d range', seeds[[f'd_{m}' for m in MODELS]].values.min().round(2), seeds[[f'd_{m}' for m in MODELS]].values.max().round(2))
T.to_csv('/home/claude/r2/stats_robust.csv', index=False)
