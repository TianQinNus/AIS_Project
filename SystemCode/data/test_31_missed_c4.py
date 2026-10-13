"""Test 31: missed-C4 diagnostics and exploratory stage/acceleration safety rules.

Uses OOF outputs from Tests 29/30; no model training, database, or writes to originals.
Run: python test_31_missed_c4.py

The official validation set was repeatedly used during development. The internal
split here reduces direct rule-selection leakage, but is NOT an independent test.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from test_30_stage_aware_rules import add_stage_features, check_input, metrics, LEVEL, ORDER

ATTENTION = {'MONITOR', 'ENHANCED_MONITOR', 'INSPECT', 'PRIORITY_INSPECT',
             'MANUAL_REVIEW', 'COLLECT_MORE_DATA'}
INSPECT = {'INSPECT', 'PRIORITY_INSPECT'}


def acceleration_features(history):
    needed = {'vehicle_id', 'time_step', 'Risk6', 'Risk12'}
    if not needed.issubset(history.columns):
        raise ValueError(f'History missing: {sorted(needed-set(history.columns))}')
    if history.duplicated(['vehicle_id', 'time_step']).any():
        raise ValueError('Duplicate history time_step per vehicle')
    rows = []
    for vehicle, g in history.groupby('vehicle_id', sort=False):
        g = g.sort_values('time_step')
        t = g.time_step.to_numpy(float)
        row = {'vehicle_id': vehicle, 'accel_observations': len(t)}
        for horizon in (6, 12):
            y = g[f'Risk{horizon}'].to_numpy(float)
            dt = np.diff(t)
            valid = (dt > 0) & np.isfinite(dt) & np.isfinite(np.diff(y))
            if len(dt) >= 2 and valid[-2:].all():
                v1, v2 = np.diff(y)[-2:] / dt[-2:]
                # Change in rate; avoid claiming physical acceleration.
                row[f'Risk{horizon}_acceleration'] = float((v2-v1) / ((dt[-1]+dt[-2])/2))
                row[f'Risk{horizon}_last_slope'] = float(v2)
            else:
                row[f'Risk{horizon}_acceleration'] = np.nan
                row[f'Risk{horizon}_last_slope'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_signals(df, history):
    d = add_stage_features(df)
    d = d.merge(acceleration_features(history), on='vehicle_id', validate='one_to_one')
    # Within-stratum Risk6 is a distinct signal from within-stratum share.
    d['risk6_within_stratum_pct'] = d.groupby('risk_stratum').Risk6_current.rank(pct=True)*100
    d['risk12_within_stratum_pct'] = d.groupby('risk_stratum').Risk12_current.rank(pct=True)*100
    # Rank acceleration only where a valid temporal derivative exists.
    for h in (6, 12):
        d[f'Risk{h}_accel_pct'] = d[f'Risk{h}_acceleration'].rank(pct=True)*100
    return d


def candidate_action(row, variant):
    base = str(row.stage_attention_action)
    if base in {'MANUAL_REVIEW', 'COLLECT_MORE_DATA'}:
        return base, 'BASE_REVIEW'
    base_level = LEVEL.get(base, 0)
    stratum = row.risk_stratum
    near = float(row.near_share_within_stratum_pct)
    immin = float(row.imminent_share_within_stratum_pct)
    abs6 = float(row.Risk6_percentile)
    abs12 = float(row.Risk12_percentile)
    same6 = float(row.risk6_within_stratum_pct)
    accel6 = float(row.Risk6_accel_pct) if pd.notna(row.Risk6_accel_pct) else -1
    accel12 = float(row.Risk12_accel_pct) if pd.notna(row.Risk12_accel_pct) else -1
    # Predeclared hypotheses. Conservative absolute-risk gates guard tiny denominators.
    stage = (stratum in ('MEDIUM', 'HIGH') and same6 >= 85 and
             immin >= 75 and abs6 >= 50)
    trend = (stratum in ('MEDIUM', 'HIGH') and near >= 70 and
             abs12 >= 50 and (accel6 >= 90 or accel12 >= 90) and
             (float(row.Risk6_last_slope) > 0 if pd.notna(row.Risk6_last_slope) else False))
    if variant == 'stage' and stage:
        return ORDER[max(base_level, 2)], 'STAGE_RISK6'
    if variant == 'accel' and trend:
        return ORDER[max(base_level, 1)], 'ACCELERATION'
    if variant == 'combined':
        if stage:
            return ORDER[max(base_level, 2)], 'STAGE_RISK6'
        if trend:
            return ORDER[max(base_level, 1)], 'ACCELERATION'
    return base, 'BASE'


def compact(df, action_col):
    y = df.class_label.to_numpy(int)
    a = df[action_col].astype(str)
    att = a.isin(ATTENTION).to_numpy()
    ins = a.isin(INSPECT).to_numpy()
    c4 = y == 4
    c34 = y >= 3
    c0 = y == 0
    return {
        'c4_covered': int((c4 & att).sum()),
        'c4_missed': int((c4 & ~att).sum()),
        'c4_recall': float((c4 & att).sum()/c4.sum()) if c4.any() else np.nan,
        'c34_recall': float((c34 & att).sum()/c34.sum()) if c34.any() else np.nan,
        'c4_immediate': int((c4 & ins).sum()),
        'c0_attention': int((c0 & att).sum()),
        'c0_immediate': int((c0 & ins).sum()),
        'attention_total': int(att.sum()),
        'immediate_total': int(ins.sum()),
    }


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument('--decisions', type=Path, default=root/'processed/test_30_stage_aware_decisions.csv')
    p.add_argument('--history', type=Path, default=root/'processed/test_29_oof_history.csv')
    p.add_argument('--out-dir', type=Path, default=root/'processed')
    p.add_argument('--eval-fraction', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not 0.2 <= args.eval_fraction <= 0.8:
        p.error('--eval-fraction must be between 0.2 and 0.8')
    d = pd.read_csv(args.decisions)
    h = pd.read_csv(args.history)
    check_input(d)
    if 'stage_attention_action' not in d.columns:
        raise ValueError('Run Test 30 first; missing stage_attention_action')
    if set(d.vehicle_id) != set(h.vehicle_id):
        raise ValueError('History and decisions vehicle IDs do not match')
    d = add_signals(d, h)
    dev_idx, eval_idx = train_test_split(
        np.arange(len(d)), test_size=args.eval_fraction,
        stratify=d.class_label, random_state=args.seed)
    d['subset'] = 'development'
    d.loc[eval_idx, 'subset'] = 'evaluation'
    # Existing stage-aware rule is the fixed reference.
    for name in ('stage', 'accel', 'combined'):
        result = d.apply(lambda r: candidate_action(r, name), axis=1)
        d[f'{name}_action'] = [x[0] for x in result]
        d[f'{name}_trigger'] = [x[1] for x in result]

    cols = {'baseline': 'stage_attention_action', 'stage': 'stage_action',
            'accel': 'accel_action', 'combined': 'combined_action'}
    records = []
    for subset in ('development', 'evaluation'):
        part = d[d.subset == subset]
        for name, col in cols.items():
            records.append({'subset': subset, 'method': name, 'vehicles': len(part),
                            'c4_total': int((part.class_label == 4).sum()), **compact(part, col)})
    report = pd.DataFrame(records)
    # Choose a candidate on development only; penalize extra C0 attention.
    # One extra C4 must be worth at least 20 additional C0 attention cases.
    dev = report[report.subset == 'development'].set_index('method')
    base = dev.loc['baseline']
    score = {name: 20*(dev.loc[name,'c4_covered']-base.c4_covered)
             -(dev.loc[name,'c0_attention']-base.c0_attention)
             for name in cols}
    winner = max(cols, key=lambda n: (score[n], n == 'baseline'))

    # Diagnose missed C4 in the held-out evaluation split without using them
    # to select a candidate or its thresholds.
    ev = d[d.subset == 'evaluation'].copy()
    missed = ev[(ev.class_label == 4) & ~ev.stage_attention_action.isin(ATTENTION)].copy()
    inspect_cols = ['vehicle_id', 'class_label', 'risk_stratum', 'Risk48_current',
                    'Risk6_current', 'Risk6_percentile', 'near_share', 'imminent_share',
                    'risk6_within_stratum_pct', 'Risk6_acceleration', 'Risk12_acceleration',
                    'stage_attention_action', 'stage_action', 'accel_action', 'combined_action']
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out_dir/'test_31_comparison.csv', index=False)
    d.to_csv(args.out_dir/'test_31_vehicle_decisions.csv', index=False)
    missed[inspect_cols].to_csv(args.out_dir/'test_31_missed_c4.csv', index=False)
    print('\n=== DEVELOPMENT VS EVALUATION (OOF, EXPLORATORY) ===')
    print(report.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print('\nDevelopment-only utility scores:', score)
    print('Selected for inspection (NOT deployment approval):', winner)
    evr = report[report.subset == 'evaluation'].set_index('method')
    print('\n=== EVALUATION DELTA vs STAGE-AWARE BASELINE ===')
    for name in ('stage', 'accel', 'combined'):
        print(f'{name}: C4 recovered {int(evr.loc[name,"c4_covered"]-evr.loc["baseline","c4_covered"]):+d}; '
              f'extra C0 attention {int(evr.loc[name,"c0_attention"]-evr.loc["baseline","c0_attention"]):+d}')
    print('\n=== MISSED C4 UNDER BASELINE: EVALUATION ONLY ===')
    print('Missed:', len(missed))
    print(missed[inspect_cols].head(20).to_string(index=False))
    print('\nSaved: test_31_comparison.csv, test_31_vehicle_decisions.csv, test_31_missed_c4.csv')
    print('WARNING: Prior repeated use of official validation makes this exploratory, not independent validation.')
    print('OOF replay scores and temporal changes are model signals, not confirmed physical failure mechanisms.')


if __name__ == '__main__':
    main()
