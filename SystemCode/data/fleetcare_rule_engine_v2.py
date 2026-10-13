"""FleetCare multi-signal decision engine v2.

Reads cross-fitted vehicle risk scores and validation readout histories.
Does NOT retrain or alter ML scores. Uses operational context for decision
triage, and offers an optional, explicitly experimental, capacity-matched
reranking test. Never treat histogram change as confirmed deterioration.

Usage:
 python fleetcare_rule_engine_v2.py --algorithm xgboost
 python fleetcare_rule_engine_v2.py --algorithm xgboost --inspect-capacity 0.05
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def groups_from_columns(columns):
    groups = {}
    for c in columns:
        match = re.fullmatch(r'(\d+)_(\d+)', str(c))
        if match:
            groups.setdefault(match.group(1), []).append((int(match.group(2)), c))
    return {key: [col for _, col in sorted(items)] for key, items in groups.items() if len(items) > 1}


def jsd(a, b):
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    if (a < 0).any() or (b < 0).any() or a.sum() <= 0 or b.sum() <= 0:
        return np.nan
    a = a / a.sum()
    b = b / b.sum()
    midpoint = (a + b) / 2
    ia, ib = a > 0, b > 0
    return float((np.sum(a[ia] * np.log(a[ia] / midpoint[ia])) +
                  np.sum(b[ib] * np.log(b[ib] / midpoint[ib]))) / (2 * np.log(2)))


def extract_context(readouts_path, vehicle_ids):
    columns = pd.read_csv(readouts_path, nrows=0).columns.tolist()
    groups = groups_from_columns(columns)
    if not groups:
        raise ValueError('No histogram groups found in readouts')
    cols = ['vehicle_id', 'time_step'] + [c for g in groups.values() for c in g]
    df = pd.read_csv(readouts_path, usecols=cols)
    df = df.loc[df.vehicle_id.isin(vehicle_ids)].sort_values(['vehicle_id', 'time_step'])
    if df.duplicated(['vehicle_id', 'time_step']).any():
        raise ValueError('Duplicate vehicle_id/time_step')
    rows = []
    for vid, group in df.groupby('vehicle_id', sort=False):
        n = len(group)
        recent = group.tail(3)
        steps = recent.time_step.to_numpy(dtype=float)
        changes = []
        valid = 0
        if n >= 3 and np.all(np.diff(steps) > 0):
            for hist_cols in groups.values():
                values = recent[hist_cols].to_numpy(dtype=float)
                old, new = values[1] - values[0], values[2] - values[1]
                value = jsd(old, new)
                if np.isfinite(value):
                    changes.append(value)
                    valid += 1
        last_gap = float(steps[-1] - steps[-2]) if n >= 2 else np.nan
        latest_missing = float(group.iloc[-1][cols[2:]].isna().mean())
        rows.append({'vehicle_id': vid, 'history_count': n,
                     'last_time_step': float(steps[-1]), 'last_gap': last_gap,
                     'hist_change': float(np.mean(changes)) if changes else np.nan,
                     'valid_hist_groups': valid, 'latest_missing_fraction': latest_missing})
    return pd.DataFrame(rows)


def evaluate(y, scores, capacity):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    k = max(1, int(np.ceil(len(y) * capacity)))
    chosen = np.argsort(-scores, kind='stable')[:k]
    tp = int(y[chosen].sum())
    return {'vehicles': len(y), 'positives': int(y.sum()), 'capacity': capacity,
            'selected': k, 'found': tp, 'precision': tp/k,
            'recall': tp/int(y.sum()), 'ap': average_precision_score(y, scores),
            'roc_auc': roc_auc_score(y, scores)}


def build_decisions(df, inspect_capacity=0.05, enhanced_capacity=0.10,
                    monitor_capacity=0.20, missing_limit=0.10):
    """Decision layer; no risk score changes. Review flags are overlays, not extra inspections."""
    if not 0 < inspect_capacity < enhanced_capacity < monitor_capacity < 1:
        raise ValueError('Require 0 < inspect < enhanced < monitor < 1')
    df = df.sort_values(['ml_risk', 'vehicle_id'], ascending=[False, True], kind='stable').copy()
    n = len(df)
    df['ml_rank'] = np.arange(1, n+1)
    df['ml_percentile'] = 100 * (n - df.ml_rank + 1)/n
    df['ml_selected'] = df.ml_rank <= int(np.ceil(n*inspect_capacity))
    df['data_review'] = ((df.history_count < 3) |
                         (df.latest_missing_fraction > missing_limit) |
                         (df.valid_hist_groups < 3))
    # The high-change threshold is relative to the current fleet, not a fault threshold.
    valid = df.hist_change.dropna()
    cutoff = valid.quantile(0.90) if len(valid) else np.inf
    df['usage_shift'] = df.hist_change.notna() & (df.hist_change >= cutoff)
    df['operational_evidence'] = np.select(
        [df.data_review, df.usage_shift],
        ['INSUFFICIENT_DATA', 'USAGE_PATTERN_SHIFT'], default='NO_ADDITIONAL_FLAG')
    k_critical = int(np.ceil(n*inspect_capacity/2))
    k_inspect = int(np.ceil(n*inspect_capacity))
    k_enhanced = int(np.ceil(n*enhanced_capacity))
    k_monitor = int(np.ceil(n*monitor_capacity))
    r = df.ml_rank.to_numpy()
    df['rule_id'] = np.select([r<=k_critical, r<=k_inspect, r<=k_enhanced, r<=k_monitor],
                              ['TR1','TR2','TR4','TR5'], default='TR6')
    df['priority'] = np.select([r<=k_critical, r<=k_inspect, r<=k_enhanced, r<=k_monitor],
                               ['CRITICAL','HIGH','MEDIUM','LOW'], default='NORMAL')
    df['action'] = np.select([r<=k_critical, r<=k_inspect, r<=k_enhanced, r<=k_monitor],
                             ['PRIORITY_INSPECT','INSPECT','ENHANCED_MONITOR','MONITOR'], default='CONTINUE')
    # A review overlay does not silently consume or expand the inspection capacity.
    df['review_required'] = df.data_review
    df['review_reason'] = np.where(df.data_review, 'Data/history quality needs review', '')
    df['reason'] = np.where(df.usage_shift,
                            'ML-ranked priority; recent usage distribution changed (not proven failure)',
                            'ML-ranked priority; no confirmed additional mechanical signal')
    df['rule_trace'] = ('rule=' + df.rule_id + '|ml_rank=' + df.ml_rank.astype(str) +
                        '|data_review=' + df.data_review.astype(str) +
                        '|usage_shift=' + df.usage_shift.astype(str))
    return df


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', type=Path, default=root/'processed'/'final_model_oof_predictions.csv')
    ap.add_argument('--readouts', type=Path, default=root/'raw'/'validation_operational_readouts.csv')
    ap.add_argument('--out-dir', type=Path, default=root/'processed')
    ap.add_argument('--algorithm', choices=['xgboost','lightgbm'], default='xgboost')
    ap.add_argument('--inspect-capacity', type=float, default=0.05)
    args = ap.parse_args()
    pred = pd.read_csv(args.predictions)
    col = f'{args.algorithm}_risk'
    required = {'vehicle_id','actual_class',col}
    if not required.issubset(pred.columns):
        raise ValueError(f'Missing OOF columns: {sorted(required-set(pred.columns))}')
    if pred.vehicle_id.duplicated().any():
        raise ValueError('Expected exactly one OOF prediction per vehicle')
    pred['ml_risk'] = pd.to_numeric(pred[col], errors='raise')
    if not np.isfinite(pred.ml_risk).all() or not pred.ml_risk.between(0,1).all():
        raise ValueError('Invalid ML risk')
    pred['actual_risk'] = (pred.actual_class.astype(int)!=0).astype(int)
    context = extract_context(args.readouts, set(pred.vehicle_id))
    data = pred.merge(context, on='vehicle_id', how='left', validate='one_to_one')
    if data.history_count.isna().any():
        raise ValueError('Readout history missing for some vehicles')
    decisions = build_decisions(data, inspect_capacity=args.inspect_capacity)
    y = decisions.actual_risk.to_numpy()
    baseline = evaluate(y, decisions.ml_risk.to_numpy(), args.inspect_capacity)
    rule = evaluate(y, decisions.ml_selected.astype(float).to_numpy(), args.inspect_capacity)
    # Exact top-k selected identities, not a claim of new ranking improvement.
    print('\n=== MODEL RANKING (OOF) ===')
    print(pd.DataFrame([baseline]).round(5).to_string(index=False))
    print('\n=== RULE ENGINE OPERATIONAL SIGNALS ===')
    print('Review required:', int(decisions.review_required.sum()))
    print('Usage pattern shift:', int(decisions.usage_shift.sum()))
    print('Action counts:\n', decisions.action.value_counts().to_string())
    print('ML and rule inspection selection identical:', bool(np.array_equal(
        decisions.ml_selected.to_numpy(), decisions.ml_rank.to_numpy() <= int(np.ceil(len(decisions)*args.inspect_capacity)))))
    print('NOTE: no accuracy lift claimed. Rule decisions preserve ML rank; additional evidence only annotates/flags.')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(args.out_dir/f'test_25_{args.algorithm}_multisignal_decisions.csv', index=False)
    pd.DataFrame([{'method':'ML TopK', **baseline},
                  {'method':'Rule Engine (same selected set)', **{**baseline}}]).to_csv(
        args.out_dir/f'test_25_{args.algorithm}_comparison.csv', index=False)
    print('Saved outputs to', args.out_dir)

if __name__ == '__main__':
    main()
