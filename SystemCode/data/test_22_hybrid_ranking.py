"""FleetCare Test 22: exploratory risk-ranking fusion with temporal histogram change.

Inputs:
  processed/final_model_oof_predictions.csv (from compare_final_lgbm_xgb.py)
  raw/validation_operational_readouts.csv

No model training. OOF model scores are used to avoid direct in-sample scoring.
The temporal signal measures changes in *usage distributions*, not confirmed deterioration.
The official validation set has been repeatedly used in development; results are exploratory.
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def hist_groups(columns):
    groups = {}
    for c in columns:
        m = re.fullmatch(r'(\d+)_(\d+)', str(c))
        if m:
            groups.setdefault(m.group(1), []).append((int(m.group(2)), c))
    return {k: [c for _, c in sorted(v)] for k, v in groups.items() if len(v) > 1}


def js_divergence(a, b):
    """Jensen-Shannon divergence for nonnegative vectors, natural log, 0..log(2)."""
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    if (a < 0).any() or (b < 0).any() or a.sum() <= 0 or b.sum() <= 0:
        return np.nan
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    mask_a, mask_b = a > 0, b > 0
    return float(0.5 * np.sum(a[mask_a] * np.log(a[mask_a] / m[mask_a]))
                 + 0.5 * np.sum(b[mask_b] * np.log(b[mask_b] / m[mask_b])))


def temporal_signal(op_path, vehicle_ids, min_history=3):
    header = pd.read_csv(op_path, nrows=0).columns.tolist()
    groups = hist_groups(header)
    if not groups:
        raise ValueError('No multi-bin histogram groups detected')
    usecols = ['vehicle_id', 'time_step'] + [c for cols in groups.values() for c in cols]
    df = pd.read_csv(op_path, usecols=usecols)
    df = df.loc[df.vehicle_id.isin(vehicle_ids)].sort_values(['vehicle_id', 'time_step'])
    out = []
    for vid, g in df.groupby('vehicle_id', sort=False):
        g = g.tail(3)  # compare two successive histogram increments; earlier readouts are not needed
        scores = []
        if len(g) >= min_history:
            for cols in groups.values():
                values = g[cols].to_numpy(dtype=np.float64)
                # Cumulative histogram differences approximate usage between readouts.
                old = values[-2] - values[-3]
                new = values[-1] - values[-2]
                v = js_divergence(old, new)
                if np.isfinite(v):
                    scores.append(v / np.log(2))
        out.append((vid, float(np.mean(scores)) if scores else np.nan, len(scores)))
    return pd.DataFrame(out, columns=['vehicle_id', 'hist_change', 'valid_hist_groups'])


def evaluate(y, score, capacity=0.05):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError('Evaluation requires both positive and negative vehicles')
    k = max(1, int(np.ceil(len(y) * capacity)))
    selected = np.argsort(-score, kind='stable')[:k]
    tp = int(y[selected].sum())
    return {'n': len(y), 'positives': int(y.sum()), 'selected': k,
            'found': tp, 'ap': float(average_precision_score(y, score)),
            'roc_auc': float(roc_auc_score(y, score)),
            'recall_at_capacity': float(tp / y.sum()),
            'precision_at_capacity': float(tp / k)}


def percentile_from_tuning(tune, values):
    # Calibrate ranks only using tuning vehicles, avoiding holdout distribution fitting.
    t = np.sort(np.asarray(tune, dtype=float))
    return np.searchsorted(t, values, side='right') / len(t)


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument('--predictions', type=Path, default=root/'processed'/'final_model_oof_predictions.csv')
    ap.add_argument('--readouts', type=Path, default=root/'raw'/'validation_operational_readouts.csv')
    ap.add_argument('--out-dir', type=Path, default=root/'processed')
    ap.add_argument('--algorithm', choices=['xgboost', 'lightgbm'], default='xgboost')
    ap.add_argument('--capacity', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if not 0 < args.capacity < 1:
        raise ValueError('capacity must be between 0 and 1')
    df = pd.read_csv(args.predictions)
    col = f'{args.algorithm}_risk'
    required = {'vehicle_id', 'actual_class', col}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns {required-set(df.columns)}. Run compare_final_lgbm_xgb.py --mode compare first.')
    if df.vehicle_id.duplicated().any():
        raise ValueError('OOF predictions must have one row per vehicle')
    df['actual_risk'] = (df.actual_class != 0).astype(int)
    df['ml_risk'] = pd.to_numeric(df[col], errors='raise')
    if not np.isfinite(df.ml_risk).all():
        raise ValueError('Non-finite ML risk scores')
    print('Extracting histogram change signal (may take a minute)...', flush=True)
    changes = temporal_signal(args.readouts, set(df.vehicle_id))
    df = df.merge(changes, on='vehicle_id', how='left', validate='one_to_one')
    coverage = df.hist_change.notna().mean()
    print(f'Valid temporal signal coverage: {coverage:.1%}')
    if coverage < 0.5:
        raise RuntimeError('Less than 50% temporal coverage; not enough for meaningful test')
    # Missing signal is assigned median based on tuning vehicles only.
    idx_t, idx_h = train_test_split(np.arange(len(df)), test_size=0.5,
                                     stratify=df.actual_risk, random_state=args.seed)
    tune, hold = df.iloc[idx_t].copy(), df.iloc[idx_h].copy()
    median = tune.hist_change.median()
    tune['change_filled'] = tune.hist_change.fillna(median)
    hold['change_filled'] = hold.hist_change.fillna(median)
    # Each score transformed to percentile using TUNE distribution only.
    tune_ml = percentile_from_tuning(tune.ml_risk, tune.ml_risk)
    hold_ml = percentile_from_tuning(tune.ml_risk, hold.ml_risk)
    tune_t = percentile_from_tuning(tune.change_filled, tune.change_filled)
    hold_t = percentile_from_tuning(tune.change_filled, hold.change_filled)

    # Limited predefined grid; tune on tune partition only.
    weights = [0.0, 0.05, 0.10, 0.20, 0.30]
    candidates = []
    for w in weights:
        s = (1-w)*tune_ml + w*tune_t
        m = evaluate(tune.actual_risk, s, args.capacity)
        candidates.append({'temporal_weight': w, **m})
    # Maximize recall@capacity; tie-break by AP, then smaller weight.
    best = max(candidates, key=lambda d: (d['recall_at_capacity'], d['ap'], -d['temporal_weight']))
    w = best['temporal_weight']
    hold_hybrid = (1-w)*hold_ml + w*hold_t
    baseline = evaluate(hold.actual_risk, hold.ml_risk, args.capacity)
    hybrid = evaluate(hold.actual_risk, hold_hybrid, args.capacity)
    print('\n=== TUNING CANDIDATES (exploratory) ===')
    print(pd.DataFrame(candidates)[['temporal_weight','ap','roc_auc','recall_at_capacity','precision_at_capacity']].round(4).to_string(index=False))
    print(f'\nSelected temporal weight: {w}')
    print('\n=== HELD-OUT COMPARISON ===')
    summary = pd.DataFrame([{'method':'ML Only', **baseline}, {'method':'ML + Histogram Change', **hybrid}])
    print(summary.round(4).to_string(index=False))
    print('\nCaution: split is within a validation set already reused for development; not an independent test.')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).to_csv(args.out_dir/'test_22_tuning.csv', index=False)
    summary.to_csv(args.out_dir/'test_22_holdout_comparison.csv', index=False)
    hold['hybrid_score'] = hold_hybrid
    hold['selected_temporal_weight'] = w
    hold.to_csv(args.out_dir/'test_22_holdout_predictions.csv', index=False)
    print('Saved:', args.out_dir/'test_22_holdout_comparison.csv')

if __name__ == '__main__':
    main()
