"""FleetCare Test 23: target-domain validation-weight sensitivity.

Run beside compare_final_lgbm_xgb.py and test_20_normalized.py:
  python test_23_weight_sensitivity.py --algorithm xgboost --weights 1,3,10,30
  python test_23_weight_sensitivity.py --algorithm lightgbm --weights 1,3,10,30

Every held-out vehicle is excluded from its fold's fitting data. The original
training rows remain in each fold. These are exploratory cross-fitted results:
the official validation data were previously reused for model development.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score

from compare_final_lgbm_xgb import prepare, make_model, CLASSES, SEED

CAPACITIES = (0.025, 0.05, 0.10, 0.20)


def score_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if p.shape != (len(y), 5) or not np.isfinite(p).all():
        raise ValueError('OOF probability shape or values invalid')
    pos = (y != 0).astype(int)
    if pos.sum() == 0 or pos.sum() == len(pos):
        raise ValueError('Need both positive and negative validation vehicles')
    risk = 1.0 - p[:, CLASSES.index(0)]
    pred = np.asarray(CLASSES)[p.argmax(axis=1)]
    order = np.argsort(-risk, kind='stable')
    result = {
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0)),
        'risk_ap': float(average_precision_score(pos, risk)),
        'risk_roc_auc': float(roc_auc_score(pos, risk)),
    }
    for cap in CAPACITIES:
        k = max(1, int(np.ceil(len(y) * cap)))
        found = int(pos[order[:k]].sum())
        suffix = str(cap).replace('.', 'p')
        result[f'found_top_{suffix}'] = found
        result[f'recall_top_{suffix}'] = found / int(pos.sum())
        result[f'precision_top_{suffix}'] = found / k
    return result


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=root / 'raw')
    ap.add_argument('--output-dir', type=Path, default=root / 'processed')
    ap.add_argument('--algorithm', choices=('xgboost', 'lightgbm'), default='xgboost')
    ap.add_argument('--weights', default='1,3,10,30')
    ap.add_argument('--k-repair', type=int, default=3)
    ap.add_argument('--k-nonrepair', type=int, default=2)
    ap.add_argument('--lags', default='5,20')
    ap.add_argument('--val-weight', type=float, default=10.0)  # used by imported prepare() metadata only
    args = ap.parse_args()
    weights_to_try = [float(v.strip()) for v in args.weights.split(',') if v.strip()]
    if not weights_to_try or any(not np.isfinite(v) or v <= 0 for v in weights_to_try):
        ap.error('All weights must be positive finite numbers')
    if len(set(weights_to_try)) != len(weights_to_try):
        ap.error('Weights must be unique')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    x_tr, x_va, y_tr, y_va, sw, ids, _ = prepare(args)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Validation vehicle IDs must be unique')
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(x_va, y_va))
    summary = []
    predictions = pd.DataFrame({'vehicle_id': ids, 'actual_class': y_va})
    for weight in weights_to_try:
        print(f'\n=== {args.algorithm.upper()} | validation weight={weight:g} ===', flush=True)
        oof = np.full((len(y_va), 5), np.nan, dtype=np.float32)
        for fold_no, (adapt_idx, hold_idx) in enumerate(folds, start=1):
            # Fold held-out validation vehicles never enter this fold's fit.
            x_fit = np.vstack((x_tr, x_va[adapt_idx]))
            y_fit = np.concatenate((y_tr, y_va[adapt_idx]))
            fit_weights = np.concatenate((sw, np.full(len(adapt_idx), weight, dtype=float)))
            m = make_model(args.algorithm)
            m.fit(x_fit, y_fit, sample_weight=fit_weights)
            if list(m.classes_) != CLASSES:
                raise ValueError(f'Class order mismatch: {m.classes_}')
            oof[hold_idx] = m.predict_proba(x_va[hold_idx]).astype(np.float32)
            print(f'  fold {fold_no}/5 complete ({len(hold_idx)} held out)', flush=True)
            del m, x_fit, y_fit, fit_weights
            gc.collect()
        row = {'algorithm': args.algorithm, 'val_weight': weight, **score_metrics(y_va, oof)}
        summary.append(row)
        prefix = f'w{str(weight).replace(".", "p")}'
        for c in CLASSES:
            predictions[f'{prefix}_p_c{c}'] = oof[:, CLASSES.index(c)]
        predictions[f'{prefix}_risk'] = 1.0 - oof[:, CLASSES.index(0)]
        print(pd.Series(row).to_string(), flush=True)
        del oof
        gc.collect()

    table = pd.DataFrame(summary)
    # Weight 10 is a reference only; do not select the best weight from this
    # repeatedly reused validation set and claim an independent test result.
    ref = table.loc[np.isclose(table['val_weight'], 10.0)]
    if len(ref) == 1:
        for metric in ('risk_ap', 'risk_roc_auc', 'recall_top_0p05'):
            table[f'delta_{metric}_vs_w10'] = table[metric] - float(ref.iloc[0][metric])

    prefix = f'test_23_{args.algorithm}'
    table.to_csv(args.output_dir / f'{prefix}_summary.csv', index=False)
    predictions.to_csv(args.output_dir / f'{prefix}_oof_predictions.csv', index=False)
    (args.output_dir / f'{prefix}_config.json').write_text(
        json.dumps({'algorithm': args.algorithm, 'weights': weights_to_try,
                    'seed': SEED, 'folds': 5, 'lags': args.lags,
                    'k_repair': args.k_repair, 'k_nonrepair': args.k_nonrepair,
                    'risk_definition': 'C1-C4 vs C0; risk = 1 - p_c0',
                    'evaluation_note': 'Exploratory OOF; official validation previously reused.'}, indent=2),
        encoding='utf-8')
    print('\n=== WEIGHT SENSITIVITY SUMMARY ===')
    cols = ['val_weight', 'risk_ap', 'risk_roc_auc', 'recall_top_0p05',
            'precision_top_0p05', 'macro_f1']
    print(table[cols].round(5).to_string(index=False))
    print('\nSaved:', args.output_dir / f'{prefix}_summary.csv')
    print('Exploratory results only: do not report the best weight as independent test performance.')


if __name__ == '__main__':
    main()
