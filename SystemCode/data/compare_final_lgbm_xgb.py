"""FleetCare: fair LightGBM vs XGBoost comparison on normalized features.

Requires test_20_normalized.py alongside this file and the original raw CSVs.

python compare_final_lgbm_xgb.py --mode compare
python compare_final_lgbm_xgb.py --mode compare --train-final
python compare_final_lgbm_xgb.py --mode predict --algorithm xgboost --op-csv raw/test_operational_readouts.csv --spec-csv raw/test_specifications.csv

Cross-fitting: each held-out validation fold is excluded from its model's training.
WARNING: the official validation labels were previously used during development;
these are exploratory, not independent test-set estimates.
"""
import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, average_precision_score,
                             roc_auc_score, classification_report)
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from test21 import (load, choose_rows_multi, detect_groups,
                                build_features, spec_matrix, LABELS)

SEED = 42
CLASSES = list(LABELS)


def make_model(algorithm):
    if algorithm == 'lightgbm':
        return LGBMClassifier(objective='multiclass', num_class=5,
                              n_estimators=350, learning_rate=0.04,
                              num_leaves=15, min_child_samples=45,
                              colsample_bytree=0.8, subsample=0.8,
                              subsample_freq=1, reg_lambda=2.0,
                              random_state=SEED, n_jobs=4, verbosity=-1)
    if algorithm == 'xgboost':
        return XGBClassifier(objective='multi:softprob', num_class=5,
                             n_estimators=150, learning_rate=0.08,
                             max_depth=5, min_child_weight=3,
                             subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, reg_alpha=0.0,
                             eval_metric='mlogloss', tree_method='hist',
                             random_state=SEED, n_jobs=4)
    raise ValueError(algorithm)


def prepare(args):
    d = args.data_dir
    print('Loading raw files...', flush=True)
    tr_op = load(d, 'train_operational_readouts.csv')
    va_op = load(d, 'validation_operational_readouts.csv')
    tr_spec = load(d, 'train_specifications.csv')
    va_spec = load(d, 'validation_specifications.csv')
    tte = load(d, 'train_tte.csv')
    labels = load(d, 'validation_labels.csv')
    tr_op = tr_op.sort_values(['vehicle_id', 'time_step']).reset_index(drop=True)
    va_op = va_op.sort_values(['vehicle_id', 'time_step']).reset_index(drop=True)
    raw_cols = [c for c in tr_op if c not in ('vehicle_id', 'time_step')]
    tr_sel, va_sel = choose_rows_multi(tr_op, tte, va_op, labels,
                                       args.k_repair, args.k_nonrepair)
    y_tr = tr_sel['class_label'].to_numpy(dtype=int)
    y_va = va_sel['class_label'].to_numpy(dtype=int)
    sw = tr_sel['sw'].to_numpy(dtype=float)
    lags = [int(s) for s in args.lags.split(',') if s]
    hist, singles = detect_groups(raw_cols)
    print('Building normalized features...', flush=True)
    tr_f, names, _, _ = build_features(tr_op, tr_sel, raw_cols, hist, singles, lags)
    va_f, va_names, _, _ = build_features(va_op, va_sel, raw_cols, hist, singles, lags)
    if list(names) != list(va_names):
        raise ValueError('Feature name/order mismatch')
    del tr_op, va_op, tte, labels
    gc.collect()
    st, sv, spec_names = spec_matrix(tr_spec, va_spec,
                                     tr_sel['vehicle_id'].to_numpy(),
                                     va_sel['vehicle_id'].to_numpy())
    x_tr = np.hstack([tr_f, st]).astype(np.float32)
    x_va = np.hstack([va_f, sv]).astype(np.float32)
    del tr_f, va_f, st, sv, tr_spec, va_spec
    gc.collect()
    if x_tr.shape[1] != x_va.shape[1]:
        raise ValueError('Train/validation feature width mismatch')
    print('Matrices:', x_tr.shape, x_va.shape, flush=True)
    print('Train counts:', np.bincount(y_tr, minlength=5), flush=True)
    print('Val counts:', np.bincount(y_va, minlength=5), flush=True)
    metadata = {'raw_cols': list(map(str, raw_cols)), 'lags': lags,
                'spec_columns': list(map(str, spec_names)),
                'feature_names': list(names) + [f'spec:{n}' for n in spec_names],
                'n_features': int(x_tr.shape[1]),
                'val_weight': args.val_weight,
                'k_repair': args.k_repair, 'k_nonrepair': args.k_nonrepair}
    return x_tr, x_va, y_tr, y_va, sw, va_sel['vehicle_id'].to_numpy(), metadata


def metrics(y, p):
    pred = np.asarray(CLASSES)[np.argmax(p, axis=1)]
    actual = (y != 0).astype(int)
    score = 1.0 - p[:, CLASSES.index(0)]
    if actual.sum() == 0 or actual.sum() == len(actual):
        raise ValueError('Need both risk and no-risk examples')
    order = np.argsort(-score, kind='stable')
    result = {
        'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=CLASSES, average='macro', zero_division=0)),
        'risk_ap': float(average_precision_score(actual, score)),
        'risk_roc_auc': float(roc_auc_score(actual, score)),
    }
    for cap in (0.025, 0.05, 0.10, 0.20):
        k = max(1, int(np.ceil(len(y) * cap)))
        found = int(actual[order[:k]].sum())
        result[f'recall_top_{cap:g}'] = float(found / actual.sum())
        result[f'precision_top_{cap:g}'] = float(found / k)
    return result


def compare(args, data):
    x_tr, x_va, y_tr, y_va, sw, ids, meta = data
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Validation must have one row per vehicle')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(x_va, y_va))
    summary = []
    out = pd.DataFrame({'vehicle_id': ids, 'actual_class': y_va})
    for alg in ('lightgbm', 'xgboost'):
        print(f'\n=== {alg.upper()} CROSS-FITTING ===', flush=True)
        oof = np.full((len(y_va), 5), np.nan, dtype=np.float32)
        for fold, (adapt_idx, hold_idx) in enumerate(folds, 1):
            # Only OTHER validation vehicles may be used for adaptation.
            x_fit = np.vstack((x_tr, x_va[adapt_idx]))
            y_fit = np.concatenate((y_tr, y_va[adapt_idx]))
            weights = np.concatenate((sw, np.full(len(adapt_idx), args.val_weight)))
            m = make_model(alg)
            m.fit(x_fit, y_fit, sample_weight=weights)
            if list(m.classes_) != CLASSES:
                raise ValueError(f'{alg} class order mismatch: {m.classes_}')
            oof[hold_idx] = m.predict_proba(x_va[hold_idx]).astype(np.float32)
            print(f'  Fold {fold}/5: {len(hold_idx)} held out', flush=True)
            del x_fit, y_fit, weights, m
            gc.collect()
        if not np.isfinite(oof).all():
            raise RuntimeError('Incomplete OOF predictions')
        result = {'algorithm': alg, **metrics(y_va, oof)}
        summary.append(result)
        for i in CLASSES:
            out[f'{alg}_p_c{i}'] = oof[:, CLASSES.index(i)]
        out[f'{alg}_risk'] = 1 - oof[:, CLASSES.index(0)]
        print(pd.Series(result).to_string(), flush=True)
        pred = np.asarray(CLASSES)[oof.argmax(axis=1)]
        print(classification_report(y_va, pred, labels=CLASSES, zero_division=0))
    summary_df = pd.DataFrame(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_dir / 'final_model_comparison.csv', index=False)
    out.to_csv(args.output_dir / 'final_model_oof_predictions.csv', index=False)
    print('\n=== MODEL COMPARISON ===\n', summary_df.to_string(index=False), flush=True)
    print('Exploratory OOF results; official validation reused during development.', flush=True)


def train_final(args, data):
    x_tr, x_va, y_tr, y_va, sw, ids, meta = data
    x = np.vstack((x_tr, x_va))
    y = np.concatenate((y_tr, y_va))
    weights = np.concatenate((sw, np.full(len(y_va), args.val_weight)))
    for alg in ('lightgbm', 'xgboost'):
        print('Training final:', alg, flush=True)
        m = make_model(alg)
        m.fit(x, y, sample_weight=weights)
        if list(m.classes_) != CLASSES:
            raise ValueError('Class order mismatch')
        target = args.output_dir / f'final_{alg}'
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(m, target / 'model.joblib')
        record = {**meta, 'algorithm': alg, 'model_class': type(m).__name__,
                  'performance_note': 'Model trained on all validation labels. Do not report in-sample metrics as independent performance.'}
        (target / 'meta.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
        print('Saved:', target, flush=True)
        del m
        gc.collect()


def predict(args):
    if not args.op_csv or not args.spec_csv:
        raise SystemExit('--mode predict requires --op-csv and --spec-csv')
    target = args.output_dir / f'final_{args.algorithm}'
    meta = json.loads((target / 'meta.json').read_text(encoding='utf-8'))
    m = joblib.load(target / 'model.joblib')
    op = pd.read_csv(args.op_csv)
    spec = pd.read_csv(args.spec_csv)
    raw_cols = meta['raw_cols']
    missing = set(raw_cols) - set(op.columns)
    if missing:
        raise ValueError(f'Missing raw features: {sorted(missing)[:10]}')
    op = op.sort_values(['vehicle_id', 'time_step']).reset_index(drop=True)
    last = op.groupby('vehicle_id', sort=False).tail(1)
    selected = pd.DataFrame({'vehicle_id': last['vehicle_id'].to_numpy(),
                             'row_id': last.index.to_numpy()})
    hist, singles = detect_groups(raw_cols)
    x_f, names, _, _ = build_features(op, selected, raw_cols, hist, singles, meta['lags'])
    if list(names) != meta['feature_names'][:len(names)]:
        raise ValueError('Feature name/order mismatch')
    feat = spec.set_index('vehicle_id')
    cats = [c for c in feat if not pd.api.types.is_numeric_dtype(feat[c])]
    feat = pd.get_dummies(feat, columns=cats, dtype=np.float32)
    feat = feat.reindex(columns=meta['spec_columns'], fill_value=0.0)
    if feat.index.has_duplicates:
        raise ValueError('Duplicate vehicle IDs in specification file')
    x_s = feat.reindex(selected['vehicle_id']).to_numpy(dtype=np.float32)
    x = np.hstack((x_f, x_s)).astype(np.float32)
    if x.shape[1] != meta['n_features']:
        raise ValueError('Feature width mismatch')
    p = m.predict_proba(x)
    if list(m.classes_) != CLASSES:
        raise ValueError('Class order mismatch')
    out = pd.DataFrame({'vehicle_id': selected['vehicle_id'].to_numpy()})
    for i in CLASSES:
        out[f'p_c{i}'] = p[:, CLASSES.index(i)]
    out['risk'] = 1 - p[:, CLASSES.index(0)]
    out['risk_rank'] = out['risk'].rank(ascending=False, method='first').astype(int)
    out['top_class'] = np.asarray(CLASSES[1:])[np.argmax(p[:, 1:], axis=1)]
    out = out.sort_values('risk_rank')
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print('Saved predictions:', args.out_csv, 'vehicles:', len(out), flush=True)


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('compare', 'predict'), default='compare')
    ap.add_argument('--data-dir', type=Path, default=root / 'raw')
    ap.add_argument('--output-dir', type=Path, default=root / 'processed')
    ap.add_argument('--k-repair', type=int, default=3)
    ap.add_argument('--k-nonrepair', type=int, default=2)
    ap.add_argument('--lags', default='5,20')
    ap.add_argument('--val-weight', type=float, default=10.0)
    ap.add_argument('--train-final', action='store_true')
    ap.add_argument('--algorithm', choices=('lightgbm', 'xgboost'), default='xgboost')
    ap.add_argument('--op-csv', type=Path)
    ap.add_argument('--spec-csv', type=Path)
    ap.add_argument('--out-csv', type=Path, default=root / 'processed' / 'predictions.csv')
    args = ap.parse_args()
    if args.mode == 'predict':
        predict(args)
    else:
        data = prepare(args)
        compare(args, data)
        if args.train_final:
            train_final(args, data)


if __name__ == '__main__':
    main()
