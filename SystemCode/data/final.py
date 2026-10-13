"""FleetCare final XGBoost model: train on training + weighted validation rows.

Requires test21.py with load, choose_rows_multi, detect_groups,
build_features, spec_matrix, LABELS.

Training:
  python final_model_xgboost.py
  python final_model_xgboost.py --val-weight 10

Prediction:
  python final_model_xgboost.py --mode predict --op-csv raw/test_operational_readouts.csv --spec-csv raw/test_specifications.csv

IMPORTANT: Validation rows are included in final training. Do not report
in-sample metrics from this model as independent evaluation results.
"""
import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from test21 import load, choose_rows_multi, detect_groups, build_features, spec_matrix, LABELS


def make_model():
    return XGBClassifier(
        objective='multi:softprob', num_class=5,
        n_estimators=150, learning_rate=0.08,
        max_depth=5, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0,
        eval_metric='mlogloss', tree_method='hist',
        random_state=42, n_jobs=4,
    )


def spec_for(spec_df, ids, s_names):
    if spec_df['vehicle_id'].duplicated().any():
        raise ValueError('Duplicate vehicle_id in specifications')
    feat = spec_df.set_index('vehicle_id')
    cat_cols = [c for c in feat if not pd.api.types.is_numeric_dtype(feat[c])]
    feat = pd.get_dummies(feat, columns=cat_cols, dtype=np.float32)
    feat = feat.reindex(columns=s_names, fill_value=0.0).astype(np.float32)
    aligned = feat.reindex(ids)
    if aligned.isna().all(axis=1).any():
        print('Warning: some vehicles may have missing specification values')
    return aligned.to_numpy(dtype=np.float32)


def train(args):
    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    print('Loading files...', flush=True)
    train_op = load(args.data_dir, 'train_operational_readouts.csv')
    val_op = load(args.data_dir, 'validation_operational_readouts.csv')
    train_spec = load(args.data_dir, 'train_specifications.csv')
    val_spec = load(args.data_dir, 'validation_specifications.csv')
    tte = load(args.data_dir, 'train_tte.csv')
    labels = load(args.data_dir, 'validation_labels.csv')
    train_op = train_op.sort_values(['vehicle_id', 'time_step']).reset_index(drop=True)
    val_op = val_op.sort_values(['vehicle_id', 'time_step']).reset_index(drop=True)
    raw_cols = [c for c in train_op if c not in ('vehicle_id', 'time_step')]

    train_sel, val_sel = choose_rows_multi(
        train_op, tte, val_op, labels, args.k_repair, args.k_nonrepair
    )
    y_train = train_sel['class_label'].to_numpy(dtype=int)
    y_val = val_sel['class_label'].to_numpy(dtype=int)
    sw = train_sel['sw'].to_numpy(dtype=float)
    print('Train rows:', len(y_train), np.bincount(y_train, minlength=5), flush=True)
    print('Val rows:', len(y_val), np.bincount(y_val, minlength=5), flush=True)

    lags = [int(x) for x in args.lags.split(',') if x]
    hist, singles = detect_groups(raw_cols)
    Xtr_f, names, _, _ = build_features(train_op, train_sel, raw_cols, hist, singles, lags)
    Xva_f, _, _, _ = build_features(val_op, val_sel, raw_cols, hist, singles, lags)
    del train_op, val_op
    gc.collect()

    s_tr, s_va, s_names = spec_matrix(
        train_spec, val_spec,
        train_sel['vehicle_id'].to_numpy(), val_sel['vehicle_id'].to_numpy()
    )
    X_train = np.hstack([Xtr_f, s_tr]).astype(np.float32)
    X_val = np.hstack([Xva_f, s_va]).astype(np.float32)
    feat_names = names + [f'spec:{n}' for n in s_names]
    del Xtr_f, Xva_f, s_tr, s_va
    gc.collect()

    X = np.vstack([X_train, X_val])
    y = np.r_[y_train, y_val]
    w = np.r_[sw, np.full(len(y_val), args.val_weight, dtype=float)]
    del X_train, X_val
    gc.collect()
    print('Final matrix:', X.shape, 'validation weight:', args.val_weight, flush=True)

    m = make_model()
    print('Training final XGBoost...', flush=True)
    m.fit(X, y, sample_weight=w)
    from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
    )

    p = m.predict_proba(X)
    y_pred = m.predict(X)

    print("\n=== FINAL MODEL TRAINING METRICS ===")

    print("Accuracy:", accuracy_score(y, y_pred))
    print("Macro F1:", f1_score(y, y_pred, average="macro"))
    print("Weighted F1:", f1_score(y, y_pred, average="weighted"))

    print("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    # Risk6 = C4
    y_c4 = (y == 4).astype(int)
    risk6 = p[:, 4]

    print("\nRisk6 ROC-AUC:", roc_auc_score(y_c4, risk6))
    print("Risk6 AP:", average_precision_score(y_c4, risk6))

    # Top 5% Recall
    k = int(np.ceil(len(y) * 0.05))
    top_idx = np.argsort(-risk6)[:k]

    recall_at_5 = y_c4[top_idx].sum() / y_c4.sum()

    print("C4 Recall@Top5%:", recall_at_5)
    if list(m.classes_) != list(LABELS):
        raise ValueError(f'Unexpected class order: {m.classes_}')
    joblib.dump(m, model_dir / 'model.joblib')
    m.save_model(str(model_dir / 'model.json'))
    meta = {
        'model_family': 'XGBoost', 'model_params': m.get_params(),
        'raw_cols': [str(c) for c in raw_cols],
        'lags': lags, 'spec_columns': list(s_names),
        'feature_names': list(feat_names), 'n_features': int(X.shape[1]),
        'val_weight': float(args.val_weight),
        'k_repair': args.k_repair, 'k_nonrepair': args.k_nonrepair,
        'train_rows': int(len(y_train)), 'val_rows': int(len(y_val)),
        'evaluation_warning': 'Validation rows were used for training. Do not evaluate this final model on those rows as independent test data. Any previously reported cross-fitting metrics refer to a separate evaluation procedure, not this fitted artifact.'
    }
    (model_dir / 'meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print('Saved XGBoost model to:', model_dir, flush=True)


def predict(args):
    meta = json.loads((args.model_dir / 'meta.json').read_text(encoding='utf-8'))
    if meta.get('model_family') != 'XGBoost':
        raise ValueError('Saved model metadata does not identify XGBoost')
    m = joblib.load(args.model_dir / 'model.joblib')
    raw_cols = meta['raw_cols']
    op = pd.read_csv(args.op_csv)
    spec = pd.read_csv(args.spec_csv)
    missing = set(raw_cols) - set(op.columns)
    if missing:
        raise ValueError(f'Missing raw columns: {sorted(missing)[:8]}')
    op = op.sort_values(['vehicle_id', 'time_step']).reset_index(drop=True)
    last = op.groupby('vehicle_id', sort=False).tail(1)
    sel = pd.DataFrame({'vehicle_id': last['vehicle_id'].to_numpy(), 'row_id': last.index.to_numpy()})
    hist, singles = detect_groups(raw_cols)
    X_f, names, _, _ = build_features(op, sel, raw_cols, hist, singles, meta['lags'])
    X_s = spec_for(spec, sel['vehicle_id'].to_numpy(), meta['spec_columns'])
    X = np.hstack([X_f, X_s]).astype(np.float32)
    expected_names = list(names) + [f'spec:{n}' for n in meta['spec_columns']]
    if expected_names != meta['feature_names']:
        raise ValueError('Feature order or names differ from training')
    if X.shape[1] != meta['n_features']:
        raise ValueError(f'Feature count mismatch: {X.shape[1]} vs {meta["n_features"]}')
    p = m.predict_proba(X)
    if list(m.classes_) != list(LABELS):
        raise ValueError('Unexpected model class order')
    risk = 1 - p[:, 0]
    out = pd.DataFrame({
        'vehicle_id': sel['vehicle_id'].to_numpy(),
        **{f'p_c{i}': p[:, i] for i in LABELS},
        'risk': risk,
    })
    out['risk_rank'] = out['risk'].rank(ascending=False, method='first').astype(int)
    out['top_class'] = 1 + np.argmax(p[:, 1:], axis=1)
    out = out.sort_values('risk_rank')
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print('Predictions saved:', args.out_csv, '| vehicles:', len(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument('--mode', choices=['train', 'predict'], default='train')
    ap.add_argument('--data-dir', type=Path, default=root / 'raw')
    ap.add_argument('--model-dir', type=Path, default=root / 'processed' / 'final_model_xgboost')
    ap.add_argument('--k-repair', type=int, default=3)
    ap.add_argument('--k-nonrepair', type=int, default=2)
    ap.add_argument('--lags', type=str, default='5,20')
    ap.add_argument('--val-weight', type=float, default=10.0)
    ap.add_argument('--op-csv', type=Path)
    ap.add_argument('--spec-csv', type=Path)
    ap.add_argument('--out-csv', type=Path, default=root / 'processed' / 'predictions_xgboost.csv')
    args = ap.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        if not (args.op_csv and args.spec_csv):
            ap.error('predict mode requires --op-csv and --spec-csv')
        predict(args)


if __name__ == '__main__':
    main()
