"""FleetCare Test 18: ordinal cumulative XGBoost vs ordinary multiclass.

Run alongside test_15_16gb.py:
    python test_18_ordinal_16gb.py
    python test_18_ordinal_16gb.py --features histogram

Uses exactly the sampled-row preprocessing from Test 15; no test-label training.
Official validation has been reused in prior experiments: exploratory only.
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier
from test_15_16gb import (
    load, histogram_groups, choose_rows, history_selected,
    add_histogram_summaries, add_specs, transformed, model,
    score_rows, SEED,
)

CLASSES = np.arange(5)


def ordinal_probabilities(cumulative):
    """Enforce P(Y>=1)>=...>=P(Y>=4) per row, then recover 5-class PMF."""
    cumulative = np.asarray(cumulative, dtype=np.float64)
    if cumulative.ndim != 2 or cumulative.shape[1] != 4:
        raise ValueError('Expected (n_samples, 4) cumulative probabilities')
    q = np.minimum.accumulate(np.clip(cumulative, 0.0, 1.0), axis=1)
    p = np.column_stack([
        1 - q[:, 0],
        q[:, 0] - q[:, 1],
        q[:, 1] - q[:, 2],
        q[:, 2] - q[:, 3],
        q[:, 3],
    ])
    if np.any(p < -1e-10) or not np.allclose(p.sum(axis=1), 1.0):
        raise AssertionError('Ordinal probabilities are invalid')
    return np.maximum(p, 0.0)


def evaluate(name, y_true, probabilities):
    pred = probabilities.argmax(axis=1)
    macro = f1_score(y_true, pred, labels=CLASSES, average='macro', zero_division=0)
    c4 = (y_true == 4).astype(int)
    scores = probabilities[:, 4]
    ranking = score_rows(name, scores, c4)
    top5 = next(r for r in ranking if np.isclose(r['capacity'], 0.05))
    summary = {
        'experiment': name,
        'macro_f1': macro,
        'accuracy': float((pred == y_true).mean()),
        'c4_class_recall': float(((pred == 4) & (y_true == 4)).sum() / max(1, (y_true == 4).sum())),
        'risk6_ap': average_precision_score(c4, scores),
        'risk6_roc_auc': roc_auc_score(c4, scores),
        'c4_recall_at_5pct': top5['recall_at_capacity'],
    }
    print('\n===', name, '===', flush=True)
    print(pd.Series(summary).to_string(), flush=True)
    print(classification_report(y_true, pred, labels=CLASSES, zero_division=0, digits=4), flush=True)
    print('Confusion matrix:\n', confusion_matrix(y_true, pred, labels=CLASSES), flush=True)
    return summary, ranking, pred


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=root / 'raw')
    parser.add_argument('--output-dir', type=Path, default=root / 'processed')
    parser.add_argument('--features', choices=['original', 'histogram'], default='original')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading data...', flush=True)
    train_op = load(args.data_dir, 'train_operational_readouts.csv')
    val_op = load(args.data_dir, 'validation_operational_readouts.csv')
    train_spec = load(args.data_dir, 'train_specifications.csv')
    val_spec = load(args.data_dir, 'validation_specifications.csv')
    tte = load(args.data_dir, 'train_tte.csv')
    labels = load(args.data_dir, 'validation_labels.csv')
    train_op = train_op.sort_values(['vehicle_id', 'time_step']).reset_index(drop=True)
    val_op = val_op.sort_values(['vehicle_id', 'time_step']).reset_index(drop=True)
    for name, df in [('train', train_op), ('validation', val_op)]:
        if df.duplicated(['vehicle_id', 'time_step']).any():
            raise ValueError(f'Duplicate vehicle/time_step rows: {name}')
    raw_cols = [c for c in train_op if c not in ('vehicle_id', 'time_step')]
    if set(raw_cols) - set(val_op.columns):
        raise ValueError('Validation lacks raw feature columns')
    train_sel, val_sel = choose_rows(train_op, tte, val_op, labels)
    y_train = train_sel['class_label'].to_numpy(dtype=int)
    y_val = val_sel['class_label'].to_numpy(dtype=int)
    print('Training:', np.bincount(y_train, minlength=5), flush=True)
    print('Validation:', np.bincount(y_val, minlength=5), flush=True)

    train_feat = history_selected(train_op, train_sel, raw_cols, 'TRAIN')
    val_feat = history_selected(val_op, val_sel, raw_cols, 'VALIDATION')
    if args.features == 'histogram':
        groups = histogram_groups(raw_cols)
        train_feat = add_histogram_summaries(train_feat, train_op, train_sel, groups)
        val_feat = add_histogram_summaries(val_feat, val_op, val_sel, groups)
    train_feat = add_specs(train_feat, train_sel, train_spec)
    val_feat = add_specs(val_feat, val_sel, val_spec)
    X_train, X_val = transformed(train_feat, val_feat)
    print('Matrices:', X_train.shape, X_val.shape, flush=True)
    del train_op, val_op, train_feat, val_feat
    gc.collect()

    print('\nTraining multiclass baseline...', flush=True)
    baseline = model(binary=False)
    baseline.fit(X_train, y_train)
    baseline_proba = baseline.predict_proba(X_val)
    baseline_proba = baseline_proba[:, [list(baseline.classes_).index(i) for i in CLASSES]]
    del baseline
    gc.collect()

    cumulative = np.empty((len(y_val), 4), dtype=np.float64)
    for threshold in range(1, 5):
        target = (y_train >= threshold).astype(np.int8)
        print(f'Training ordinal threshold Y >= C{threshold}: '
              f'{int(target.sum())}/{len(target)} positives', flush=True)
        if len(np.unique(target)) != 2:
            raise ValueError(f'Only one class for threshold {threshold}')
        binary = model(binary=True)
        binary.fit(X_train, target)
        cumulative[:, threshold - 1] = binary.predict_proba(X_val)[:, list(binary.classes_).index(1)]
        del binary
        gc.collect()

    ordinal_proba = ordinal_probabilities(cumulative)
    summaries, all_rankings = [], []
    preds = pd.DataFrame({'vehicle_id': val_sel['vehicle_id'].to_numpy(), 'actual_class': y_val})
    for name, p in [('multiclass_baseline', baseline_proba), ('ordinal_cumulative', ordinal_proba)]:
        summary, rankings, y_pred = evaluate(name, y_val, p)
        summaries.append(summary)
        all_rankings.extend(rankings)
        preds[f'{name}_predicted_class'] = y_pred
        for i in CLASSES:
            preds[f'{name}_pC{i}'] = p[:, i]

    suffix = args.features
    summary_df = pd.DataFrame(summaries)
    capacity_df = pd.DataFrame(all_rankings)
    summary_df.to_csv(args.output_dir / f'test_18_{suffix}_summary.csv', index=False)
    capacity_df.to_csv(args.output_dir / f'test_18_{suffix}_capacity.csv', index=False)
    preds.to_csv(args.output_dir / f'test_18_{suffix}_predictions.csv', index=False)
    print('\n=== TEST 18 SUMMARY ===\n', summary_df.round(4).to_string(index=False), flush=True)
    print('\n=== C4 RECALL BY CAPACITY ===\n', capacity_df.pivot(index='capacity', columns='experiment', values='recall_at_capacity').round(4).to_string(), flush=True)
    print('\nSaved in:', args.output_dir, flush=True)
    print('WARNING: Official validation was previously reused for model selection; results are exploratory.', flush=True)


if __name__ == '__main__':
    main()
