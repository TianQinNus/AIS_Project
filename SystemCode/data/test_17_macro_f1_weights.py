"""FleetCare Test 17: post-hoc multiclass decision-weight experiment.

Uses the memory-optimized Test 15 feature pipeline. Retrains the original
multiclass XGBoost once because Test 15 CSVs save P(C4) only, not all 5 columns.

IMPORTANT: The official validation set has already been used repeatedly.
The held-out split below is a diagnostic split within that set, NOT an
independent final test. Report results as exploratory only.

Run next to test_15_16gb.py:
  python test_17_macro_f1_weights.py
  python test_17_macro_f1_weights.py --features histogram
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from test_15_16gb import (
    load, histogram_groups, choose_rows, history_selected,
    add_histogram_summaries, add_specs, transformed, model,
)

LABELS = [0, 1, 2, 3, 4]


def predict_weighted(probabilities, log_weights):
    # Fix C0 weight = 1; only relative weights matter.
    weights = np.exp(np.r_[0.0, log_weights])
    return np.argmax(probabilities * weights[None, :], axis=1)


def macro_f1(y, pred):
    return f1_score(y, pred, labels=LABELS, average='macro', zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument('--data-dir', type=Path, default=root / 'raw')
    parser.add_argument('--output-dir', type=Path, default=root / 'processed')
    parser.add_argument('--features', choices=['original', 'histogram'], default='original')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--maxiter', type=int, default=60)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading files...', flush=True)
    train_op = load(args.data_dir, 'train_operational_readouts.csv')
    val_op = load(args.data_dir, 'validation_operational_readouts.csv')
    train_spec = load(args.data_dir, 'train_specifications.csv')
    val_spec = load(args.data_dir, 'validation_specifications.csv')
    tte = load(args.data_dir, 'train_tte.csv')
    labels = load(args.data_dir, 'validation_labels.csv')
    train_op = train_op.sort_values(['vehicle_id', 'time_step']).reset_index(drop=True)
    val_op = val_op.sort_values(['vehicle_id', 'time_step']).reset_index(drop=True)
    for name, frame in [('train', train_op), ('validation', val_op)]:
        if frame.duplicated(['vehicle_id', 'time_step']).any():
            raise ValueError(f'Duplicate vehicle/time_step in {name}')
    raw_cols = [c for c in train_op if c not in ('vehicle_id', 'time_step')]
    if set(raw_cols) - set(val_op.columns):
        raise ValueError('Missing validation features')
    train_sel, val_sel = choose_rows(train_op, tte, val_op, labels)
    y_train = train_sel['class_label'].to_numpy(dtype=int)
    y_val = val_sel['class_label'].to_numpy(dtype=int)
    print('Train class counts:', np.bincount(y_train, minlength=5), flush=True)
    print('Validation class counts:', np.bincount(y_val, minlength=5), flush=True)

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

    classifier = model(binary=False)
    print('Training XGBoost multiclass once...', flush=True)
    classifier.fit(X_train, y_train)
    probabilities = classifier.predict_proba(X_val)
    if list(classifier.classes_) != LABELS:
        raise ValueError(f'Unexpected class order: {classifier.classes_}')
    if probabilities.shape != (len(y_val), 5):
        raise ValueError('Expected 5-class probabilities')

    # Split at vehicle level: official validation has one row per vehicle.
    # Stratification protects tiny minority classes but leaves very few examples.
    tune_idx, hold_idx = train_test_split(
        np.arange(len(y_val)), test_size=0.5,
        random_state=args.seed, stratify=y_val,
    )
    y_tune, p_tune = y_val[tune_idx], probabilities[tune_idx]
    y_hold, p_hold = y_val[hold_idx], probabilities[hold_idx]
    print('Tune counts:', np.bincount(y_tune, minlength=5))
    print('Holdout counts:', np.bincount(y_hold, minlength=5))

    baseline_tune = macro_f1(y_tune, np.argmax(p_tune, axis=1))
    baseline_hold = macro_f1(y_hold, np.argmax(p_hold, axis=1))

    def objective(log_weights):
        return -macro_f1(y_tune, predict_weighted(p_tune, log_weights))

    # Conservative search bounds: each minority weight 0.25..4 relative to C0.
    result = differential_evolution(
        objective, bounds=[(np.log(0.25), np.log(4.0))] * 4,
        seed=args.seed, maxiter=args.maxiter,
        popsize=8, polish=False, workers=1,
    )
    weights = np.exp(np.r_[0.0, result.x])
    pred_tune = predict_weighted(p_tune, result.x)
    pred_hold = predict_weighted(p_hold, result.x)
    tuned_tune = macro_f1(y_tune, pred_tune)
    tuned_hold = macro_f1(y_hold, pred_hold)

    print('\n=== EXPLORATORY MACRO F1 RESULTS ===')
    print('Weights C0..C4:', np.round(weights, 4))
    print(f'Tuning split:  argmax={baseline_tune:.4f} adjusted={tuned_tune:.4f}')
    print(f'Held-out split: argmax={baseline_hold:.4f} adjusted={tuned_hold:.4f}')
    print('\nHeld-out argmax report:')
    print(classification_report(y_hold, np.argmax(p_hold, axis=1), labels=LABELS, zero_division=0))
    print('\nHeld-out adjusted report:')
    print(classification_report(y_hold, pred_hold, labels=LABELS, zero_division=0))
    print('\nHeld-out adjusted confusion matrix:')
    print(confusion_matrix(y_hold, pred_hold, labels=LABELS))

    prefix = f'test_17_{args.features}'
    out = args.output_dir
    pd.DataFrame({
        'vehicle_id': val_sel.iloc[hold_idx]['vehicle_id'].to_numpy(),
        'actual_class': y_hold,
        'argmax_class': np.argmax(p_hold, axis=1),
        'adjusted_class': pred_hold,
        **{f'p_c{i}': p_hold[:, i] for i in LABELS},
    }).to_csv(out / f'{prefix}_holdout_predictions.csv', index=False)
    summary = {
        'features': args.features,
        'seed': args.seed,
        'tuning_n': int(len(tune_idx)),
        'holdout_n': int(len(hold_idx)),
        'weights': [float(w) for w in weights],
        'tuning_argmax_macro_f1': float(baseline_tune),
        'tuning_adjusted_macro_f1': float(tuned_tune),
        'holdout_argmax_macro_f1': float(baseline_hold),
        'holdout_adjusted_macro_f1': float(tuned_hold),
        'warning': 'Exploratory only: official validation set used previously for model development. Not an untouched test.',
    }
    (out / f'{prefix}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nSaved results to:', out, flush=True)


if __name__ == '__main__':
    main()
