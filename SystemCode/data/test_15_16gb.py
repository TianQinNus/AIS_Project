"""FleetCare Test 16: LightGBM histogram ablation (16GB-friendly).
No future readouts used. Same sampled rows for all four experiments.
Usage: python test_15_16gb.py [--data-dir raw] [--output-dir processed]
"""
import argparse
import gc
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

SEED = 42
CAPACITIES = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
HORIZON = 6
EXCLUDE = {'vehicle_id', 'time_step', 'class_label', 'in_study_repair', 'length_of_study_time_step', 'time_to_endpoint'}


def load(folder, name):
    p = folder / name
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def histogram_groups(cols):
    d = {}
    for col in cols:
        m = re.fullmatch(r'(\d+)_(\d+)', col)
        if m:
            d.setdefault(m.group(1), []).append((int(m.group(2)), col))
    return {k: [c for _, c in sorted(v)] for k, v in d.items() if len(v) > 1}


def label_from_tte(t):
    # Matches Test 13 boundary convention.
    return np.select([t <= 6, t <= 12, t <= 24, t <= 48], [4, 3, 2, 1], default=0)


def choose_rows(train_op, tte, val_op, val_labels):
    # The source readouts are sorted only once; positional indices are stable.
    tr = train_op[['vehicle_id', 'time_step']].merge(tte, on='vehicle_id', how='left', validate='many_to_one')
    if tr[['in_study_repair', 'length_of_study_time_step']].isna().any().any():
        raise ValueError('Missing TTE metadata for some training vehicles')
    remaining = tr['length_of_study_time_step'] - tr['time_step']
    repaired = tr['in_study_repair'].eq(1) & remaining.ge(0)
    safe = tr['in_study_repair'].eq(0) & remaining.gt(48)
    eligible = repaired | safe
    label = np.full(len(tr), -1, dtype=np.int8)
    label[repaired.to_numpy()] = label_from_tte(remaining[repaired].to_numpy()).astype(np.int8)
    label[safe.to_numpy()] = 0
    subset = tr.loc[eligible, ['vehicle_id']].copy()
    subset['class_label'] = label[eligible.to_numpy()]
    subset['row_id'] = np.flatnonzero(eligible.to_numpy())
    repair_subset = subset.loc[tr.loc[eligible, 'in_study_repair'].to_numpy() == 1]
    nonrepair_subset = subset.loc[tr.loc[eligible, 'in_study_repair'].to_numpy() == 0]
    repair_pick = repair_subset.groupby(['vehicle_id', 'class_label'], sort=False).sample(n=1, random_state=SEED)
    nonrepair_pick = nonrepair_subset.groupby('vehicle_id', sort=False).sample(n=1, random_state=SEED)
    selected = pd.concat([repair_pick, nonrepair_pick], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    if selected['row_id'].duplicated().any():
        raise ValueError('Duplicate training row selection')
    val_rows = val_op[['vehicle_id', 'time_step']].copy()
    val_rows['row_id'] = np.arange(len(val_rows))
    val_last = val_rows.groupby('vehicle_id', sort=False).tail(1).copy()
    val_last = val_last.merge(val_labels, on='vehicle_id', how='left', validate='one_to_one')
    if val_last['class_label'].isna().any():
        raise ValueError('Missing validation labels')
    return selected, val_last


def history_selected(readouts, selected, raw_cols, label):
    """Compute history for ONE raw column at a time; materialize ONLY selected rows.
    Each rolling operation sees all earlier readouts of the same vehicle.
    """
    idx = selected['row_id'].to_numpy(dtype=np.int64)
    vids = readouts['vehicle_id']
    time = pd.to_numeric(readouts['time_step'], errors='raise')
    gap = time.groupby(vids, sort=False).diff().replace(0, np.nan)
    out = {'time_step': time.iloc[idx].to_numpy(dtype=np.float32),
           'time_gap': gap.iloc[idx].to_numpy(dtype=np.float32)}
    print(f'[{label}] extracting {len(raw_cols)} raw features for {len(idx)} selected rows', flush=True)
    for i, col in enumerate(raw_cols, 1):
        x = pd.to_numeric(readouts[col], errors='coerce').astype(np.float32)
        g = x.groupby(vids, sort=False)
        prev = g.shift(1)
        delta = x - prev
        values = {
            col: x,
            f'{col}__prev': prev,
            f'{col}__delta': delta,
            f'{col}__rate': delta / gap,
            f'{col}__change3': x - g.shift(3),
            f'{col}__change5': x - g.shift(5),
        }
        for w in (3, 5):
            rolling = g.rolling(w, min_periods=1)
            mean = rolling.mean().reset_index(level=0, drop=True)
            std = g.rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
            values[f'{col}__mean{w}'] = mean
            values[f'{col}__std{w}'] = std
        for name, s in values.items():
            v = s.iloc[idx].to_numpy(dtype=np.float32, copy=True)
            v[~np.isfinite(v)] = np.nan
            out[name] = v
        del x, g, prev, delta, values
        if i % 15 == 0 or i == len(raw_cols):
            print(f'[{label}] {i}/{len(raw_cols)} columns completed', flush=True)
    return pd.DataFrame(out)


def add_histogram_summaries(frame, readouts, selected, groups):
    """Only three compact summary features per histogram, for selected rows."""
    idx = selected['row_id'].to_numpy(dtype=np.int64)
    added = {}
    for key, cols in groups.items():
        vals = readouts.iloc[idx][cols].to_numpy(dtype=np.float32, copy=True)
        total = vals.sum(axis=1)
        valid = np.isfinite(vals).all(axis=1) & (vals >= 0).all(axis=1) & (total > 0)
        p = np.zeros_like(vals)
        p[valid] = vals[valid] / total[valid, None]
        pos = np.arange(len(cols), dtype=np.float32)
        mean_bin = p @ pos
        entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
        concentration = p.max(axis=1)
        for name, a in ((f'hist_{key}__mean_bin', mean_bin),
                        (f'hist_{key}__entropy', entropy),
                        (f'hist_{key}__max_ratio', concentration)):
            a = a.astype(np.float32)
            a[~valid] = np.nan
            added[name] = a
        del vals, p
    return pd.concat([frame, pd.DataFrame(added)], axis=1)


def add_specs(frame, selected, specs):
    ids = selected[['vehicle_id']].reset_index(drop=True)
    merged = ids.merge(specs, on='vehicle_id', how='left', validate='many_to_one', sort=False)
    if len(merged) != len(frame):
        raise ValueError('Specification join changed row count')
    return pd.concat([frame.reset_index(drop=True), merged.drop(columns='vehicle_id')], axis=1)


def transformed(train, val):
    if not train.columns.equals(val.columns):
        raise ValueError('Train/validation feature columns differ')
    categorical = [c for c in train if c.startswith('Spec_')]
    numeric = [c for c in train if c not in categorical]
    transformer = ColumnTransformer([
        ('num', SimpleImputer(strategy='median', keep_empty_features=True), numeric),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('enc', OneHotEncoder(handle_unknown='ignore'))]), categorical),
    ], sparse_threshold=0)
    a = transformer.fit_transform(train)
    b = transformer.transform(val)
    return np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)


def model(binary):
    # Fixed settings across both feature sets; LightGBM-specific hyperparameters.
    common = dict(n_estimators=350, learning_rate=0.04, num_leaves=15,
                  max_depth=-1, min_child_samples=45,
                  colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                  reg_lambda=2.0, verbosity=-1, random_state=SEED, n_jobs=4)
    if binary:
        return LGBMClassifier(objective='binary', **common)
    return LGBMClassifier(objective='multiclass', num_class=5, **common)


def score_rows(name, scores, truth):
    ap = average_precision_score(truth, scores)
    auc = roc_auc_score(truth, scores)
    order = np.argsort(-scores, kind='stable')
    total = int(truth.sum())
    return [{'experiment': name, 'capacity': cap, 'selected': k,
             'c4_found': int(truth[order[:k]].sum()),
             'precision_at_capacity': float(truth[order[:k]].mean()),
             'recall_at_capacity': float(truth[order[:k]].sum() / total),
             'risk6_ap': ap, 'risk6_roc_auc': auc}
            for cap in CAPACITIES for k in [max(1, int(np.ceil(len(truth) * cap)))]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'raw')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'processed')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading CSVs...', flush=True)
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
            raise ValueError(f'{name} contains duplicate vehicle/time_step rows')
    raw_cols = [c for c in train_op if c not in ('vehicle_id', 'time_step')]
    if set(raw_cols) - set(val_op):
        raise ValueError('Validation lacks raw training features')
    groups = histogram_groups(raw_cols)
    print('Histogram groups:', {k: len(v) for k, v in groups.items()}, flush=True)
    train_sel, val_sel = choose_rows(train_op, tte, val_op, labels)
    print('Training distribution:', train_sel['class_label'].value_counts().sort_index().to_dict(), flush=True)
    y_multi = train_sel['class_label'].to_numpy(dtype=int)
    y_binary = (y_multi == 4).astype(int)
    truth = (val_sel['class_label'].to_numpy(dtype=int) == 4).astype(int)
    print('Validation C4:', truth.sum(), '/', len(truth), flush=True)
    if truth.sum() == 0 or truth.sum() == len(truth):
        raise ValueError('Validation needs both positive and negative samples')

    # Original history is computed only once and reused across both feature sets.
    train_base = history_selected(train_op, train_sel, raw_cols, 'TRAIN')
    val_base = history_selected(val_op, val_sel, raw_cols, 'VALIDATION')
    print('History extraction finished; training matrices now have sampled rows only.', flush=True)
    results, preds = [], pd.DataFrame({'vehicle_id': val_sel['vehicle_id'].to_numpy(),
                                        'actual_class': val_sel['class_label'].to_numpy(),
                                        'actual_c4': truth})
    for feat in ('original', 'histogram'):
        print(f'\n=== Feature set: {feat} ===', flush=True)
        if feat == 'histogram':
            tr = add_histogram_summaries(train_base, train_op, train_sel, groups)
            va = add_histogram_summaries(val_base, val_op, val_sel, groups)
        else:
            tr, va = train_base, val_base
        tr = add_specs(tr, train_sel, train_spec)
        va = add_specs(va, val_sel, val_spec)
        Xtr, Xva = transformed(tr, va)
        print('Model matrix shapes:', Xtr.shape, Xva.shape, flush=True)
        del tr, va
        gc.collect()
        for target in ('multiclass', 'binary'):
            name = f'{feat}_{target}_lightgbm'
            print('Training', name, flush=True)
            m = model(binary=(target == 'binary'))
            m.fit(Xtr, y_binary if target == 'binary' else y_multi)
            p = m.predict_proba(Xva)
            scores = p[:, 1] if target == 'binary' else p[:, list(m.classes_).index(4)]
            preds[name] = scores
            rows = score_rows(name, scores, truth)
            results.extend(rows)
            print(f"{name}: AP={rows[0]['risk6_ap']:.4f}, AUC={rows[0]['risk6_roc_auc']:.4f}, "
                  f"Recall@5%={next(r['recall_at_capacity'] for r in rows if r['capacity'] == 0.05):.4f}", flush=True)
            del m, p, scores
            gc.collect()
        del Xtr, Xva
        gc.collect()
    table = pd.DataFrame(results)
    summary = table.drop_duplicates('experiment')[['experiment', 'risk6_ap', 'risk6_roc_auc']]
    print('\n=== TEST 16 LIGHTGBM SUMMARY ===\n', summary.to_string(index=False), flush=True)
    print('\n=== C4 RECALL BY CAPACITY ===\n', table.pivot(index='capacity', columns='experiment', values='recall_at_capacity').round(4).to_string(), flush=True)
    previous = args.output_dir / 'test_15_summary.csv'
    if previous.exists():
        xgb = pd.read_csv(previous).copy()
        xgb['model_family'] = 'XGBoost'
        summary_compare = summary.copy()
        summary_compare['model_family'] = 'LightGBM'
        summary_compare['experiment'] = summary_compare['experiment'].str.replace('_lightgbm', '', regex=False)
        comparison = pd.concat([xgb, summary_compare], ignore_index=True)
        comparison.to_csv(args.output_dir / 'test_16_vs_xgboost.csv', index=False)
        print('\n=== XGBOOST VS LIGHTGBM ===\n', comparison.to_string(index=False), flush=True)
    summary.to_csv(args.output_dir / 'test_16_lightgbm_summary.csv', index=False)
    table.to_csv(args.output_dir / 'test_16_lightgbm_capacity_results.csv', index=False)
    preds.to_csv(args.output_dir / 'test_16_lightgbm_predictions.csv', index=False)
    print('\nSaved results in:', args.output_dir, flush=True)

if __name__ == '__main__':
    main()
