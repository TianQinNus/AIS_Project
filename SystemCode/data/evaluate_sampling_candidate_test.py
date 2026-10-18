#!/usr/bin/env python3
"""Post-hoc test evaluation of saved sampling-ablation candidate. No training or DB writes.
Place next to fleetcare_sampling_ablation.py.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fleetcare_sampling_ablation import read, labels, make_features, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('raw'))
    parser.add_argument('--model-dir', type=Path, default=Path('processed/final_model_xgboost'))
    parser.add_argument('--candidate-dir', type=Path, default=Path('processed/sampling_ablation'))
    parser.add_argument('--output-dir', type=Path, default=Path('processed/sampling_ablation/test_only'))
    parser.add_argument('--batch-vehicles', type=int, default=250)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.model_dir / 'meta.json').read_text(encoding='utf-8'))
    candidate_meta_path = args.candidate_dir / 'sampling_candidate_meta.json'
    if candidate_meta_path.exists():
        candidate_meta = json.loads(candidate_meta_path.read_text(encoding='utf-8'))
        if candidate_meta.get('feature_names') != meta.get('feature_names'):
            raise ValueError('Candidate and original feature orders do not match')
        print('Candidate sampling strategy:', candidate_meta.get('sampling_strategy'))

    model_path = args.candidate_dir / 'sampling_candidate_model.joblib'
    if not model_path.is_file():
        raise FileNotFoundError(f'Missing candidate model: {model_path}')
    model = joblib.load(model_path)
    if getattr(model, 'n_features_in_', None) != meta['n_features']:
        raise ValueError('Model input dimension does not match metadata')
    if list(model.classes_) != [0, 1, 2, 3, 4]:
        raise ValueError(f'Unexpected classes: {model.classes_}')
    print('Loaded saved model:', model_path, flush=True)

    op = read(args.data_dir / 'test_operational_readouts.csv')
    op = op.sort_values(['vehicle_id', 'time_step'], kind='stable').reset_index(drop=True)
    spec = read(args.data_dir / 'test_specifications.csv')
    selected = op.groupby('vehicle_id', sort=False).tail(1)[['vehicle_id', 'time_step']].reset_index(drop=True)
    selected = selected.merge(labels(args.data_dir / 'test_labels.csv'), on='vehicle_id', how='left', validate='one_to_one')
    if selected.class_label.isna().any():
        raise ValueError('Test labels missing for some vehicles')
    print('Engineering 423 test features...', flush=True)
    X = make_features(op, spec, selected, meta, args.batch_vehicles)
    p = np.asarray(model.predict_proba(X), dtype=float)
    if p.shape != (len(selected), 5) or not np.isfinite(p).all():
        raise ValueError('Invalid predicted probability array')
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError('Class probabilities do not sum to one')
    y = selected.class_label.to_numpy(dtype=int)
    result = {'model': 'sampling_candidate_model.joblib', 'vehicles': len(y), **metrics(y, p)}
    print('\n=== SAMPLING CANDIDATE POST-HOC TEST ===', flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print('\n=== ORIGINAL MODEL TEST (REFERENCE) ===')
    print('Risk48 AP 0.054408 | AUC 0.694792 | Top5 found 15/142')
    print('Risk6  AP 0.020532 | AUC 0.672217 | Top5 found 4/60')
    print('C4 class recall 0.000000')

    pred = selected[['vehicle_id', 'time_step', 'class_label']].rename(columns={'class_label':'true_class'}).copy()
    pred['predicted_class'] = p.argmax(axis=1)
    for cls in range(5):
        pred[f'p_c{cls}'] = p[:, cls]
    for horizon, start in [(48,1),(24,2),(12,3),(6,4)]:
        pred[f'Risk{horizon}'] = p[:, start:].sum(axis=1)
    pred.to_csv(args.output_dir / 'sampling_candidate_test_predictions.csv', index=False)
    (args.output_dir / 'sampling_candidate_test_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('\nSaved:', args.output_dir)
    print('WARNING: This test set has been inspected previously; results are post-hoc, not fresh independent confirmation.')

if __name__ == '__main__':
    main()
