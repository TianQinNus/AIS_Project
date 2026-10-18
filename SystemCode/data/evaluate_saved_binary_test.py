#!/usr/bin/env python3
"""Evaluate already-saved FleetCare binary models on test CSVs. No fitting, no DB writes.
Requires fleetcare_objective_overhaul.py in the same folder (reuses exact feature builder).
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fleetcare_objective_overhaul import read, labels, make_features, risk_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('raw'))
    parser.add_argument('--model-dir', type=Path, default=Path('processed/final_model_xgboost'))
    parser.add_argument('--binary-dir', type=Path, default=Path('processed/objective_overhaul'))
    parser.add_argument('--output-dir', type=Path, default=Path('processed/objective_overhaul/test_only'))
    parser.add_argument('--batch-vehicles', type=int, default=250)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.model_dir / 'meta.json').read_text(encoding='utf-8'))
    binary_meta_path = args.binary_dir / 'binary_metadata.json'
    if binary_meta_path.exists():
        binary_meta = json.loads(binary_meta_path.read_text(encoding='utf-8'))
        if binary_meta.get('feature_names') != meta.get('feature_names'):
            raise ValueError('Binary model metadata does not match feature order in final meta.json')

    models = {}
    for horizon in (48, 6):
        path = args.binary_dir / f'binary_risk{horizon}.joblib'
        if not path.exists():
            raise FileNotFoundError(f'Missing saved model: {path}')
        models[horizon] = joblib.load(path)
        if getattr(models[horizon], 'n_features_in_', None) != meta['n_features']:
            raise ValueError(f'Risk{horizon} feature count mismatch')
        print(f'Loaded Risk{horizon}: {path}', flush=True)

    op = read(args.data_dir / 'test_operational_readouts.csv')
    op = op.sort_values(['vehicle_id', 'time_step'], kind='stable').reset_index(drop=True)
    spec = read(args.data_dir / 'test_specifications.csv')
    selected = op.groupby('vehicle_id', sort=False).tail(1)[['vehicle_id', 'time_step']].reset_index(drop=True)
    selected = selected.merge(labels(args.data_dir / 'test_labels.csv'), on='vehicle_id', validate='one_to_one')
    if selected['class_label'].isna().any():
        raise ValueError('Missing labels for test vehicles')
    print('Building test features only...', flush=True)
    x_test = make_features(op, spec, selected, meta, args.batch_vehicles)
    y_true = selected['class_label'].to_numpy(dtype=int)
    output = pd.DataFrame({'vehicle_id': selected['vehicle_id'], 'y_true': y_true})
    results = {}
    for horizon in (48, 6):
        score = models[horizon].predict_proba(x_test)[:, 1]
        if not np.isfinite(score).all():
            raise ValueError(f'Risk{horizon} produced non-finite scores')
        output[f'Risk{horizon}_binary'] = score
        results[f'Risk{horizon}'] = risk_metrics(y_true, score, horizon)
        print(f'\n=== SAVED BINARY RISK{horizon} TEST ===')
        print(json.dumps(results[f'Risk{horizon}'], indent=2), flush=True)

    output['risk6_exceeds_risk48'] = output['Risk6_binary'] > output['Risk48_binary']
    results['risk6_exceeds_risk48_count'] = int(output['risk6_exceeds_risk48'].sum())
    output.to_csv(args.output_dir / 'saved_binary_test_predictions.csv', index=False)
    (args.output_dir / 'saved_binary_test_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('\nRisk6 > Risk48 vehicles:', results['risk6_exceeds_risk48_count'])
    print('Saved:', args.output_dir)
    print('WARNING: Test has been inspected previously; these are post-hoc results, not untouched confirmation.')


if __name__ == '__main__':
    main()
