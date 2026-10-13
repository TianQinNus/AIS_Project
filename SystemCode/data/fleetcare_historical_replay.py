"""FleetCare historical replay using the final exposure-normalised model.

Uses the SAME build_features implementation as final training, and only scores
selected historical readouts. Does not modify the database or train a model.

Example:
 python fleetcare_historical_replay.py --model-dir processed/final_xgboost \
   --op-csv raw/validation_operational_readouts.csv \
   --spec-csv raw/validation_specifications.csv --last-n 5

Requires test_20_normalized.py and final_model_xgboost.py in the same directory.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from test21  import build_features, detect_groups
from final import spec_for

HORIZONS = {48: 1, 24: 2, 12: 3, 6: 4}


def load_model(model_dir):
    meta = json.loads((model_dir / 'meta.json').read_text(encoding='utf-8'))
    model = joblib.load(model_dir / 'model.joblib')
    if model.__class__.__name__ != 'XGBClassifier':
        raise TypeError(f'Expected XGBClassifier; got {type(model)}')
    if list(model.classes_) != [0, 1, 2, 3, 4]:
        raise ValueError('Model classes must be [0,1,2,3,4]')
    for key in ('raw_cols', 'lags', 'spec_columns', 'n_features'):
        if key not in meta:
            raise KeyError(f'Missing metadata field: {key}')
    return model, meta


def replay(op, spec, model, meta, last_n=5, batch_vehicles=250):
    if last_n < 1 or batch_vehicles < 1:
        raise ValueError('last_n and batch_vehicles must be positive')
    if not {'vehicle_id', 'time_step'}.issubset(op.columns):
        raise ValueError('Operational CSV needs vehicle_id and time_step')
    if 'vehicle_id' not in spec or spec['vehicle_id'].duplicated().any():
        raise ValueError('Specification CSV needs unique vehicle_id')
    raw_cols = meta['raw_cols']
    missing = sorted(set(raw_cols) - set(op.columns))
    if missing:
        raise ValueError(f'Missing operational columns: {missing[:8]}')
    op = op.sort_values(['vehicle_id', 'time_step'], kind='stable').reset_index(drop=True)
    if op.duplicated(['vehicle_id', 'time_step']).any():
        raise ValueError('Duplicate vehicle_id + time_step')
    hist, singles = detect_groups(raw_cols)
    vehicle_ids = op['vehicle_id'].drop_duplicates().tolist()
    results = []
    for offset in range(0, len(vehicle_ids), batch_vehicles):
        ids = vehicle_ids[offset:offset + batch_vehicles]
        # Keep ALL prior readouts for these vehicles to calculate lags 5 and 20.
        subset = op[op['vehicle_id'].isin(ids)].copy().reset_index(drop=True)
        selected = subset.groupby('vehicle_id', sort=False).tail(last_n)
        sel = pd.DataFrame({
            'vehicle_id': selected['vehicle_id'].to_numpy(),
            'row_id': selected.index.to_numpy(dtype=np.int64),
        })
        x_f, names, _, _ = build_features(
            subset, sel, raw_cols, hist, singles, meta['lags']
        )
        x_s = spec_for(spec, sel['vehicle_id'].to_numpy(), meta['spec_columns'])
        x = np.hstack([x_f, x_s]).astype(np.float32)
        if x.shape[1] != meta['n_features']:
            raise ValueError(f'Feature count mismatch: {x.shape[1]} vs {meta["n_features"]}')
        expected_names = list(names) + [f'spec:{n}' for n in meta['spec_columns']]
        if 'feature_names' in meta and expected_names != list(meta['feature_names']):
            raise ValueError('Feature order mismatch with training metadata')
        p = np.asarray(model.predict_proba(x), dtype=np.float64)
        if p.shape != (len(sel), 5) or not np.isfinite(p).all():
            raise ValueError('Invalid prediction probability shape or values')
        if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
            raise ValueError('Predicted probabilities do not sum to 1')
        rows = pd.DataFrame({
            'vehicle_id': sel['vehicle_id'].to_numpy(),
            'time_step': subset.loc[sel['row_id'], 'time_step'].to_numpy(),
        })
        for cls in range(5):
            rows[f'p_c{cls}'] = p[:, cls]
        for horizon, start in HORIZONS.items():
            rows[f'Risk{horizon}'] = p[:, start:].sum(axis=1)
        rows['risk'] = rows['Risk48']
        rows['predicted_class'] = p.argmax(axis=1)
        results.append(rows)
        print(f'Processed {min(offset + batch_vehicles, len(vehicle_ids))}/{len(vehicle_ids)} vehicles', flush=True)
    return pd.concat(results, ignore_index=True).sort_values(
        ['vehicle_id', 'time_step'], kind='stable'
    ).reset_index(drop=True)


def summarize(history):
    rows = []
    for vid, group in history.groupby('vehicle_id', sort=False):
        g = group.sort_values('time_step')
        recent = g.tail(5)
        times = recent['time_step'].to_numpy(dtype=float)
        record = {
            'vehicle_id': vid,
            'time_step': g['time_step'].iloc[-1],
            'history_count': len(g),
            'history_quality': 'STRONG' if len(g) >= 5 else 'MODERATE' if len(g) >= 3 else 'LIMITED',
        }
        for horizon in HORIZONS:
            name = f'Risk{horizon}'
            values = recent[name].to_numpy(dtype=float)
            record[f'{name}_current'] = values[-1]
            record[f'{name}_change'] = values[-1] - values[0] if len(values) >= 2 else np.nan
            record[f'{name}_slope'] = (
                float(np.polyfit(times - times[0], values, 1)[0])
                if len(values) >= 3 and np.ptp(times) > 0 else np.nan
            )
            record[f'{name}_volatility'] = (
                float(np.std(np.diff(values))) if len(values) >= 3 else np.nan
            )
        rows.append(record)
    out = pd.DataFrame(rows)
    for horizon in HORIZONS:
        name = f'Risk{horizon}'
        out[f'{name}_percentile'] = out[f'{name}_current'].rank(pct=True, method='average') * 100
    return out


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Replay last N vehicle predictions without DB writes')
    parser.add_argument('--model-dir', type=Path, default=root / 'processed' / 'final_xgboost')
    parser.add_argument('--op-csv', type=Path, default=root / 'raw' / 'validation_operational_readouts.csv')
    parser.add_argument('--spec-csv', type=Path, default=root / 'raw' / 'validation_specifications.csv')
    parser.add_argument('--output-dir', type=Path, default=root / 'processed')
    parser.add_argument('--last-n', type=int, default=5)
    parser.add_argument('--batch-vehicles', type=int, default=250)
    args = parser.parse_args()
    model, meta = load_model(args.model_dir)
    op = pd.read_csv(args.op_csv)
    spec = pd.read_csv(args.spec_csv)
    print(f'Loaded {len(op):,} readouts for {op.vehicle_id.nunique():,} vehicles', flush=True)
    history = replay(op, spec, model, meta, args.last_n, args.batch_vehicles)
    summary = summarize(history)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.output_dir / 'prediction_history.csv', index=False)
    summary.to_csv(args.output_dir / 'prediction_temporal_summary.csv', index=False)
    print('\nSample replay:')
    print(history[['vehicle_id', 'time_step', 'Risk48', 'Risk24', 'Risk12', 'Risk6']].head(12).to_string(index=False))
    print('\nSaved:', args.output_dir / 'prediction_history.csv')
    print('Saved:', args.output_dir / 'prediction_temporal_summary.csv')
    print('Note: historical replay is descriptive, not independent backtesting.')


if __name__ == '__main__':
    main()
