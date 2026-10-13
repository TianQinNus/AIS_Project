"""FleetCare: OOF historical replay + ML floor / temporal rule comparison.

Run from SystemCode/data:
  python test_29_oof_temporal_safety.py

Requires compare_final_lgbm_xgb.py, fleetcare_historical_replay.py,
 test_26_temporal_rules.py and raw Scania CSVs in their usual locations.
No DB writes. Validation previously used for development: exploratory only.
"""
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from compare_final_lgbm_xgb import prepare, make_model, SEED
from fleetcare_historical_replay import replay, summarize
from test_26_temporal_rules import enrich, decide as decide_v2

INSPECT = {'INSPECT', 'PRIORITY_INSPECT'}
SAFETY = INSPECT | {'ENHANCED_MONITOR', 'MONITOR', 'MANUAL_REVIEW', 'COLLECT_MORE_DATA'}
LEVEL = {'CONTINUE': 0, 'MONITOR': 1, 'ENHANCED_MONITOR': 2,
         'INSPECT': 3, 'PRIORITY_INSPECT': 4}


def ml_floor(row):
    # Reference policy, not a calibrated probability threshold.
    if row.Risk6_percentile >= 99:
        return 'INSPECT'
    if row.Risk12_percentile >= 97:
        return 'INSPECT'
    if row.Risk24_percentile >= 90:
        return 'ENHANCED_MONITOR'
    if row.Risk48_percentile >= 80:
        return 'MONITOR'
    return 'CONTINUE'


def combine(floor, temporal):
    # Review/data requests are independent flags; they do not erase inspection.
    if temporal in ('MANUAL_REVIEW', 'COLLECT_MORE_DATA'):
        return floor, temporal
    return max((floor, temporal), key=lambda a: LEVEL[a]), ''


def assess(df, name, column):
    y = df.class_label.to_numpy(dtype=int)
    action = df[column]
    immediate = action.isin(INSPECT).to_numpy()
    safety = action.isin(SAFETY).to_numpy()
    rows = []
    for label, pos, selected in [
        ('C1-C4 safety', y >= 1, safety),
        ('C3-C4 safety', y >= 3, safety),
        ('C4 safety', y == 4, safety),
        ('C3-C4 immediate', y >= 3, immediate),
        ('C4 immediate', y == 4, immediate),
    ]:
        n = int(pos.sum()); found = int((pos & selected).sum())
        rows.append(dict(method=name, metric=label, positives=n, covered=found,
                         missed=n-found, recall=found/n if n else np.nan))
    c0 = y == 0
    for label, mask in [('C0 immediate false escalation', immediate),
                        ('C0 safety false escalation', safety)]:
        rows.append(dict(method=name, metric=label, positives=int(c0.sum()),
                         covered=int((c0 & mask).sum()), missed=np.nan,
                         recall=float((c0 & mask).sum()/c0.sum())))
    cont = action.eq('CONTINUE').to_numpy()
    rows.append(dict(method=name, metric='CONTINUE NPV', positives=int(cont.sum()),
                     covered=int((cont & c0).sum()), missed=int((cont & ~c0).sum()),
                     recall=float((cont & c0).sum()/cont.sum()) if cont.sum() else np.nan))
    rows.append(dict(method=name, metric='CONTINUE fleet fraction', positives=len(df),
                     covered=int(cont.sum()), missed=np.nan, recall=float(cont.mean())))
    rows.append(dict(method=name, metric='Immediate inspection fraction', positives=len(df),
                     covered=int(immediate.sum()), missed=np.nan, recall=float(immediate.mean())))
    return rows


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=root/'raw')
    ap.add_argument('--output-dir', type=Path, default=root/'processed')
    ap.add_argument('--val-weight', type=float, default=10)
    ap.add_argument('--k-repair', type=int, default=3)
    ap.add_argument('--k-nonrepair', type=int, default=2)
    ap.add_argument('--lags', default='5,20')
    ap.add_argument('--last-n', type=int, default=5)
    ap.add_argument('--batch-vehicles', type=int, default=150)
    args = ap.parse_args()
    if args.val_weight <= 0:
        ap.error('val-weight must be positive')
    x_tr, x_va, y_tr, y_va, sw, ids, meta = prepare(args)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Expected one validation target row per vehicle')
    op = pd.read_csv(args.data_dir/'validation_operational_readouts.csv')
    spec = pd.read_csv(args.data_dir/'validation_specifications.csv')
    labels = pd.DataFrame({'vehicle_id': ids, 'class_label': y_va})
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    histories = []
    for fold_no, (fit_idx, hold_idx) in enumerate(folds.split(x_va, y_va), 1):
        held_ids = ids[hold_idx]
        print(f'\nFold {fold_no}/5: train and replay {len(held_ids)} unseen vehicles', flush=True)
        fit_x = np.vstack((x_tr, x_va[fit_idx]))
        fit_y = np.concatenate((y_tr, y_va[fit_idx]))
        fit_w = np.concatenate((sw, np.full(len(fit_idx), args.val_weight)))
        m = make_model('xgboost')
        m.fit(fit_x, fit_y, sample_weight=fit_w)
        if list(m.classes_) != [0, 1, 2, 3, 4]:
            raise ValueError('Unexpected class order')
        held_op = op.loc[op.vehicle_id.isin(held_ids)].copy()
        held_spec = spec.loc[spec.vehicle_id.isin(held_ids)].copy()
        h = replay(held_op, held_spec, m, meta,
                   last_n=args.last_n, batch_vehicles=args.batch_vehicles)
        h['fold'] = fold_no
        histories.append(h)
        del fit_x, fit_y, fit_w, m, h, held_op, held_spec
        gc.collect()
    history = pd.concat(histories, ignore_index=True)
    if history.vehicle_id.nunique() != len(ids):
        raise ValueError('OOF history does not cover every validation vehicle')
    summary = summarize(history)
    # Enrich on ALL OOF held-out vehicles to form a single fleet-relative reference.
    # No held-out labels are used for percentile/persistence thresholds.
    enriched = enrich(history, summary)
    decisions = enriched.apply(decide_v2, axis=1, result_type='expand')
    decisions.columns = ['v2_rule_id','v2_priority','v2_action','v2_reason']
    out = pd.concat([enriched.reset_index(drop=True), decisions.reset_index(drop=True)], axis=1)
    out['ml_floor_action'] = out.apply(ml_floor, axis=1)
    combined = [combine(a,b) for a,b in zip(out.ml_floor_action, out.v2_action)]
    out['safety_floor_action'] = [a for a,_ in combined]
    out['review_flag'] = [flag for _,flag in combined]
    # For safety coverage, an independent review flag counts as attention.
    out['safety_floor_attention_action'] = np.where(
        out.review_flag.ne('') & out.safety_floor_action.eq('CONTINUE'),
        out.review_flag, out.safety_floor_action)
    out = out.merge(labels, on='vehicle_id', how='left', validate='one_to_one')
    if out.class_label.isna().any():
        raise ValueError('Missing labels')
    measures = pd.DataFrame(
        assess(out, 'ML Floor', 'ml_floor_action') +
        assess(out, 'Temporal V2', 'v2_action') +
        assess(out, 'V2 + Safety Floor', 'safety_floor_attention_action')
    )
    # Independent review flags must not suppress an immediate inspection.
    for method, col in [('ML Floor','ml_floor_action'), ('Temporal V2','v2_action'),
                        ('V2 + Safety Floor','safety_floor_attention_action')]:
        print(f'\n=== {method} ACTION COUNTS ===')
        print(out[col].value_counts().to_string())
    print('\n=== OOF SAFETY COMPARISON ===')
    print(measures.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.output_dir/'test_29_oof_history.csv', index=False)
    out.to_csv(args.output_dir/'test_29_oof_decisions.csv', index=False)
    measures.to_csv(args.output_dir/'test_29_oof_metrics.csv', index=False)
    print('\nSaved test_29_oof_history.csv, test_29_oof_decisions.csv, test_29_oof_metrics.csv')
    print('CAUTION: official validation previously reused for model development; exploratory OOF only.')
    print('Risk horizons and historical trend scores are model outputs, not verified failure mechanisms.')

if __name__ == '__main__':
    main()
