"""Test 27: Compare temporal decisions with the SAME final-model ML risk ranking.
This is an in-sample diagnostic, NOT an independent validation of improvement.
Run: python test_27_rule_effectiveness.py
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

ROOT = Path(__file__).resolve().parent


def metrics(truth, selected):
    truth = np.asarray(truth, dtype=bool)
    selected = np.asarray(selected, dtype=bool)
    tp = int((truth & selected).sum())
    fp = int((~truth & selected).sum())
    fn = int((truth & ~selected).sum())
    tn = int((~truth & ~selected).sum())
    return dict(alerts=int(selected.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                accuracy=accuracy_score(truth, selected),
                precision=precision_score(truth, selected, zero_division=0),
                recall=recall_score(truth, selected, zero_division=0),
                binary_f1=f1_score(truth, selected, zero_division=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--decisions', type=Path, default=ROOT/'processed/prediction_decisions.csv')
    p.add_argument('--labels', type=Path, default=ROOT/'raw/validation_labels.csv')
    p.add_argument('--oof', type=Path, default=ROOT/'processed/final_model_oof_predictions.csv')
    p.add_argument('--output-dir', type=Path, default=ROOT/'processed')
    p.add_argument('--positive', choices=['nonzero', 'c4'], default='nonzero',
                   help='nonzero means C1-C4; c4 means only C4')
    args = p.parse_args()
    d = pd.read_csv(args.decisions)
    lab = pd.read_csv(args.labels)
    required = {'vehicle_id','action','rule_id','Risk48_current'}
    if required-set(d):
        raise ValueError(f'Decisions missing {required-set(d)}')
    if not {'vehicle_id','class_label'}.issubset(lab):
        raise ValueError('Labels must have vehicle_id, class_label')
    if d.vehicle_id.duplicated().any() or lab.vehicle_id.duplicated().any():
        raise ValueError('Expected unique vehicle_id in both files')
    data = d.merge(lab[['vehicle_id','class_label']], on='vehicle_id', how='left', validate='one_to_one')
    if data.class_label.isna().any():
        raise ValueError('Missing labels for some decisions')
    if len(data) != len(lab):
        raise ValueError('Decisions and validation labels cover different numbers of vehicles')
    truth = (data.class_label.to_numpy() != 0) if args.positive=='nonzero' else (data.class_label.to_numpy()==4)
    selected = data.action.isin(['INSPECT','PRIORITY_INSPECT']).to_numpy()
    n = int(selected.sum())
    score = pd.to_numeric(data.Risk48_current, errors='raise').to_numpy()
    if not np.isfinite(score).all():
        raise ValueError('Non-finite risk score')
    # Stable tie-breaking by vehicle ID; same final-model predictions for both policies.
    order = np.lexsort((data.vehicle_id.to_numpy(), -score))
    ml_selected = np.zeros(len(data), dtype=bool)
    ml_selected[order[:n]] = True
    results = pd.DataFrame([
        {'method':'ML Only (same final model, Top N)', **metrics(truth, ml_selected)},
        {'method':'Temporal Rule Engine', **metrics(truth, selected)},
    ])
    print(f'\n=== SAME-MODEL COMPARISON: {args.positive.upper()} ===')
    print(f'Vehicles: {len(data)}, actual positives: {truth.sum()}, inspection count: {n}')
    print(results.round(5).to_string(index=False))
    added = selected & ~ml_selected
    removed = ml_selected & ~selected
    print('\n=== DECISION CHANGES ===')
    print(f'Rule-only inspected: {added.sum()} (positives: {int(truth[added].sum())})')
    print(f'ML-only inspected:   {removed.sum()} (positives: {int(truth[removed].sum())})')
    print(f'Net additional positives: {int(truth[added].sum()-truth[removed].sum())}')
    data['actual_positive'] = truth.astype(int)
    data['ml_inspect'] = ml_selected
    data['rule_inspect'] = selected
    data['decision_change'] = np.select([added,removed],['RULE_ONLY','ML_ONLY'],default='BOTH_OR_NEITHER')
    by_rule = data.groupby(['rule_id','action'], dropna=False).agg(
        vehicles=('vehicle_id','size'), positives=('actual_positive','sum')).reset_index()
    by_rule['positive_rate'] = by_rule['positives']/by_rule['vehicles']
    print('\n=== RISK RATE BY RULE ===')
    print(by_rule.round(4).to_string(index=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir/'test_27_comparison.csv',index=False)
    by_rule.to_csv(args.output_dir/'test_27_by_rule.csv',index=False)
    data.to_csv(args.output_dir/'test_27_vehicle_decisions.csv',index=False)
    if args.oof.exists():
        oof = pd.read_csv(args.oof)
        if {'vehicle_id','xgboost_risk'}.issubset(oof):
            ref = data[['vehicle_id','actual_positive','rule_inspect']].merge(
                oof[['vehicle_id','xgboost_risk']],on='vehicle_id',validate='one_to_one')
            order_oof = np.lexsort((ref.vehicle_id.to_numpy(),-ref.xgboost_risk.to_numpy()))
            mask = np.zeros(len(ref),dtype=bool); mask[order_oof[:n]]=True
            print('\n=== OOF REFERENCE (NOT A FAIR MODEL-VS-RULE COMPARISON) ===')
            print(pd.Series(metrics(ref.actual_positive.to_numpy(),mask)).to_string())
            print('Rule decisions were produced using a final model trained on validation labels; OOF predictions were not.')
    print('\nSaved comparison, by-rule rates, and vehicle decisions to:',args.output_dir)
    print('WARNING: In-sample diagnostic. Cannot claim independent predictive improvement.')

if __name__=='__main__':
    main()
