"""Test 24: Evaluate FleetCare risk-ranking rules using out-of-fold predictions.

No historical per-time-step model scores are available, so this evaluates ONLY
current-risk percentile rules (TR1/TR2/TR4/TR5/TR6). It does not claim to test
persistence, deterioration, volatility, or SHAP/RAG effects.

Run:
 python test_24_rule_engine_evaluation.py --algorithm xgboost
 python test_24_rule_engine_evaluation.py --algorithm lightgbm
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score, roc_auc_score, confusion_matrix


def evaluate(y, alert, label):
    return {
        'method': label,
        'vehicles': len(y),
        'actual_at_risk': int(y.sum()),
        'alerts': int(alert.sum()),
        'alert_rate': float(alert.mean()),
        'tp': int(((y == 1) & (alert == 1)).sum()),
        'fp': int(((y == 0) & (alert == 1)).sum()),
        'fn': int(((y == 1) & (alert == 0)).sum()),
        'tn': int(((y == 0) & (alert == 0)).sum()),
        'accuracy': float(accuracy_score(y, alert)),
        'precision': float(precision_score(y, alert, zero_division=0)),
        'recall': float(recall_score(y, alert, zero_division=0)),
        'binary_f1': float(f1_score(y, alert, zero_division=0)),
    }


def assign_current_risk_rules(df):
    """Assign policy actions by global ML rank; no history-dependent rules."""
    n = len(df)
    df = df.sort_values(['risk', 'vehicle_id'], ascending=[False, True], kind='stable').copy()
    df['ranking'] = np.arange(1, n + 1)
    # Exact, disjoint bands using ceil of inspection capacities.
    k25 = int(np.ceil(n * 0.025))
    k5 = int(np.ceil(n * 0.05))
    k10 = int(np.ceil(n * 0.10))
    k20 = int(np.ceil(n * 0.20))
    r = df['ranking'].to_numpy()
    df['rule_id'] = np.select([r <= k25, r <= k5, r <= k10, r <= k20],
                              ['TR1', 'TR2', 'TR4', 'TR5'], default='TR6')
    df['priority'] = np.select([r <= k25, r <= k5, r <= k10, r <= k20],
                               ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'], default='NORMAL')
    df['action'] = np.select([r <= k25, r <= k5, r <= k10, r <= k20],
                             ['PRIORITY_INSPECT', 'INSPECT', 'ENHANCED_MONITOR', 'MONITOR'],
                             default='CONTINUE')
    df['risk_percentile'] = 100 * (n - df['ranking'] + 1) / n
    df['reason'] = 'Current ML risk rank; temporal evidence unavailable'
    df['rule_trace'] = 'current_risk_only | ' + df['rule_id'] + ' | rank=' + df['ranking'].astype(str)
    return df


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument('--predictions', type=Path, default=root/'processed'/'final_model_oof_predictions.csv')
    parser.add_argument('--output-dir', type=Path, default=root/'processed')
    parser.add_argument('--algorithm', choices=['xgboost', 'lightgbm'], default='xgboost')
    args = parser.parse_args()
    df = pd.read_csv(args.predictions)
    col = f'{args.algorithm}_risk'
    required = {'vehicle_id', 'actual_class', col}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(df.columns))}')
    if df['vehicle_id'].isna().any() or df['vehicle_id'].duplicated().any():
        raise ValueError('OOF input must have one unique row per vehicle')
    df['risk'] = pd.to_numeric(df[col], errors='raise')
    if not np.isfinite(df['risk']).all() or not df['risk'].between(0, 1).all():
        raise ValueError('Invalid risk scores')
    df['actual_risk'] = (df['actual_class'].astype(int) != 0).astype(int)
    df = assign_current_risk_rules(df)
    y = df['actual_risk'].to_numpy()
    results = []
    for capacity in (0.025, 0.05, 0.10, 0.20):
        k = int(np.ceil(len(df) * capacity))
        ml_alert = (df['ranking'].to_numpy() <= k).astype(int)
        # For the first two capacities, the rule actions reproduce the same
        # ML rank selection exactly. The other capacities are monitoring policies.
        allowed = ({'PRIORITY_INSPECT'} if capacity == 0.025 else
                   {'PRIORITY_INSPECT', 'INSPECT'} if capacity == 0.05 else
                   {'PRIORITY_INSPECT', 'INSPECT', 'ENHANCED_MONITOR'} if capacity == 0.10 else
                   {'PRIORITY_INSPECT', 'INSPECT', 'ENHANCED_MONITOR', 'MONITOR'})
        rule_alert = df['action'].isin(allowed).astype(int).to_numpy()
        for label, pred in [('ML TopK', ml_alert), ('Rule Engine', rule_alert)]:
            row = evaluate(y, pred, label)
            row['capacity'] = capacity
            results.append(row)
        assert np.array_equal(ml_alert, rule_alert), 'Unexpected rule/ranking discrepancy'
    table = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / f'test_24_{args.algorithm}_evaluation.csv', index=False)
    df.to_csv(args.output_dir / f'test_24_{args.algorithm}_decisions.csv', index=False)
    print('\n=== RANKING PERFORMANCE (unchanged by rule engine) ===')
    print('ROC-AUC:', round(roc_auc_score(y, df['risk']), 6))
    print('AP:', round(average_precision_score(y, df['risk']), 6))
    print('\n=== BINARY ACTION EVALUATION ===')
    print(table[['capacity','method','alerts','tp','fp','fn','accuracy','precision','recall','binary_f1']].round(5).to_string(index=False))
    print('\nRule action counts:')
    print(df['action'].value_counts().to_string())
    print('\nCONCLUSION: Rule Engine maps the SAME ML ranking to action bands. '
          'With no historical risk series, it cannot improve ROC-AUC, AP or TopK recall. '
          'The matching rows above are expected, not a bug.')
    print('These are exploratory OOF predictions from a previously reused validation set.')


if __name__ == '__main__':
    main()
