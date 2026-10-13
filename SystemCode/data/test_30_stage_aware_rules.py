"""Test 30: Stage-aware temporal decision rules on Test 29 OOF replay.

Run: python test_30_stage_aware_rules.py
No training, no DB writes. Exploratory diagnostics only: validation labels have
already been used during development. Fixed heuristic thresholds, not tuned.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

INSPECT = {'INSPECT', 'PRIORITY_INSPECT'}
ATTENTION = INSPECT | {'MONITOR', 'ENHANCED_MONITOR', 'MANUAL_REVIEW', 'COLLECT_MORE_DATA'}
LEVEL = {'CONTINUE': 0, 'MONITOR': 1, 'ENHANCED_MONITOR': 2,
         'INSPECT': 3, 'PRIORITY_INSPECT': 4}
ORDER = ['CONTINUE', 'MONITOR', 'ENHANCED_MONITOR', 'INSPECT', 'PRIORITY_INSPECT']


def check_input(df):
    needed = {'vehicle_id', 'class_label', 'Risk48_current', 'Risk12_current',
              'Risk6_current', 'Risk48_percentile', 'Risk12_percentile',
              'Risk6_percentile', 'history_count', 'v2_action', 'ml_floor_action'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'Missing Test 29 columns: {sorted(missing)}')
    if df.vehicle_id.duplicated().any():
        raise ValueError('Duplicate vehicle_id')
    for col in ('Risk48_current', 'Risk12_current', 'Risk6_current'):
        if df[col].isna().any() or not df[col].between(0, 1).all():
            raise ValueError(f'Invalid {col}')
    if not ((df.Risk48_current + 1e-7 >= df.Risk12_current) &
            (df.Risk12_current + 1e-7 >= df.Risk6_current)).all():
        raise ValueError('Horizon risk monotonicity violated')


def add_stage_features(df):
    d = df.copy()
    # Quantiles of total non-C0 risk define LOW/MEDIUM/HIGH strata.
    # These strata use scores only, not labels.
    q50, q80 = d.Risk48_current.quantile([0.5, 0.8]).to_numpy()
    d['risk_stratum'] = np.select(
        [d.Risk48_current >= q80, d.Risk48_current >= q50],
        ['HIGH', 'MEDIUM'], default='LOW')
    total = d.Risk48_current.to_numpy(float)
    near = d.Risk12_current.to_numpy(float)
    imminent = d.Risk6_current.to_numpy(float)
    # Near-term share conditional on being non-C0; not an absolute failure probability.
    d['near_share'] = np.divide(near, total, out=np.zeros_like(near), where=total > 1e-8)
    d['imminent_share'] = np.divide(imminent, total, out=np.zeros_like(imminent), where=total > 1e-8)
    # Compare each vehicle to others with similar overall Risk48.
    d['near_share_within_stratum_pct'] = (
        d.groupby('risk_stratum')['near_share'].rank(pct=True, method='average') * 100)
    d['imminent_share_within_stratum_pct'] = (
        d.groupby('risk_stratum')['imminent_share'].rank(pct=True, method='average') * 100)
    # Fixed floors prevent tiny total risk with large ratio from causing inspection.
    # Use fleet-relative absolute near-term scores as an additional gate.
    d['near_abs_pct'] = d.Risk12_current.rank(pct=True, method='average') * 100
    d['imminent_abs_pct'] = d.Risk6_current.rank(pct=True, method='average') * 100
    return d


def temporal_evidence(row):
    # Prefer already computed historical trend/persistence fields from Test 29.
    # Missing evidence never becomes positive evidence.
    rising = any(float(row.get(f'Risk{h}_change', 0) or 0) > 0 and
                 float(row.get(f'Risk{h}_slope', 0) or 0) > 0 for h in (12, 6))
    persistent = any(bool(row.get(f'Risk{h}_persistent', False)) for h in (12, 6))
    # Alternative field names from test_26 enrich (if available).
    persistent |= any(bool(row.get(f'is_risk{h}_persistent', False)) for h in (12, 6))
    return rising, persistent


def decide_stage(row):
    floor = row['ml_floor_action']
    v2 = row['v2_action']
    rising, persistent = temporal_evidence(row)
    evidence = rising or persistent
    stratum = row['risk_stratum']
    near_pct = float(row['near_share_within_stratum_pct'])
    immin_pct = float(row['imminent_share_within_stratum_pct'])
    near_abs = float(row['near_abs_pct'])
    immin_abs = float(row['imminent_abs_pct'])

    # Keep independent data-quality review flags. Do not turn review into
    # an automatic inspection or silently override a higher ML action.
    review = v2 if v2 in {'MANUAL_REVIEW', 'COLLECT_MORE_DATA'} else ''
    base = max(LEVEL.get(floor, 0), LEVEL.get(v2, 0))
    level = base
    rule = 'BASE'
    # Near-term-heavy vehicles within their own Risk48 stratum can escalate.
    # Strongest upgrade requires nontrivial absolute Risk6 AND temporal evidence.
    if (stratum in ('HIGH', 'MEDIUM') and immin_pct >= 95 and
            immin_abs >= 90 and evidence):
        level = max(level, 3 if stratum == 'HIGH' else 2)
        rule = 'ST1'
    elif (stratum in ('HIGH', 'MEDIUM') and near_pct >= 90 and
          near_abs >= 80 and evidence):
        level = max(level, 2)
        rule = 'ST2'
    elif (stratum == 'LOW' and near_pct >= 95 and near_abs >= 65 and evidence):
        level = max(level, 1)
        rule = 'ST3'
    # No temporal evidence: elevated stage share alone does not force inspection.
    action = ORDER[level]
    attention = review if review and action == 'CONTINUE' else action
    return pd.Series({'stage_rule': rule, 'stage_action': action,
                      'stage_attention_action': attention, 'stage_review_flag': review,
                      'stage_upgraded': level > base, 'stage_rising': rising,
                      'stage_persistent': persistent,
                      'stage_reason': (f'{rule}: stratum={stratum}; near_share_pct={near_pct:.1f}; '
                                       f'imminent_share_pct={immin_pct:.1f}; '
                                       f'near_abs_pct={near_abs:.1f}; imminent_abs_pct={immin_abs:.1f}; '
                                       f'rising={rising}; persistent={persistent}; '
                                       f'floor={floor}; v2={v2}; action={action}; review={review}')})


def metrics(df, method, col):
    y = df.class_label.to_numpy(int)
    a = df[col].astype(str)
    immediate = a.isin(INSPECT).to_numpy()
    attention = a.isin(ATTENTION).to_numpy()
    out = []
    for name, pos, selected in [
        ('C1-C4 safety recall', y >= 1, attention),
        ('C3-C4 safety recall', y >= 3, attention),
        ('C4 safety recall', y == 4, attention),
        ('C3-C4 immediate recall', y >= 3, immediate),
        ('C4 immediate recall', y == 4, immediate),
    ]:
        n = int(pos.sum()); found = int((pos & selected).sum())
        out.append(dict(method=method, metric=name, count=found, denominator=n,
                        value=found/n if n else np.nan))
    c0 = y == 0
    for name, mask in [('C0 immediate false alarms', immediate),
                       ('C0 attention false alarms', attention)]:
        n = int(c0.sum()); count = int((c0 & mask).sum())
        out.append(dict(method=method, metric=name, count=count, denominator=n, value=count/n))
    cont = a.eq('CONTINUE').to_numpy()
    out.append(dict(method=method, metric='CONTINUE NPV', count=int((cont & c0).sum()),
                    denominator=int(cont.sum()), value=float((cont & c0).sum()/cont.sum()) if cont.sum() else np.nan))
    out.append(dict(method=method, metric='Immediate inspection count', count=int(immediate.sum()),
                    denominator=len(df), value=float(immediate.mean())))
    out.append(dict(method=method, metric='Attention count', count=int(attention.sum()),
                    denominator=len(df), value=float(attention.mean())))
    return out


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=root/'processed/test_29_oof_decisions.csv')
    ap.add_argument('--output-dir', type=Path, default=root/'processed')
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    check_input(df)
    df = add_stage_features(df)
    decisions = df.apply(decide_stage, axis=1)
    df = pd.concat([df.reset_index(drop=True), decisions.reset_index(drop=True)], axis=1)
    baseline = 'safety_floor_attention_action' if 'safety_floor_attention_action' in df else 'safety_floor_action'
    comparisons = [('ML Floor', 'ml_floor_action'), ('Temporal V2', 'v2_action'),
                   ('V2 + Floor', baseline), ('Stage-aware', 'stage_attention_action')]
    report = pd.DataFrame([r for name, col in comparisons for r in metrics(df, name, col)])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir/'test_30_stage_aware_decisions.csv', index=False)
    report.to_csv(args.output_dir/'test_30_stage_aware_metrics.csv', index=False)
    print('\n=== STAGE-AWARE SAFETY COMPARISON (OOF, EXPLORATORY) ===')
    print(report.pivot(index='metric', columns='method', values='value').round(4).to_string())
    print('\n=== ACTION COUNTS ===')
    for name, col in comparisons:
        print(f'\n{name}:\n{df[col].value_counts().to_string()}')
    print('\n=== STAGE RULE COUNTS ===')
    print(df.stage_rule.value_counts().to_string())
    print('\n=== STAGE UPGRADES ===')
    print('Upgraded:', int(df.stage_upgraded.sum()))
    changed = df.stage_attention_action.ne(df[baseline])
    print('Changed from V2 + Floor:', int(changed.sum()))
    print(df.loc[changed, ['vehicle_id', 'class_label', 'risk_stratum', 'near_share',
                           'imminent_share', baseline, 'stage_attention_action']].head(15).to_string(index=False))
    print('\nSaved test_30_stage_aware_decisions.csv and test_30_stage_aware_metrics.csv')
    print('CAUTION: official validation reused for model development; exploratory OOF only.')


if __name__ == '__main__':
    main()
