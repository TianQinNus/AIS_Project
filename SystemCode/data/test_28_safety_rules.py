"""FleetCare Test 28: safety-first horizon-aware rules v3 vs v2.

Uses existing historical replay CSVs, no retraining or DB writes.
IMPORTANT: final model trained on validation labels; evaluation is IN-SAMPLE ONLY.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from test_26_temporal_rules import enrich, decide as decide_v2, HORIZONS

INSPECTION = {'PRIORITY_INSPECT', 'INSPECT'}
SAFETY = INSPECTION | {'ENHANCED_MONITOR', 'MONITOR', 'MANUAL_REVIEW', 'COLLECT_MORE_DATA'}


def decide_v3(r):
    """Ordered, conservative safety rules; no fixed inspection capacity.

    Fleet percentile cutoffs are exploratory policies, NOT calibrated probabilities.
    A high near-term risk cannot be cleared just because its trend is falling.
    """
    n = int(r['history_count'])
    def high(h, pct):
        return float(r[f'Risk{h}_percentile']) >= pct
    def persistent(h):
        return bool(r[f'Risk{h}_persistent'])
    def rising(h):
        return bool(r[f'Risk{h}_rising'])
    def fast(h):
        return bool(r[f'Risk{h}_fast'])

    if n < 2:
        return ('TRD', 'REVIEW', 'COLLECT_MORE_DATA', 'Insufficient historical observations; do not clear.')
    if n < 3 and (high(6, 90) or high(12, 90) or high(48, 90)):
        return ('TR0', 'REVIEW', 'MANUAL_REVIEW', 'Elevated risk with limited history.')
    if bool(r['is_volatile']) and bool(r['is_uncertain']):
        return ('TRV', 'REVIEW', 'MANUAL_REVIEW', 'Volatile and uncertain prediction.')

    # Near-term safety overrides: trend confirmation is not required.
    if high(6, 99) and (persistent(6) or rising(6) or fast(6)):
        return ('TR1', 'CRITICAL', 'PRIORITY_INSPECT', 'Extreme Risk6 supported by temporal evidence.')
    if high(6, 97) or high(12, 97):
        return ('TR2', 'HIGH', 'INSPECT', 'High near-term risk; safety override regardless of trend.')
    if high(6, 90) or high(12, 90):
        return ('TR3', 'REVIEW', 'MANUAL_REVIEW', 'Near-term risk elevated; manual review required.')
    if high(24, 90) or (high(48, 85) and (persistent(48) or rising(48))):
        return ('TR4', 'MEDIUM', 'ENHANCED_MONITOR', 'Elevated medium-term risk or supported early-risk signal.')
    if high(48, 80) or any(rising(h) and high(h, 65) for h in HORIZONS) or bool(r['is_volatile']):
        return ('TR5', 'LOW', 'MONITOR', 'Early risk, rising risk, or volatile history.')
    if n < 3:
        return ('TR0', 'REVIEW', 'MANUAL_REVIEW', 'Limited history: insufficient evidence to clear.')
    return ('TR6', 'NORMAL', 'CONTINUE', 'All horizon risk percentiles low with sufficient history.')


def apply_rules(enriched, decide_fn, prefix):
    decisions = enriched.apply(decide_fn, axis=1, result_type='expand')
    decisions.columns = [f'{prefix}_{c}' for c in ('rule_id','priority','action','reason')]
    return decisions


def evaluate(df, prefix):
    action = df[f'{prefix}_action']
    y = df['class_label'].to_numpy(dtype=int)
    inspected = action.isin(INSPECTION).to_numpy()
    covered = action.isin(SAFETY).to_numpy()
    cont = (action == 'CONTINUE').to_numpy()
    rows = []
    for name, positive, selected in [
        ('C1-C4 safety coverage', y >= 1, covered),
        ('C3-C4 safety coverage', y >= 3, covered),
        ('C4 safety coverage', y == 4, covered),
        ('C3-C4 immediate inspection', y >= 3, inspected),
        ('C4 immediate inspection', y == 4, inspected),
    ]:
        total = int(positive.sum())
        tp = int((positive & selected).sum())
        rows.append({'method': prefix, 'metric': name, 'positives': total,
                     'covered': tp, 'missed': total-tp,
                     'recall': tp/total if total else np.nan})
    c0 = y == 0
    rows.append({'method':prefix,'metric':'C0 unnecessary immediate inspection',
                 'positives':int(c0.sum()),'covered':int((c0 & inspected).sum()),
                 'missed':np.nan,'recall':float((c0 & inspected).sum()/c0.sum())})
    rows.append({'method':prefix,'metric':'CONTINUE negative predictive value',
                 'positives':int(cont.sum()),'covered':int((cont & c0).sum()),
                 'missed':int((cont & ~c0).sum()),
                 'recall':float((cont & c0).sum()/cont.sum()) if cont.sum() else np.nan})
    rows.append({'method':prefix,'metric':'CONTINUE fleet fraction',
                 'positives':len(df),'covered':int(cont.sum()),'missed':np.nan,
                 'recall':float(cont.mean())})
    return rows


def self_test():
    def base(**kw):
        d = {'history_count':5, 'is_volatile':False, 'is_uncertain':False}
        for h in HORIZONS:
            d.update({f'Risk{h}_percentile':10.0, f'Risk{h}_persistent':False,
                      f'Risk{h}_rising':False, f'Risk{h}_fast':False})
        d.update(kw)
        return d
    cases = [
        ('TRD',base(history_count=1)),
        ('TR0',base(history_count=2, Risk12_percentile=98)),
        ('TRV',base(is_volatile=True,is_uncertain=True)),
        ('TR1',base(Risk6_percentile=99.5,Risk6_persistent=True)),
        ('TR2',base(Risk6_percentile=98)),
        ('TR2',base(Risk12_percentile=98,Risk12_rising=False)),
        ('TR3',base(Risk12_percentile=93)),
        ('TR4',base(Risk24_percentile=92)),
        ('TR5',base(Risk48_percentile=82)),
        ('TR0',base(history_count=2)),
        ('TR6',base()),
        ('TR2',base(Risk6_percentile=98,Risk24_percentile=99)),
    ]
    for expected, sample in cases:
        actual = decide_v3(sample)[0]
        assert actual == expected, (expected,actual)
    print(f'PASS: {len(cases)} safety/precedence unit tests')


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument('--history', type=Path, default=root/'processed/prediction_history.csv')
    parser.add_argument('--summary', type=Path, default=root/'processed/prediction_temporal_summary.csv')
    parser.add_argument('--labels', type=Path, default=root/'raw/validation_labels.csv')
    parser.add_argument('--output-dir', type=Path, default=root/'processed')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    h = pd.read_csv(args.history)
    s = pd.read_csv(args.summary)
    labels = pd.read_csv(args.labels)
    if labels.duplicated('vehicle_id').any() or not {'vehicle_id','class_label'}.issubset(labels):
        raise ValueError('Expected unique vehicle_id and class_label in labels')
    enriched = enrich(h, s)
    df = pd.concat([enriched.reset_index(drop=True),
                    apply_rules(enriched,'dummy' if False else decide_v2,'v2').reset_index(drop=True),
                    apply_rules(enriched,decide_v3,'v3').reset_index(drop=True)], axis=1)
    df = df.merge(labels[['vehicle_id','class_label']],on='vehicle_id',how='left',validate='one_to_one')
    if df.class_label.isna().any():
        raise ValueError('Missing labels for some vehicles')
    df['class_label'] = df.class_label.astype(int)
    results = pd.DataFrame(evaluate(df,'v2') + evaluate(df,'v3'))
    print('\n=== V2 VS V3 SAFETY METRICS (IN-SAMPLE) ===')
    print(results.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    counts = pd.concat([df.groupby(['v2_rule_id','v2_action']).size().rename('vehicles').reset_index().rename(columns={'v2_rule_id':'rule_id','v2_action':'action'}).assign(version='v2'),
                        df.groupby(['v3_rule_id','v3_action']).size().rename('vehicles').reset_index().rename(columns={'v3_rule_id':'rule_id','v3_action':'action'}).assign(version='v3')],ignore_index=True)
    print('\n=== ACTION COUNTS ===')
    print(counts.to_string(index=False))
    print('\n=== CHANGED ACTIONS ===')
    changed = df[df.v2_action != df.v3_action]
    print(f'{len(changed)} of {len(df)} vehicles changed action')
    print(changed[['vehicle_id','class_label','Risk48_current','Risk6_percentile','v2_action','v3_action']].head(15).to_string(index=False))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.output_dir/'test_28_v2_v3_decisions.csv',index=False)
    results.to_csv(args.output_dir/'test_28_safety_metrics.csv',index=False)
    counts.to_csv(args.output_dir/'test_28_rule_counts.csv',index=False)
    print('\nWARNING: The final model was trained using validation labels. These are in-sample diagnostic metrics, NOT generalization performance.')
    print('Saved results to:',args.output_dir)

if __name__ == '__main__':
    main()
