"""FleetCare Test 26: horizon-aware temporal rule engine and functional tests.

Run: python test_26_temporal_rules.py
     python test_26_temporal_rules.py --self-test

Input: processed/prediction_history.csv, processed/prediction_temporal_summary.csv
Output: processed/prediction_decisions.csv, processed/test_26_rule_counts.csv

Rules are exploratory operational policies, not validated fault predictors.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS = (48, 24, 12, 6)
# Fleet-relative percentiles, not inspection capacity constraints.
PCT = {48: 80.0, 24: 90.0, 12: 97.0, 6: 99.0}
PERSISTENCE_FRACTION = 0.60
MIN_HISTORY = 3


def enrich(history, summary):
    required_h = {'vehicle_id', 'time_step'} | {f'Risk{h}' for h in HORIZONS}
    required_s = {'vehicle_id', 'time_step', 'history_count'} | {
        f'Risk{h}_current' for h in HORIZONS
    } | {f'Risk{h}_percentile' for h in HORIZONS}
    if required_h - set(history) or required_s - set(summary):
        raise ValueError(f'Missing history columns: {sorted(required_h-set(history))}; '
                         f'summary columns: {sorted(required_s-set(summary))}')
    if history.duplicated(['vehicle_id', 'time_step']).any() or summary.vehicle_id.duplicated().any():
        raise ValueError('Duplicate vehicle/time keys')
    if not np.isfinite(history[[f'Risk{h}' for h in HORIZONS]].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite risk in history')
    if not history[[f'Risk{h}' for h in HORIZONS]].apply(lambda c: c.between(0, 1)).all().all():
        raise ValueError('Risk scores must lie in [0,1]')
    history = history.sort_values(['vehicle_id', 'time_step'])
    summary = summary.copy()
    if set(history.vehicle_id) != set(summary.vehicle_id):
        raise ValueError('History and summary vehicle IDs differ')
    latest = history.groupby('vehicle_id', sort=False).tail(1).set_index('vehicle_id')
    aligned = summary.set_index('vehicle_id')
    if not np.allclose(latest.loc[aligned.index, 'time_step'], aligned.time_step):
        raise ValueError('Summary time_step is not latest history time_step')
    for horizon in HORIZONS:
        if not np.allclose(latest.loc[aligned.index, f'Risk{horizon}'],
                           aligned[f'Risk{horizon}_current'], atol=1e-6):
            raise ValueError(f'Latest Risk{horizon} mismatch')

    # Fixed reference thresholds derived from current fleet, applied to each
    # vehicle's past scores. These are descriptive, not prospective validation.
    for horizon in HORIZONS:
        key = f'Risk{horizon}'
        cutoff = float(summary[f'{key}_current'].quantile(PCT[horizon] / 100.0))
        persist = history.groupby('vehicle_id', sort=False)[key].apply(
            lambda v: float((v.tail(5) >= cutoff).mean())
        )
        summary[f'{key}_persistence'] = summary.vehicle_id.map(persist)
        summary[f'{key}_persistent'] = (
            (summary.history_count >= MIN_HISTORY)
            & (summary[f'{key}_persistence'] >= PERSISTENCE_FRACTION)
        )
        slope = summary.get(f'{key}_slope', pd.Series(np.nan, index=summary.index))
        change = summary.get(f'{key}_change', pd.Series(np.nan, index=summary.index))
        summary[f'{key}_rising'] = (
            (summary.history_count >= MIN_HISTORY)
            & (pd.to_numeric(slope, errors='coerce') > 0)
            & (pd.to_numeric(change, errors='coerce') > 0)
        )
        # Relative to fleet change, avoiding an arbitrary absolute slope cutoff.
        rising_change = pd.to_numeric(change, errors='coerce').clip(lower=0)
        change_cutoff = float(rising_change.quantile(0.90))
        summary[f'{key}_fast'] = summary[f'{key}_rising'] & (rising_change > max(change_cutoff, 0.0))

    # High volatility is assessed relative to current fleet, with a minimum
    # 3-readout history; it is not evidence of mechanical damage.
    vol = pd.to_numeric(summary.get('Risk48_volatility', pd.Series(np.nan, index=summary.index)), errors='coerce')
    vol_cutoff = float(vol.quantile(0.95)) if vol.notna().any() else np.inf
    summary['is_volatile'] = (summary.history_count >= MIN_HISTORY) & (vol > max(vol_cutoff, 0.0))
    summary['is_uncertain'] = False  # No calibrated confidence/uncertainty signal in these CSVs.
    return summary


def decide(r):
    n = int(r['history_count'])
    def high(h, threshold=None):
        return float(r[f'Risk{h}_percentile']) >= (PCT[h] if threshold is None else threshold)
    def persistent(h):
        return bool(r[f'Risk{h}_persistent'])
    def rising(h):
        return bool(r[f'Risk{h}_rising'])
    def fast(h):
        return bool(r[f'Risk{h}_fast'])

    if n < 2:
        return ('TRD', 'REVIEW', 'COLLECT_MORE_DATA', 'Only one risk observation; temporal assessment unavailable.')
    if n < MIN_HISTORY and (high(6, 97) or high(12, 97)):
        return ('TR0', 'REVIEW', 'MANUAL_REVIEW', 'High near-term relative risk but fewer than three observations.')
    if bool(r['is_volatile']) and bool(r['is_uncertain']):
        return ('TRV', 'REVIEW', 'MANUAL_REVIEW', 'Volatile risk and independently flagged prediction uncertainty.')
    if high(6) and (persistent(6) or fast(6) or fast(12)):
        return ('TR1', 'CRITICAL', 'PRIORITY_INSPECT', 'Risk6 extreme with persistent or rapidly rising evidence.')
    if high(12) and (persistent(12) or rising(12) or fast(6)):
        return ('TR2', 'HIGH', 'INSPECT', 'Risk12 elevated with temporal support.')
    if high(6, 97) and sum(rising(h) for h in HORIZONS) >= 2:
        return ('TR3', 'HIGH', 'INSPECT', 'Risk6 elevated with multiple rising risk horizons.')
    if high(24) and (persistent(24) or rising(24)):
        return ('TR4', 'MEDIUM', 'ENHANCED_MONITOR', 'Risk24 elevated with persistence or upward trend.')
    if high(48) and (persistent(48) or rising(48)):
        return ('TR5', 'LOW', 'MONITOR', 'Risk48 elevated with persistence or upward trend.')
    return ('TR6', 'NORMAL', 'CONTINUE', 'No temporal escalation rule met.')


def run_engine(history, summary):
    summary = enrich(history, summary)
    rules = summary.apply(decide, axis=1, result_type='expand')
    rules.columns = ['rule_id', 'priority', 'action', 'reason']
    out = pd.concat([summary.reset_index(drop=True), rules.reset_index(drop=True)], axis=1)
    out['evidence'] = out.apply(
        lambda r: '; '.join(
            f'Risk{h}:pct={r[f"Risk{h}_percentile"]:.1f},'
            f'persistence={r[f"Risk{h}_persistence"]:.2f},'
            f'rising={bool(r[f"Risk{h}_rising"])}'
            for h in HORIZONS
        ), axis=1)
    out['rule_trace'] = out.apply(
        lambda r: f'{r.rule_id} | vehicle={r.vehicle_id} | n={r.history_count} | '
                  f'volatile={bool(r.is_volatile)} | {r.evidence} | action={r.action}', axis=1)
    out['ml_ranking'] = out['Risk48_current'].rank(ascending=False, method='first').astype(int)
    return out.sort_values('ml_ranking').reset_index(drop=True)


def self_test():
    # Test precedence, each action, and no arbitrary capacity restriction.
    def base(**kw):
        r = {'history_count': 5, 'is_volatile': False, 'is_uncertain': False}
        for h in HORIZONS:
            r.update({f'Risk{h}_percentile': 10.0, f'Risk{h}_persistent': False,
                      f'Risk{h}_rising': False, f'Risk{h}_fast': False})
        r.update(kw)
        return r
    cases = [
        ('TRD', base(history_count=1)),
        ('TR0', base(history_count=2, Risk6_percentile=99.9)),
        ('TRV', base(is_volatile=True, is_uncertain=True)),
        ('TR1', base(Risk6_percentile=99.9, Risk6_persistent=True)),
        ('TR2', base(Risk12_percentile=98, Risk12_rising=True)),
        ('TR3', base(Risk6_percentile=98, Risk48_rising=True, Risk24_rising=True)),
        ('TR4', base(Risk24_percentile=95, Risk24_persistent=True)),
        ('TR5', base(Risk48_percentile=85, Risk48_rising=True)),
        ('TR6', base()),
        ('TR1', base(Risk6_percentile=99.9, Risk6_persistent=True,
                     Risk24_percentile=95, Risk24_rising=True)),
    ]
    for expected, row in cases:
        actual = decide(row)[0]
        assert actual == expected, f'Expected {expected}, got {actual}'
    print(f'PASS: {len(cases)} rule/precedence unit tests')
    print('NOTE: TRV requires a separate uncertainty flag; historical replay CSV does not contain one.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--history', type=Path, default=Path(__file__).resolve().parent / 'processed/prediction_history.csv')
    ap.add_argument('--summary', type=Path, default=Path(__file__).resolve().parent / 'processed/prediction_temporal_summary.csv')
    ap.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'processed')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    self_test()
    if args.self_test:
        return
    history = pd.read_csv(args.history)
    summary = pd.read_csv(args.summary)
    out = run_engine(history, summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_dir / 'prediction_decisions.csv', index=False)
    counts = out.groupby(['rule_id', 'priority', 'action']).size().rename('vehicles').reset_index()
    counts.to_csv(args.output_dir / 'test_26_rule_counts.csv', index=False)
    print('\n=== RULE COUNTS ===')
    print(counts.to_string(index=False))
    print('\n=== SAMPLE DECISIONS ===')
    print(out[['vehicle_id', 'Risk48_current', 'Risk6_percentile', 'history_count',
               'rule_id', 'priority', 'action']].head(12).to_string(index=False))
    print('\nSaved:', args.output_dir / 'prediction_decisions.csv')
    print('Saved:', args.output_dir / 'test_26_rule_counts.csv')
    print('Descriptive rules only: not an independent performance evaluation.')


if __name__ == '__main__':
    main()
