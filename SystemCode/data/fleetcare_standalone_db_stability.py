"""FleetCare standalone inference (no imports from other project scripts).

Requires the EXISTING final_model_xgboost/model.joblib and meta.json, plus Python packages:
    pip install numpy pandas xgboost joblib sqlalchemy psycopg2-binary

Examples:
    python fleetcare_standalone_db_stability.py --model-dir processed/final_model_xgboost \
        --op-csv raw/validation_operational_readouts.csv \
        --spec-csv raw/validation_specifications.csv
    python fleetcare_standalone_db_stability.py --model-dir processed/final_model_xgboost
    python fleetcare_standalone_db_stability.py --model-dir processed/final_model_xgboost --write-db

Default is READ ONLY. SHAP uses XGBoost native pred_contribs and reports
negative C0 raw-margin attribution, NOT calibrated Risk48 probability SHAP.
Stage-aware rules are exploratory heuristics, NOT verified maintenance safety.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Dict iteration yields the four horizons, and .items() yields class starting indices.
HORIZONS = {48: 1, 24: 2, 12: 3, 6: 4}
PCT = {48: 80.0, 24: 90.0, 12: 97.0, 6: 99.0}
PERSISTENCE_FRACTION = 0.60
MIN_HISTORY = 3
LEVEL = {'CONTINUE': 0, 'MONITOR': 1, 'ENHANCED_MONITOR': 2,
         'INSPECT': 3, 'PRIORITY_INSPECT': 4}
ORDER = ['CONTINUE', 'MONITOR', 'ENHANCED_MONITOR', 'INSPECT', 'PRIORITY_INSPECT']
PRIORITY_RANK = {'NORMAL':0, 'LOW':1, 'MEDIUM':2, 'REVIEW':3, 'HIGH':4, 'CRITICAL':5}
PRIORITY_FOR_ACTION = {'CONTINUE':'NORMAL', 'MONITOR':'LOW', 'ENHANCED_MONITOR':'MEDIUM',
                       'INSPECT':'HIGH', 'PRIORITY_INSPECT':'CRITICAL',
                       'MANUAL_REVIEW':'REVIEW', 'INSUFFICIENT_DATA':'REVIEW'}


def detect_groups(cols, min_size=2, max_size=40):

    "Group histogram columns by the prefix before the last underscore; treat all other columns as individual counters."

    buckets = {}

    for i, c in enumerate(cols):

        c = str(c)

        key = c.rsplit('_', 1)[0] if '_' in c else c

        buckets.setdefault(key, []).append(i)

    hist = [(k, np.array(v)) for k, v in buckets.items() if min_size <= len(v) <= max_size]

    used = {i for _, v in hist for i in v}

    singles = np.array([i for i in range(len(cols)) if i not in used], dtype=int)

    return hist, singles


def vehicle_start(veh):

    change = np.r_[True, veh[1:] != veh[:-1]]

    starts = np.flatnonzero(change)

    return starts[np.cumsum(change) - 1]


def sdiv(a, b):

    a = np.asarray(a, dtype=np.float64)

    b = np.asarray(b, dtype=np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):

        out = a / b

    return np.where(np.isfinite(out) & (b > 0), out, np.nan)


def build_features(op, sel, raw_cols, hist, singles, lags):

    X = op[raw_cols].to_numpy(dtype=np.float32)

    veh = op['vehicle_id'].to_numpy()

    ts = op['time_step'].to_numpy(dtype=np.float64)

    vst = vehicle_start(veh)

    pos = sel['row_id'].to_numpy()

    cur = X[pos].astype(np.float64)

    t_cur = ts[pos]

    age = t_cur + 1.0

    nan_rate = np.isnan(cur).mean(axis=0)



    cols, names, blocks = [], [], []



    def add(arr, nms, block):

        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 1:

            arr = arr[:, None]

        cols.append(arr)

        names.extend(nms)

        blocks.extend([block] * len(nms))



    life = sdiv(cur[:, singles], age[:, None])

    add(life, [f'rate:{raw_cols[i]}' for i in singles], 'rate')

    tot_cur = {}

    for key, idx in hist:

        tot = cur[:, idx].sum(axis=1)

        tot_cur[key] = tot

        add(sdiv(cur[:, idx], tot[:, None]), [f'prop:{raw_cols[i]}' for i in idx], 'prop')

        add(sdiv(tot, age), [f'rate:{key}_total'], 'rate')



    for lag in lags:

        pl = np.maximum(pos - lag, vst[pos])

        dt = t_cur - ts[pl]

        d = cur - X[pl].astype(np.float64)

        d = np.where(d < 0, np.nan, d)

        rec = sdiv(d[:, singles], dt[:, None])

        add(sdiv(rec, life), [f'ratio{lag}:{raw_cols[i]}' for i in singles], 'ratio')

        for key, idx in hist:

            dsum = d[:, idx].sum(axis=1)

            add(sdiv(d[:, idx], dsum[:, None]), [f'mix{lag}:{raw_cols[i]}' for i in idx], 'mix')

            add(sdiv(sdiv(dsum, dt), sdiv(tot_cur[key], age)), [f'ratio{lag}:{key}_total'], 'ratio')



    del X

    gc.collect()

    return np.hstack(cols), names, np.array(blocks), nan_rate


def spec_for(spec_df, ids, s_names):
    if spec_df['vehicle_id'].duplicated().any():
        raise ValueError('Duplicate vehicle_id in specifications')
    feat = spec_df.set_index('vehicle_id')
    cat_cols = [c for c in feat if not pd.api.types.is_numeric_dtype(feat[c])]
    feat = pd.get_dummies(feat, columns=cat_cols, dtype=np.float32)
    feat = feat.reindex(columns=s_names, fill_value=0.0).astype(np.float32)
    aligned = feat.reindex(ids)
    if aligned.isna().all(axis=1).any():
        print('Warning: some vehicles may have missing specification values')
    return aligned.to_numpy(dtype=np.float32)


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


def decide_v2(r):
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
        return ('TRD', 'REVIEW', 'INSUFFICIENT_DATA', 'Only one risk observation; temporal assessment unavailable.')
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
    if temporal in ('MANUAL_REVIEW', 'INSUFFICIENT_DATA'):
        return floor, temporal
    return max((floor, temporal), key=lambda a: LEVEL[a]), ''


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
    review = v2 if v2 in {'MANUAL_REVIEW', 'INSUFFICIENT_DATA'} else ''
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


def _prediction_date(value=None):
    if value is not None:
        return pd.Timestamp(value).normalize().date()
    return pd.Timestamp.now(tz='Asia/Singapore').date()


def _validate_probabilities(h):
    p = h[[f'p_c{i}' for i in range(5)]].to_numpy(float)
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1,atol=1e-5):
        raise ValueError('Invalid class probabilities')
    if not np.allclose(h['Risk48'],p[:,1:].sum(axis=1),atol=1e-5):
        raise ValueError('Risk48 does not match class probabilities')


def detect_prediction_oscillation(history, min_jump=2, risk_jump=0.30):
    """Inspect the last five available model predictions per vehicle.

    Flag large class reversals anywhere in that window, or a large Risk48
    fall-and-rebound / rise-and-fall. These are screening heuristics.
    """
    flags = []
    for vid, group in history.groupby('vehicle_id', sort=False):
        recent = group.sort_values('time_step').tail(5)
        classes = recent['predicted_class'].astype(int).tolist()
        risk = recent['Risk48'].astype(float).tolist()
        patterns = []
        for i in range(len(classes) - 2):
            a, b, c = classes[i:i + 3]
            d1, d2 = b - a, c - b
            if abs(d1) >= min_jump and abs(d2) >= min_jump and d1 * d2 < 0 and abs(c - a) <= 1:
                patterns.append('CLASS:' + '->'.join(f'C{k}' for k in (a, b, c)))
        for i in range(len(risk) - 2):
            a, b, c = risk[i:i + 3]
            d1, d2 = b - a, c - b
            if abs(d1) >= risk_jump and abs(d2) >= risk_jump and d1 * d2 < 0:
                patterns.append(f'RISK48:{a:.3f}->{b:.3f}->{c:.3f}')
        flags.append({'vehicle_id': vid, 'stability_review': bool(patterns),
                      'stability_pattern': '; '.join(patterns)})
    return pd.DataFrame(flags)


def load_prediction_history_from_db(engine, vehicle_ids, prediction_at, history_table='analysis',
                                    max_history=5):
    """Read persisted previous predictions, not reconstructed model replay.

    Demo timestamps are display dates; original time_step is retained for audit.
    We exclude today's batch and ignore NO_DATA rows.
    """
    from sqlalchemy import MetaData, Table, select
    table = Table(history_table, MetaData(), schema='public', autoload_with=engine)
    required = {'vehicle_id', 'prediction_at', 'time_step', 'state', 'risk48_score'}
    missing = required - set(table.c.keys())
    if missing:
        raise ValueError(f'PostgreSQL history missing columns: {sorted(missing)}')
    if not vehicle_ids:
        return pd.DataFrame(columns=['vehicle_id','prediction_at','time_step','predicted_class','Risk48'])
    # Select only requested vehicles, and keep the most recent distinct prediction dates.
    # Chunk IDs to avoid excessive query parameters.
    frames = []
    for start in range(0, len(vehicle_ids), 500):
        ids = [int(v) for v in vehicle_ids[start:start+500]]
        query = select(table.c.vehicle_id, table.c.prediction_at, table.c.time_step,
                       table.c.state, table.c.risk48_score).where(
                           table.c.vehicle_id.in_(ids),
                           table.c.prediction_at < pd.Timestamp(prediction_at).tz_localize('Asia/Singapore')
                       )
        with engine.connect() as conn:
            frames.append(pd.read_sql(query, conn))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=['vehicle_id','prediction_at','time_step','predicted_class','Risk48'])
    df['predicted_class'] = pd.to_numeric(df['state'].astype(str).str.extract(r'^C([0-4])$')[0], errors='coerce')
    df['Risk48'] = pd.to_numeric(df['risk48_score'], errors='coerce')
    df = df.dropna(subset=['predicted_class','Risk48','time_step'])
    df = df[df['Risk48'].between(0,1)].copy()
    df['predicted_class'] = df['predicted_class'].astype(int)
    # Prefer latest stored version if duplicate prediction timestamps exist.
    df = df.sort_values(['vehicle_id','prediction_at'], kind='stable')
    df = df.drop_duplicates(['vehicle_id','prediction_at'], keep='last')
    df = df.groupby('vehicle_id', sort=False).tail(max_history)
    return df[['vehicle_id','prediction_at','time_step','predicted_class','Risk48']].reset_index(drop=True)


def combine_persisted_history_with_current(db_history, decisions, max_window=5):
    """Append the current (not yet persisted) prediction to prior DB predictions.

    Sort by prediction_at, not by Scania time_step: demo dates are synthetic.
    Discard rows with non-increasing source time_step to avoid replay duplicates.
    """
    current = decisions[['vehicle_id','time_step','predicted_class','Risk48_current','prediction_at']].copy()
    current = current.rename(columns={'Risk48_current':'Risk48'})
    current['prediction_at'] = pd.to_datetime(current['prediction_at'], utc=True)
    past = db_history.copy()
    if not past.empty:
        past['prediction_at'] = pd.to_datetime(past['prediction_at'], utc=True)
        past = past[past.vehicle_id.isin(current.vehicle_id)]
    combined = pd.concat([past[current.columns], current], ignore_index=True)
    combined = combined.sort_values(['vehicle_id','prediction_at'], kind='stable')
    combined = combined.drop_duplicates(['vehicle_id','prediction_at'], keep='last')
    # For each vehicle, eliminate stale or duplicate source readouts. Current row always retained.
    frames=[]
    for _, g in combined.groupby('vehicle_id', sort=False):
        g=g.sort_values('prediction_at', kind='stable')
        latest=g.iloc[-1]
        older=g.iloc[:-1]
        older=older[older['time_step'] < float(latest['time_step'])]
        older=older.drop_duplicates('time_step', keep='last')
        frames.append(pd.concat([older.tail(max_window-1),g.tail(1)], ignore_index=True))
    return pd.concat(frames, ignore_index=True)


def apply_stability_guardrail(decisions, history):
    """Escalate unstable vehicles to review, retaining existing inspections."""
    d = decisions.merge(detect_prediction_oscillation(history),
                        on='vehicle_id', how='left', validate='one_to_one')
    flagged = d['stability_review'].fillna(False).astype(bool)
    inspection = d['action'].isin(['INSPECT', 'PRIORITY_INSPECT'])
    review_only = flagged & ~inspection
    d.loc[review_only, 'action'] = 'MANUAL_REVIEW'
    d.loc[review_only, 'priority'] = 'REVIEW'
    d.loc[review_only, 'rule_id'] = 'TRV'
    d.loc[flagged, 'reason'] = (
        d.loc[flagged, 'reason'].astype(str)
        + ' | Human review required: unstable model predictions ('
        + d.loc[flagged, 'stability_pattern'].astype(str) + ').'
    )
    d.loc[flagged, 'evidence'] = (
        d.loc[flagged, 'evidence'].astype(str)
        + ' | Stability flag: ' + d.loc[flagged, 'stability_pattern'].astype(str)
    )
    d.loc[flagged, 'rule_trace'] = (
        d.loc[flagged, 'rule_trace'].astype(str)
        + ' | TRV stability_review=' + d.loc[flagged, 'stability_pattern'].astype(str)
        + ' | human_review_required=true'
    )
    d['priority_score'] = d['priority'].map(PRIORITY_RANK).astype(float) * 20
    print('TRV flagged:', int(flagged.sum()),
          '| manual review action:', int(review_only.sum()),
          '| inspection preserved:', int((flagged & inspection).sum()))
    return d


def build_decisions(history, prediction_at):
    """Exact Test26/Test29/Test30 decision logic, with no labels used."""
    _validate_probabilities(history)
    summary = summarize(history)
    d = enrich(history, summary)
    v2 = d.apply(decide_v2, axis=1, result_type='expand')
    v2.columns = ['v2_rule_id','v2_priority','v2_action','v2_reason']
    d = pd.concat([d.reset_index(drop=True),v2.reset_index(drop=True)],axis=1)
    d['ml_floor_action'] = d.apply(ml_floor,axis=1)
    both = [combine(a,b) for a,b in zip(d.ml_floor_action,d.v2_action)]
    d['safety_floor_action'] = [x[0] for x in both]
    d['review_flag'] = [x[1] for x in both]
    d = add_stage_features(d)
    stage = d.apply(decide_stage,axis=1)
    d = pd.concat([d.reset_index(drop=True),stage.reset_index(drop=True)],axis=1)
    d['action'] = d['stage_attention_action']
    d['priority'] = d['action'].map(PRIORITY_FOR_ACTION)
    d['rule_id'] = np.where(d.stage_rule.ne('BASE'),d.stage_rule,d.v2_rule_id)
    d['reason'] = np.where(d.stage_rule.ne('BASE'),d.stage_reason,d.v2_reason)
    d['evidence'] = d['stage_reason']
    d['rule_trace'] = d.apply(lambda r: f"ML floor={r.ml_floor_action}; V2={r.v2_action}; "
       f"V2 rule={r.v2_rule_id}; stage={r.stage_rule}; stratum={r.risk_stratum}; "
       f"near_share={r.near_share:.4f}; imminent_share={r.imminent_share:.4f}; "
       f"review={r.review_flag}; final={r.action}",axis=1)
    d['risk'] = d['Risk48_current']
    d['ranking'] = d['risk'].rank(ascending=False,method='first').astype(int)
    d['priority_score'] = d.priority.map(PRIORITY_RANK).astype(float)*20
    d['hybrid_ranking_score'] = d['risk']  # Legacy schema only: pure ML score, NOT hybrid reranking.
    d['prediction_at'] = prediction_at
    d['predicted_class'] = 0  # overwritten from latest history below
    latest = history.sort_values(['vehicle_id','time_step']).groupby('vehicle_id').tail(1)
    extra = latest[['vehicle_id','predicted_class']+[f'p_c{i}' for i in range(5)]].copy()
    d = d.drop(columns=['predicted_class']).merge(extra,on='vehicle_id',validate='one_to_one')
    d['predicted_state'] = 'C'+d.predicted_class.astype(str)
    d['confidence'] = d[[f'p_c{i}' for i in range(5)]].max(axis=1)
    d['trend_direction'] = np.select(
       [(d.Risk48_change>0)&(d.Risk48_slope>0), (d.Risk48_change<0)&(d.Risk48_slope<0)],
       ['RISING','FALLING'],default='MIXED_OR_UNKNOWN')
    d['class_worsening'] = False  # Not inferred from unreliable hard-class history.
    return d.sort_values('ranking').reset_index(drop=True)


def build_analysis_dataframe(d):
    """Map onto the original analysis columns; keep traceable legacy names."""
    out = pd.DataFrame({
       'vehicle_id':d.vehicle_id, 'prediction_at':d.prediction_at,
       'time_step':d.time_step, 'state':d.predicted_state,
       'confidence_score':d.confidence, 'history_quality':d.history_quality,
       'history_count':d.history_count, 'trend_direction':d.trend_direction,
       'class_worsening':d.class_worsening, 'rule_id':d.rule_id,
       'priority':d.priority, 'action':d.action, 'evidence':d.evidence,
       'reason':d.reason, 'rule_trace':d.rule_trace,
       'hybrid_ranking_score':d.hybrid_ranking_score,
       'priority_score':d.priority_score, 'ranking':d.ranking,
    })
    for h in HORIZONS:
        base=f'Risk{h}'
        out[f'risk{h}_score']=d[f'{base}_current']
        out[f'risk{h}_percentile']=d[f'{base}_percentile']
        out[f'risk{h}_persistence']=d[f'{base}_persistence']
        out[f'risk{h}_trend']=d[f'{base}_slope']
        out[f'risk{h}_deterioration_percentile']=d[f'{base}_change'].rank(pct=True)*100
        out[f'is_risk{h}_persistent']=d[f'{base}_persistent']
        out[f'fast_risk{h}_deterioration']=d[f'{base}_fast']
    return out


def calculate_top_shap(model, meta, operational, specifications, decisions, top_n=5,
                       batch_vehicles=150):
    """XGBoost native TreeSHAP for C0 raw margin, sign-reversed for non-C0 evidence.

    Values are NOT exact probability-SHAP for Risk48=1-P(C0).
    Rows are matched by vehicle_id, not by ranked dataframe positions.
    """
    import xgboost as xgb
    raw_cols=meta['raw_cols']
    hist,singles=detect_groups(raw_cols)
    names_expected=list(meta['feature_names'])
    ids=decisions.vehicle_id.tolist()
    out=[]
    for start in range(0,len(ids),batch_vehicles):
        sub_ids=ids[start:start+batch_vehicles]
        op=operational.loc[operational.vehicle_id.isin(sub_ids)].sort_values(['vehicle_id','time_step']).reset_index(drop=True)
        selected=op.groupby('vehicle_id',sort=False).tail(1)
        sel=pd.DataFrame({'vehicle_id':selected.vehicle_id.to_numpy(), 'row_id':selected.index.to_numpy()})
        xf,names,_,_=build_features(op,sel,raw_cols,hist,singles,meta['lags'])
        xs=spec_for(specifications,sel.vehicle_id.to_numpy(),meta['spec_columns'])
        X=np.hstack([xf,xs]).astype(np.float32)
        feature_names=list(names)+[f'spec:{n}' for n in meta['spec_columns']]
        if feature_names!=names_expected or X.shape[1]!=meta['n_features']:
            raise ValueError('SHAP feature order mismatch')
        booster=model.get_booster()
        contrib=np.asarray(booster.predict(xgb.DMatrix(X),pred_contribs=True))
        # Multiclass shape: [vehicles, classes, features + bias].
        if contrib.ndim==3 and contrib.shape==(len(X),X.shape[1]+1,5):
            contrib=contrib.transpose(0,2,1)
        if contrib.ndim==2 and contrib.shape[1]==5*(X.shape[1]+1):
            contrib=contrib.reshape(len(X),5,X.shape[1]+1)
        if contrib.shape!=(len(X),5,X.shape[1]+1):
            raise ValueError(f'Unexpected native SHAP shape: {contrib.shape}')
        signed=-contrib[:,0,:-1]  # negative C0 raw margin, not Risk48 probability.
        for i,vid in enumerate(sel.vehicle_id):
            inds=np.argsort(np.abs(signed[i]))[-min(top_n,X.shape[1]):][::-1]
            for rank,j in enumerate(inds,1):
                value=float(X[i,j])
                out.append({'vehicle_id':int(vid),'prediction_at':decisions.prediction_at.iloc[0],
                    'feature_name':names_expected[j],
                    'feature_value':value if np.isfinite(value) else None,
                    'shap_value':float(signed[i,j]), 'contribution_rank':rank})
    return pd.DataFrame(out)


def load_from_db(engine):
    operational = pd.read_sql(
        "SELECT * FROM truck_readout ORDER BY vehicle_id, time_step",
        engine,
    ).drop(columns=["readout_id", "collected_at"], errors="ignore")
    specifications = pd.read_sql(
        "SELECT vehicle_id, " + ", ".join(f"spec_{index}" for index in range(8)) + " FROM truck",
        engine,
    ).rename(columns={f"spec_{index}": f"Spec_{index}" for index in range(8)})
    if operational.empty or specifications.empty:
        raise ValueError("truck_readout and truck must not be empty")
    print(
        f"Loaded {len(operational):,} readouts for "
        f"{operational['vehicle_id'].nunique():,} vehicles"
    )
    return operational, specifications


def write_to_postgres(
    engine,
    analysis,
    shap_df,
    analysis_table="analysis",
    shap_table="prediction_shap",
):
    """Atomically insert analysis and linked SHAP rows."""
    from sqlalchemy import MetaData, Table, insert

    metadata = MetaData()
    analysis_db_table = Table(analysis_table, metadata, autoload_with=engine)
    shap_db_table = Table(shap_table, metadata, autoload_with=engine)
    if "prediction_at" not in analysis_db_table.c:
        raise RuntimeError(
            "Missing analysis.prediction_at. Run: "
            "ALTER TABLE public.analysis ADD COLUMN IF NOT EXISTS prediction_at DATE;"
        )

    key_columns = ["vehicle_id", "prediction_at"]
    if analysis[key_columns].isna().any().any():
        raise ValueError("Null vehicle_id or prediction_at within prediction batch")
    if analysis.duplicated(key_columns).any():
        raise ValueError("Duplicate vehicle_id + prediction_at within prediction batch")

    def prediction_key(vehicle_id, prediction_at):
        return int(vehicle_id), pd.Timestamp(prediction_at).date().isoformat()

    records = analysis.replace({np.nan: None}).to_dict(orient="records")
    with engine.begin() as connection:
        inserted = connection.execute(
            insert(analysis_db_table).returning(
                analysis_db_table.c.analysis_id,
                analysis_db_table.c.vehicle_id,
                analysis_db_table.c.prediction_at,
            ),
            records,
        ).fetchall()
        id_map = {
            prediction_key(row.vehicle_id, row.prediction_at): int(row.analysis_id)
            for row in inserted
        }
        if len(id_map) != len(analysis):
            raise RuntimeError("Unexpected duplicate/missing inserted prediction keys")

        shap_records = []
        for row in shap_df.to_dict(orient="records"):
            key = prediction_key(row["vehicle_id"], row["prediction_at"])
            if key not in id_map:
                raise RuntimeError(f"No inserted analysis row for SHAP prediction key {key}")
            shap_records.append(
                {
                    "analysis_id": id_map[key],
                    "feature_name": row["feature_name"],
                    "feature_value": row["feature_value"],
                    "shap_value": row["shap_value"],
                    "contribution_rank": row["contribution_rank"],
                }
            )
        if shap_records:
            connection.execute(insert(shap_db_table), shap_records)
    return len(inserted), len(shap_records)


def build_no_history_dataframe(operational, analysis_columns, as_of_date=None):
    """Return one INSUFFICIENT_DATA decision for each one-readout vehicle."""
    counts = operational.groupby("vehicle_id")["vehicle_id"].transform("size")
    single = operational.loc[counts == 1, ["vehicle_id", "time_step"]].copy()
    if single.empty:
        return pd.DataFrame(columns=analysis_columns)

    rows = pd.DataFrame(index=single.index, columns=analysis_columns, dtype=object)
    rows["vehicle_id"] = single["vehicle_id"].astype(int)
    rows["prediction_at"] = _prediction_date(as_of_date)
    rows["time_step"] = single["time_step"].to_numpy()
    rows["state"] = "NO_HISTORY"
    rows["history_quality"] = "INSUFFICIENT"
    rows["history_count"] = 1
    rows["trend_direction"] = "UNKNOWN"
    rows["class_worsening"] = False
    rows["rule_id"] = "TRD"
    rows["priority"] = "REVIEW"
    rows["action"] = "INSUFFICIENT_DATA"
    rows["evidence"] = "Only one operational readout is available; temporal change cannot be assessed."
    rows["reason"] = "A second readout is required before the predictive model and temporal rules can be applied."
    rows["rule_trace"] = "TRD | history_count=1 | action=INSUFFICIENT_DATA"
    rows["priority_score"] = PRIORITY_RANK["REVIEW"] / 5 * 100
    return rows.reset_index(drop=True)


def run(model_dir, dry_run=True, top_shap=5, batch_vehicles=150,
        as_of_date=None, username=None, password=None, host=None, port=None,
        database=None, analysis_table='analysis', shap_table='prediction_shap',
        op_csv=None, spec_csv=None, output_dir=None):
    model_dir=Path(model_dir)
    model,meta=load_model(model_dir)
    engine=None
    if op_csv and spec_csv:
        operational=pd.read_csv(op_csv)
        specifications=pd.read_csv(spec_csv)
    elif not op_csv and not spec_csv:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL
        engine=create_engine(URL.create('postgresql+psycopg2',
          username=username or os.getenv('PGUSER','tianqin-gg'),
          password=password if password is not None else os.getenv('PGPASSWORD',''),
          host=host or os.getenv('PGHOST','127.0.0.1'),
          port=port or int(os.getenv('PGPORT','5432')),
          database=database or os.getenv('PGDATABASE','fleetcare')),pool_pre_ping=True)
        operational,specifications=load_from_db(engine)
    else:
        raise ValueError('Provide both --op-csv and --spec-csv, or neither for DB mode')
    try:
        operational=operational.sort_values(['vehicle_id','time_step']).reset_index(drop=True)
        if operational.duplicated(['vehicle_id','time_step']).any():
            raise ValueError('Duplicate operational vehicle/time_step')
        prediction_at=_prediction_date(as_of_date)
        history=replay(operational,specifications,model,meta,last_n=5,batch_vehicles=batch_vehicles)
        decisions=build_decisions(history,prediction_at)
        if engine is None:
            raise ValueError('Database-backed stability requires PostgreSQL DB mode (no CSV inputs)')
        persisted = load_prediction_history_from_db(
            engine, decisions.vehicle_id.tolist(), prediction_at, history_table=analysis_table)
        stability_history = combine_persisted_history_with_current(persisted, decisions)
        print('Prior database predictions:', len(persisted),
              '| vehicles with prior history:', persisted.vehicle_id.nunique())
        decisions=apply_stability_guardrail(decisions, stability_history)
        print("Stability human-review flags:", int(decisions.stability_review.sum()))
        analysis=build_analysis_dataframe(decisions)
        shap_df=calculate_top_shap(model,meta,operational,specifications,decisions,
                                   top_n=top_shap,batch_vehicles=batch_vehicles)
        if len(shap_df)!=len(decisions)*top_shap:
            raise ValueError('Unexpected SHAP row count')
        if output_dir:
            folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
            # history.to_csv(folder/'prediction_history.csv',index=False)
            # decisions.to_csv(folder/'prediction_decisions.csv',index=False)
            # stability_history.to_csv(folder/'stability_db_history_used.csv',index=False)
            # shap_df.to_csv(folder/'prediction_shap.csv',index=False)
            # analysis.to_csv(folder/'analysis_preview.csv',index=False)
            # print('Saved CSV previews to:',folder)
        if not dry_run:
            if engine is None:
                raise ValueError('Database write requires DB mode (omit CSV inputs)')
            # Strict preflight schema check before any write.
            from sqlalchemy import inspect
            inspector=inspect(engine)
            for table,frame in [(analysis_table,analysis),(shap_table,shap_df)]:
                existing={col['name'] for col in inspector.get_columns(table)}
                required=set(frame.columns) if table==analysis_table else {
                    'analysis_id','feature_name','feature_value','shap_value','contribution_rank'}
                missing=required-existing
                if missing:
                    raise ValueError(f'Database table {table} missing columns: {sorted(missing)}')
            # Do not silently append a second decision for the same vehicle/date.
            from sqlalchemy import text as sql_text
            with engine.connect() as conn:
                existing_count=conn.execute(sql_text(
                    f'SELECT COUNT(*) FROM "{analysis_table}" WHERE prediction_at = :day'),
                    {'day':prediction_at}).scalar_one()
            if existing_count:
                raise RuntimeError(f'{analysis_table} already has {existing_count} rows for {prediction_at}; '
                                   'refusing duplicate daily inserts')
            inserted=write_to_postgres(engine,analysis,shap_df,analysis_table,shap_table)
            print('Inserted analysis/SHAP rows:',inserted)
        else:
            print('DRY RUN: no database writes')
        print('Actions:\n',decisions.action.value_counts().to_string())
        print('SHAP rows:',len(shap_df))
        print('NOTE: SHAP values explain negative C0 raw margin, not exact Risk48 probability.')
        return decisions,shap_df
    finally:
        if engine is not None: engine.dispose()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-dir',type=Path,required=True)
    ap.add_argument('--op-csv',type=Path,help='Not supported: stability requires database history')
    ap.add_argument('--spec-csv',type=Path)
    ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'processed'/'integrated_preview')
    ap.add_argument('--top-shap',type=int,default=5)
    ap.add_argument('--batch-vehicles',type=int,default=150)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--write-db',action='store_true',help='Explicitly permit DB inserts; never implied')
    ap.add_argument('--as-of-date')
    ap.add_argument('--username')
    ap.add_argument('--password')
    ap.add_argument('--host')
    ap.add_argument('--port',type=int)
    ap.add_argument('--database')
    ap.add_argument('--analysis-table',default='analysis')
    ap.add_argument('--shap-table',default='prediction_shap')
    args=ap.parse_args()
    if args.top_shap<1 or args.batch_vehicles<1: ap.error('top-shap and batch-vehicles must be positive')
    if args.write_db and args.dry_run: ap.error('Choose either --write-db or --dry-run')
    run(model_dir=args.model_dir,dry_run=not args.write_db,top_shap=args.top_shap,
        batch_vehicles=args.batch_vehicles,as_of_date=args.as_of_date,
        username=args.username,password=args.password,host=args.host,port=args.port,
        database=args.database,analysis_table=args.analysis_table,shap_table=args.shap_table,
        op_csv=args.op_csv,spec_csv=args.spec_csv,output_dir=args.output_dir)


if __name__ == "__main__":
    main()
