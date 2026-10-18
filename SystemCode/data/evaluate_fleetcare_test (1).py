#!/usr/bin/env python3
"""Independent test evaluation for FleetCare's frozen XGBClassifier.
Standalone: reproduces feature logic from the supplied fleetcare_standalone.py.
Does not train, write to database, or import project scripts.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix,
                             average_precision_score, roc_auc_score, precision_score,
                             recall_score)

def detect_groups(cols, min_size=2, max_size=40):
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
    cols, names = [], []
    def add(arr, nms):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        cols.append(arr)
        names.extend(nms)
    life = sdiv(cur[:, singles], age[:, None])
    add(life, [f'rate:{raw_cols[i]}' for i in singles])
    tot_cur = {}
    for key, idx in hist:
        tot = cur[:, idx].sum(axis=1)
        tot_cur[key] = tot
        add(sdiv(cur[:, idx], tot[:, None]), [f'prop:{raw_cols[i]}' for i in idx])
        add(sdiv(tot, age), [f'rate:{key}_total'])
    for lag in lags:
        pl = np.maximum(pos - lag, vst[pos])
        dt = t_cur - ts[pl]
        d = cur - X[pl].astype(np.float64)
        d = np.where(d < 0, np.nan, d)
        rec = sdiv(d[:, singles], dt[:, None])
        add(sdiv(rec, life), [f'ratio{lag}:{raw_cols[i]}' for i in singles])
        for key, idx in hist:
            dsum = d[:, idx].sum(axis=1)
            add(sdiv(d[:, idx], dsum[:, None]), [f'mix{lag}:{raw_cols[i]}' for i in idx])
            add(sdiv(sdiv(dsum, dt), sdiv(tot_cur[key], age)), [f'ratio{lag}:{key}_total'])
    return np.hstack(cols), names

def spec_for(spec_df, ids, s_names):
    if spec_df['vehicle_id'].duplicated().any():
        raise ValueError('Duplicate vehicle_id in specifications')
    feat = spec_df.set_index('vehicle_id')
    cat_cols = [c for c in feat if not pd.api.types.is_numeric_dtype(feat[c])]
    feat = pd.get_dummies(feat, columns=cat_cols, dtype=np.float32)
    feat = feat.reindex(columns=s_names, fill_value=0.0).astype(np.float32)
    aligned = feat.reindex(ids)
    if aligned.isna().all(axis=1).any():
        raise ValueError('Some test vehicles have no specifications')
    return aligned.to_numpy(dtype=np.float32)

def load_labels(path):
    df = pd.read_csv(path)
    if 'vehicle_id' not in df.columns:
        raise ValueError(f'test_labels.csv must contain vehicle_id; found {list(df.columns)}')
    choices = ['class_label', 'label', 'state', 'class', 'target', 'y_true']
    col = next((c for c in choices if c in df.columns), None)
    if col is None:
        raise ValueError(f'Cannot identify test label column. Found: {list(df.columns)}. Use --label-column.')
    return df, col

def metrics_binary(y, score, budget=0.05):
    n = len(y)
    k = max(1, int(np.ceil(n * budget)))
    order = np.argsort(-score, kind='stable')[:k]
    found = int(y[order].sum())
    positives = int(y.sum())
    return {
        'positives': positives, 'prevalence': float(y.mean()),
        'average_precision': float(average_precision_score(y, score)) if 0 < positives < n else None,
        'roc_auc': float(roc_auc_score(y, score)) if 0 < positives < n else None,
        'top5_count': k, 'found_top5': found,
        'recall_top5': found / positives if positives else None,
        'precision_top5': found / k,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, default=Path('processed/final_model_xgboost'))
    ap.add_argument('--data-dir', type=Path, default=Path('raw'))
    ap.add_argument('--output-dir', type=Path, default=Path('processed/independent_test_eval'))
    ap.add_argument('--label-column', default=None)
    ap.add_argument('--batch-vehicles', type=int, default=250)
    args = ap.parse_args()
    if args.batch_vehicles < 1:
        ap.error('--batch-vehicles must be >= 1')
    meta = json.loads((args.model_dir / 'meta.json').read_text(encoding='utf-8'))
    model = joblib.load(args.model_dir / 'model.joblib')
    if type(model).__name__ != 'XGBClassifier' or list(model.classes_) != [0,1,2,3,4]:
        raise ValueError('Expected fitted five-class XGBClassifier with classes [0..4]')
    op = pd.read_csv(args.data_dir / 'test_operational_readouts.csv')
    spec = pd.read_csv(args.data_dir / 'test_specifications.csv')
    labels = pd.read_csv(args.data_dir / 'test_labels.csv')
    label_col = args.label_column or next((c for c in ['class_label','label','state','class','target','y_true'] if c in labels.columns), None)
    if label_col is None:
        raise ValueError(f'No class label column found: {list(labels.columns)}. Pass --label-column COLUMN')
    if labels['vehicle_id'].duplicated().any():
        raise ValueError('Duplicate vehicle_id in test labels')
    raw_cols = list(meta['raw_cols'])
    missing = set(raw_cols + ['vehicle_id','time_step']) - set(op.columns)
    if missing:
        raise ValueError(f'Missing test operational columns: {sorted(missing)}')
    op = op.sort_values(['vehicle_id','time_step'], kind='stable').reset_index(drop=True)
    if op.duplicated(['vehicle_id','time_step']).any():
        raise ValueError('Duplicate test vehicle/time_step')
    hist, singles = detect_groups(raw_cols)
    vids = op.vehicle_id.drop_duplicates().tolist()
    frames = []
    for start in range(0,len(vids),args.batch_vehicles):
        ids = vids[start:start+args.batch_vehicles]
        subset = op[op.vehicle_id.isin(ids)].reset_index(drop=True)
        last = subset.groupby('vehicle_id',sort=False).tail(1)
        sel = pd.DataFrame({'vehicle_id':last.vehicle_id.to_numpy(), 'row_id':last.index.to_numpy()})
        xf, names = build_features(subset, sel, raw_cols, hist, singles, meta['lags'])
        xs = spec_for(spec, sel.vehicle_id.to_numpy(), meta['spec_columns'])
        X = np.hstack([xf,xs]).astype(np.float32)
        expected_names = names + [f'spec:{n}' for n in meta['spec_columns']]
        if X.shape[1] != meta['n_features']:
            raise ValueError(f'Feature count mismatch: {X.shape[1]} != {meta["n_features"]}')
        if 'feature_names' in meta and expected_names != list(meta['feature_names']):
            raise ValueError('Feature order mismatch against model metadata')
        p = np.asarray(model.predict_proba(X),dtype=float)
        if p.shape != (len(sel),5) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1,atol=1e-5):
            raise ValueError('Invalid probability output')
        batch = pd.DataFrame({'vehicle_id':sel.vehicle_id.to_numpy(), 'time_step':last.time_step.to_numpy()})
        for i in range(5): batch[f'p_c{i}'] = p[:,i]
        for horizon, idx in [(48,1),(24,2),(12,3),(6,4)]:
            batch[f'Risk{horizon}'] = p[:,idx:].sum(axis=1)
        batch['predicted_class'] = p.argmax(axis=1)
        frames.append(batch)
        print(f'Predicted {min(start+args.batch_vehicles,len(vids))}/{len(vids)} vehicles',flush=True)
    pred = pd.concat(frames,ignore_index=True)
    joined = pred.merge(labels[['vehicle_id',label_col]],on='vehicle_id',how='left',validate='one_to_one',indicator=True)
    if (joined['_merge']!='both').any():
        raise ValueError(f'{(joined["_merge"]!="both").sum()} predicted vehicles have no test label')
    joined = joined.drop(columns='_merge').rename(columns={label_col:'true_class'})
    joined['true_class'] = joined.true_class.astype(str).str.strip().str.upper().str.replace(r'^C','',regex=True)
    joined['true_class'] = pd.to_numeric(joined.true_class,errors='raise').astype(int)
    if not joined.true_class.isin(range(5)).all():
        raise ValueError('Test labels must be classes 0..4')
    y = joined.true_class.to_numpy()
    yp = joined.predicted_class.to_numpy()
    risk48 = (y>=1).astype(int)
    risk6 = (y==4).astype(int)
    result = {
        'dataset':'untouched_test', 'n_vehicles':len(joined),
        'class_counts':{f'C{i}':int((y==i).sum()) for i in range(5)},
        'accuracy':float(accuracy_score(y,yp)),
        'balanced_accuracy':float(balanced_accuracy_score(y,yp)),
        'macro_f1':float(f1_score(y,yp,labels=list(range(5)),average='macro',zero_division=0)),
        'weighted_f1':float(f1_score(y,yp,labels=list(range(5)),average='weighted',zero_division=0)),
        'risk48':metrics_binary(risk48,joined.Risk48.to_numpy()),
        'risk6':metrics_binary(risk6,joined.Risk6.to_numpy()),
        'c4_class_recall':float(recall_score(risk6,(yp==4).astype(int),zero_division=0)),
        'confusion_matrix':confusion_matrix(y,yp,labels=list(range(5))).tolist(),
        'classification_report':classification_report(y,yp,labels=list(range(5)),target_names=[f'C{i}' for i in range(5)],output_dict=True,zero_division=0),
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    joined.to_csv(args.output_dir/'test_predictions.csv',index=False)
    (args.output_dir/'test_metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame(result['confusion_matrix'],index=[f'true_C{i}' for i in range(5)],columns=[f'pred_C{i}' for i in range(5)]).to_csv(args.output_dir/'confusion_matrix.csv')
    print('\n=== INDEPENDENT TEST RESULTS ===')
    print('Vehicles:',result['n_vehicles'],'Class counts:',result['class_counts'])
    print('Accuracy:',round(result['accuracy'],5),'Macro F1:',round(result['macro_f1'],5))
    for name in ('risk48','risk6'):
        print(name.upper(),json.dumps(result[name],indent=2))
    print('C4 class recall:',round(result['c4_class_recall'],5))
    print('Saved to:',args.output_dir)

if __name__=='__main__':
    main()
