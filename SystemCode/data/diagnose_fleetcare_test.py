#!/usr/bin/env python3
"""Read-only FleetCare train/validation/test diagnostics; no fitting or DB writes.

Uses test_predictions.csv and available raw CSVs. This diagnoses input shifts,
label distributions, risk ranking and probability behaviour, not causal attribution.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from scipy.stats import ks_2samp


def read_csv(path, required=False):
    if path is None or not Path(path).exists():
        if required:
            raise FileNotFoundError(f'Required file missing: {path}')
        print(f'[SKIP] Not found: {path}')
        return None
    print(f'[READ] {path}')
    return pd.read_csv(path)


def class_series(df):
    if df is None:
        return None
    for col in ('class_label', 'true_class', 'label', 'class', 'state'):
        if col in df:
            v = df[col].astype(str).str.replace(r'^C', '', regex=True)
            return pd.to_numeric(v, errors='coerce')
    return None


def metrics(y, s, top_frac=.05):
    y=np.asarray(y).astype(int); s=np.asarray(s, dtype=float)
    valid=np.isfinite(s)
    y=y[valid];s=s[valid]
    if not len(y): return {}
    k=int(np.ceil(len(y)*top_frac))
    idx=np.argsort(-s,kind='stable')[:k]
    result={'n':int(len(y)),'positives':int(y.sum()),'prevalence':float(y.mean()),
            'top5_count':k,'found_top5':int(y[idx].sum()),
            'recall_top5':float(y[idx].sum()/y.sum()) if y.sum() else None,
            'precision_top5':float(y[idx].mean())}
    if len(np.unique(y))==2:
        result['ap']=float(average_precision_score(y,s))
        result['auc']=float(roc_auc_score(y,s))
    return result


def describe_shift(ref, target, name, out):
    common=[c for c in ref.columns.intersection(target.columns)
            if c not in ('vehicle_id',) and pd.api.types.is_numeric_dtype(ref[c]) and
            pd.api.types.is_numeric_dtype(target[c])]
    rows=[]
    for c in common:
        a=pd.to_numeric(ref[c],errors='coerce').to_numpy(dtype=float)
        b=pd.to_numeric(target[c],errors='coerce').to_numpy(dtype=float)
        af=a[np.isfinite(a)]; bf=b[np.isfinite(b)]
        if len(af)<5 or len(bf)<5: continue
        # Subsample only for statistical diagnostics to control runtime.
        rng=np.random.default_rng(42)
        if len(af)>10000: af=rng.choice(af,10000,replace=False)
        if len(bf)>10000: bf=rng.choice(bf,10000,replace=False)
        ks=float(ks_2samp(af,bf).statistic)
        rows.append({'feature':c,'ref_median':float(np.median(af)),
                     'test_median':float(np.median(bf)), 'ref_missing':float((~np.isfinite(a)).mean()),
                     'test_missing':float((~np.isfinite(b)).mean()), 'ks_statistic':ks})
    frame=pd.DataFrame(rows)
    if not frame.empty:
        frame=frame.sort_values('ks_statistic',ascending=False)
        frame.to_csv(out/f'{name}_shift.csv',index=False)
        print(f'\n{name} shift, largest KS differences (descriptive, not proof of cause):')
        print(frame.head(12).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    return frame


def last_readouts(df):
    if df is None: return None
    if not {'vehicle_id','time_step'}.issubset(df.columns):
        print('[WARN] Operational file lacks vehicle_id/time_step');return None
    df=df.sort_values(['vehicle_id','time_step'],kind='stable')
    return df.groupby('vehicle_id',sort=False).tail(1).reset_index(drop=True)


def feature_proxies(df):
    """Simple proxies for cumulative-size dependence, NOT the 423 training features."""
    if df is None: return None
    result=df[['vehicle_id','time_step']].copy()
    numeric=[c for c in df if c not in ('vehicle_id','time_step') and pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric[:30]:
        result['rate_proxy:'+c]=pd.to_numeric(df[c],errors='coerce')/(pd.to_numeric(df.time_step,errors='coerce')+1)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='raw')
    p.add_argument('--predictions',default='processed/independent_test_eval/test_predictions.csv')
    p.add_argument('--output-dir',default='processed/test_diagnostics')
    p.add_argument('--model-dir',default='processed/final_model_xgboost')
    args=p.parse_args()
    raw=Path(args.data_dir);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    pred=read_csv(args.predictions,True)
    required={'vehicle_id','true_class','predicted_class','Risk48','Risk6','p_c0','p_c4'}
    if required-set(pred):raise ValueError(f'Missing prediction columns: {sorted(required-set(pred))}')
    if pred.vehicle_id.duplicated().any():raise ValueError('Multiple predictions per test vehicle; expected latest readout only')
    if pred[list({'Risk48','Risk6','p_c0','p_c4'})].isna().any().any():raise ValueError('Missing predicted scores')
    y=pred.true_class.astype(int)
    if not y.isin(range(5)).all():raise ValueError('Unexpected true classes')
    print('\n=== TEST PREDICTION DIAGNOSTICS ===')
    print('True class counts:',y.value_counts().sort_index().to_dict())
    print('Predicted class counts:',pred.predicted_class.value_counts().sort_index().to_dict())
    cm=confusion_matrix(y,pred.predicted_class,labels=range(5))
    print('Confusion matrix:\n',cm)
    pred.groupby('true_class')[[c for c in ['p_c0','p_c1','p_c2','p_c3','p_c4','Risk48','Risk6'] if c in pred]].agg(['mean','median']).to_csv(out/'probabilities_by_true_class.csv')
    risk48=metrics(y.ne(0),pred.Risk48)
    risk6=metrics(y.eq(4),pred.Risk6)
    print('Risk48:',json.dumps(risk48,indent=2))
    print('Risk6:',json.dumps(risk6,indent=2))
    for score, positive, name in [('Risk48',y.ne(0),'risk48'),('Risk6',y.eq(4),'risk6')]:
        ranked=pred.sort_values(score,ascending=False).copy()
        ranked['rank']=np.arange(1,len(ranked)+1)
        ranked['positive_for_metric']=ranked.true_class.ne(0) if name=='risk48' else ranked.true_class.eq(4)
        ranked.head(253).to_csv(out/f'{name}_top5_vehicles.csv',index=False)
        positives=ranked[ranked.positive_for_metric]
        positives.to_csv(out/f'{name}_positive_ranks.csv',index=False)
        print(f'{name}: median positive rank {positives["rank"].median():.1f}; '
              f'positives in top 253: {int((positives["rank"]<=253).sum())}')
    print('\n=== RAW DATA COMPARISON (LATEST READOUT PER VEHICLE) ===')
    datasets={}
    for split in ('train','validation','test'):
        op=read_csv(raw/f'{split}_operational_readouts.csv')
        datasets[split]=last_readouts(op)
        if op is not None:
            print(f'{split}: {op.vehicle_id.nunique()} vehicles, {len(op)} readouts; '
                  f'latest time_step median={datasets[split].time_step.median():.2f}')
    if datasets['test'] is not None:
        missing=set(pred.vehicle_id)-set(datasets['test'].vehicle_id)
        print('Prediction vehicles missing from raw test:',len(missing))
        if missing:raise ValueError('Predictions do not align with raw test vehicle IDs')
        ts=pred[['vehicle_id','time_step']].merge(datasets['test'][['vehicle_id','time_step']],on='vehicle_id',suffixes=('_pred','_raw'))
        print('Mismatched latest time_step:',int((~np.isclose(ts.time_step_pred,ts.time_step_raw)).sum()))
    for split in ('train','validation'):
        if datasets[split] is not None and datasets['test'] is not None:
            describe_shift(datasets[split],datasets['test'],f'{split}_vs_test_raw',out)
            a=feature_proxies(datasets[split]);b=feature_proxies(datasets['test'])
            describe_shift(a,b,f'{split}_vs_test_rate_proxies',out)
    for split in ('validation','test'):
        labels=read_csv(raw/f'{split}_labels.csv')
        cl=class_series(labels)
        if cl is not None:
            print(f'{split} labels: {cl.value_counts(dropna=False).sort_index().to_dict()}')
    meta_path=Path(args.model_dir)/'meta.json'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        print('Model metadata:',{k:meta.get(k) for k in ('n_features','lags')})
        if datasets['test'] is not None:
            raw_cols=meta.get('raw_cols',[])
            print('Model raw feature columns missing in test:',sorted(set(raw_cols)-set(datasets['test']))[:30])
    (out/'test_metrics.json').write_text(json.dumps({'risk48':risk48,'risk6':risk6},indent=2))
    print('\nSaved diagnostics to:',out)
    print('NOTE: Differences are diagnostic only; do not tune against this test set and then call it untouched.')

if __name__=='__main__':main()
