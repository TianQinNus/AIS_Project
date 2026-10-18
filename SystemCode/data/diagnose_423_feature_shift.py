#!/usr/bin/env python3
"""Compare actual FleetCare model features across train/validation/test.
Requires fleetcare_standalone.py beside this file (same feature construction).
Read-only. Does not fit or change the model.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from fleetcare_standalone import load_model, detect_groups, build_features, spec_for


def features_for(name, data_dir, meta, batch_size=300):
    op_path = data_dir / f'{name}_operational_readouts.csv'
    spec_path = data_dir / f'{name}_specifications.csv'
    print(f'[{name}] reading {op_path}', flush=True)
    op = pd.read_csv(op_path)
    spec = pd.read_csv(spec_path)
    op = op.sort_values(['vehicle_id', 'time_step'], kind='stable').reset_index(drop=True)
    if op.duplicated(['vehicle_id','time_step']).any():
        raise ValueError(f'Duplicate vehicle/time in {name}')
    raw_cols = list(meta['raw_cols'])
    missing = set(raw_cols) - set(op.columns)
    if missing:
        raise ValueError(f'{name}: missing columns {sorted(missing)[:10]}')
    hist, singles = detect_groups(raw_cols)
    ids = op.vehicle_id.drop_duplicates().to_numpy()
    outputs = []
    for start in range(0, len(ids), batch_size):
        sub_ids = ids[start:start+batch_size]
        chunk = op[op.vehicle_id.isin(sub_ids)].reset_index(drop=True)
        latest = chunk.groupby('vehicle_id', sort=False).tail(1)
        sel = pd.DataFrame({'vehicle_id':latest.vehicle_id.to_numpy(), 'row_id':latest.index.to_numpy()})
        xf, names, blocks, _ = build_features(chunk, sel, raw_cols, hist, singles, meta['lags'])
        xs = spec_for(spec, sel.vehicle_id.to_numpy(), meta['spec_columns'])
        arr = np.hstack([xf,xs])
        expected = list(names) + [f'spec:{x}' for x in meta['spec_columns']]
        if arr.shape[1] != meta['n_features'] or expected != list(meta['feature_names']):
            raise ValueError('Feature shape/order mismatch with saved model')
        out = pd.DataFrame(arr, columns=expected)
        out.insert(0,'vehicle_id',sel.vehicle_id.to_numpy())
        outputs.append(out)
        print(f'[{name}] {min(start+batch_size,len(ids))}/{len(ids)} vehicles',flush=True)
    result = pd.concat(outputs,ignore_index=True)
    print(f'[{name}] {len(result)} vehicles, {len(result.columns)-1} features',flush=True)
    return result


def compare(a,b,an,bn,importance,groups):
    rows=[]
    for col in a.columns:
        if col == 'vehicle_id': continue
        x=pd.to_numeric(a[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
        y=pd.to_numeric(b[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
        xv=x.dropna().to_numpy(); yv=y.dropna().to_numpy()
        ks=float(ks_2samp(xv,yv).statistic) if len(xv)>1 and len(yv)>1 else np.nan
        rows.append({'feature':col,'group':groups.get(col,''),'reference':an,'target':bn,
            'reference_median':float(np.median(xv)) if len(xv) else np.nan,
            'target_median':float(np.median(yv)) if len(yv) else np.nan,
            'reference_missing':float(x.isna().mean()),'target_missing':float(y.isna().mean()),
            'ks':ks,'gain':importance.get(col,0.0)})
    return pd.DataFrame(rows)


def labels_for(data_dir, split):
    path=data_dir / f'{split}_labels.csv'
    if not path.exists(): return None
    lab=pd.read_csv(path)
    if 'vehicle_id' not in lab: raise ValueError(f'{path}: no vehicle_id')
    options=['class_label','label','class','state','target']
    found=next((x for x in options if x in lab),None)
    if found is None:
        print(f'[{split}] label columns {list(lab.columns)}: skip class-specific comparison')
        return None
    lab=lab[['vehicle_id',found]].rename(columns={found:'true_class'})
    lab['true_class']=pd.to_numeric(lab.true_class.astype(str).str.replace('C','',regex=False),errors='coerce')
    if lab.vehicle_id.duplicated().any(): raise ValueError('Duplicate label IDs')
    return lab


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('raw'))
    p.add_argument('--model-dir',type=Path,default=Path('processed/final_model_xgboost'))
    p.add_argument('--output-dir',type=Path,default=Path('processed/feature_shift_423'))
    p.add_argument('--batch-size',type=int,default=300)
    p.add_argument('--include-train',action='store_true',help='Also compare training latest readouts (larger and slower)')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    model,meta=load_model(args.model_dir)
    names=list(meta['feature_names'])
    booster=model.get_booster()
    gain_raw=booster.get_score(importance_type='gain')
    importance={name:float(gain_raw.get(f'f{i}',gain_raw.get(name,0.0))) for i,name in enumerate(names)}
    total=sum(importance.values())
    if total: importance={k:v/total for k,v in importance.items()}
    groups={n:n.split(':',1)[0].split('5')[0].split('20')[0] for n in names}
    val=features_for('validation',args.data_dir,meta,args.batch_size)
    test=features_for('test',args.data_dir,meta,args.batch_size)
    comparisons=[compare(val,test,'validation','test',importance,groups)]
    if args.include_train:
        train=features_for('train',args.data_dir,meta,args.batch_size)
        comparisons.append(compare(train,test,'train','test',importance,groups))
        del train
    shift=pd.concat(comparisons,ignore_index=True)
    shift['importance_x_ks']=shift['gain']*shift['ks']
    shift.to_csv(args.output_dir/'feature_shift_all.csv',index=False)
    for ref,part in shift.groupby('reference'):
        part.sort_values('importance_x_ks',ascending=False).head(40).to_csv(args.output_dir/f'{ref}_top_important_shift.csv',index=False)
        part.sort_values('ks',ascending=False).head(40).to_csv(args.output_dir/f'{ref}_top_ks_shift.csv',index=False)
        print(f'\n=== {ref.upper()} VS TEST: high importance x shift ===')
        print(part.sort_values('importance_x_ks',ascending=False)[['feature','group','ks','gain','reference_missing','target_missing']].head(15).to_string(index=False))
        print('\nFeature-group summary (median KS):')
        print(part.groupby('group')['ks'].median().sort_values(ascending=False).to_string())
    for split,frame in [('validation',val),('test',test)]:
        lab=labels_for(args.data_dir,split)
        if lab is None: continue
        merged=frame.merge(lab,on='vehicle_id',how='inner',validate='one_to_one')
        if len(merged)!=len(frame): raise ValueError(f'{split}: labels missing for {len(frame)-len(merged)} vehicles')
        subset=merged[merged.true_class.eq(4)]
        if subset.empty: continue
        subset[names].describe().loc[['mean','50%']].T.to_csv(args.output_dir/f'{split}_c4_feature_summary.csv')
        print(f'[{split}] C4 vehicles: {len(subset)}')
    print('\nSaved:',args.output_dir)
    print('IMPORTANT: feature shift is descriptive, not proof of the cause of poor test performance.')
    print('Do not tune using this test set and then describe it as untouched.')

if __name__=='__main__': main()
