#!/usr/bin/env python3
"""Read-only FleetCare label/sampling audit. No fitting, no database writes."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def read(path, cols=None):
    print('[READ]', path, flush=True)
    return pd.read_csv(path, usecols=cols)

def label_table(path):
    d=read(path)
    c=next((c for c in ('class_label','label','state','class','target','y_true') if c in d),None)
    if c is None: raise ValueError(f'No label column in {path}: {list(d.columns)}')
    d=d[['vehicle_id',c]].rename(columns={c:'y'})
    d['y']=d.y.astype(str).str.replace('C','',regex=False).astype(int)
    if d.vehicle_id.duplicated().any(): raise ValueError(f'Duplicate label vehicle_id: {path}')
    if not d.y.between(0,4).all(): raise ValueError('Labels must be 0..4')
    return d

def counts(name, y):
    a=y.value_counts().reindex(range(5),fill_value=0)
    print(f'\n{name}: n={len(y):,}, positive48={(y>0).mean():.4%}, C4={(y==4).mean():.4%}')
    print(a.to_string())

def latest(path):
    d=read(path, ['vehicle_id','time_step'])
    d=d.sort_values(['vehicle_id','time_step'],kind='stable')
    if d.duplicated(['vehicle_id','time_step']).any(): raise ValueError(f'Duplicate vehicle/time: {path}')
    return d.groupby('vehicle_id',sort=False).tail(1).reset_index(drop=True),len(d)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('raw'))
    p.add_argument('--predictions',type=Path,default=Path('processed/independent_test_eval/test_predictions.csv'))
    p.add_argument('--output-dir',type=Path,default=Path('processed/label_audit'))
    args=p.parse_args(); root=args.data_dir; out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    tte=read(root/'train_tte.csv')
    for c in ('vehicle_id','in_study_repair','length_of_study_time_step'):
        if c not in tte: raise ValueError(f'Missing TTE column {c}')
    if tte.vehicle_id.duplicated().any(): raise ValueError('Duplicate train TTE vehicle_id')
    val=label_table(root/'validation_labels.csv'); test=label_table(root/'test_labels.csv')
    last={}; ids={}
    for name in ('train','validation','test'):
        l,n=latest(root/f'{name}_operational_readouts.csv')
        last[name]=l;ids[name]=set(l.vehicle_id)
        print(f'{name}: {len(l):,} vehicles, {n:,} readouts, median latest time_step={l.time_step.median():.2f}')
    print('\nVehicle ID overlaps (IDs can be reused across splits; investigate before assuming leakage):')
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        print(f'{a}/{b}: {len(ids[a]&ids[b])}')
    for name, lab in (('validation',val),('test',test)):
        merged=last[name].merge(lab,on='vehicle_id',how='outer',indicator=True)
        print(f'{name} readout-label merge: {merged._merge.value_counts().to_dict()}')
        if (merged._merge!='both').any(): raise ValueError(f'{name} unmatched labels/readouts')
        counts(name+' last readout',merged.y)
        stats=merged.groupby('y').time_step.agg(['count','median','mean','min','max'])
        print('Latest time_step by class:\n',stats.to_string())
        stats.to_csv(out/f'{name}_time_by_class.csv')
    train=last['train'].merge(tte,on='vehicle_id',validate='one_to_one',how='left')
    if train[['in_study_repair','length_of_study_time_step']].isna().any().any():raise ValueError('Train TTE missing')
    remaining=train.length_of_study_time_step-train.time_step
    repaired=train.in_study_repair.eq(1)
    eligible=(repaired & remaining.ge(0)) | (~repaired & remaining.gt(48))
    y=np.select([remaining.le(6),remaining.le(12),remaining.le(24),remaining.le(48)],[4,3,2,1],default=0)
    y=np.where(repaired,y,0)
    train['remaining']=remaining;train['eligible']=eligible;train['y']=y
    print('\nTRAIN last-readout eligibility:')
    print(pd.crosstab(repaired,eligible,rownames=['repaired'],colnames=['eligible']).to_string())
    print('Repaired vehicles remaining < 0:',int((repaired & remaining.lt(0)).sum()))
    print('Unrepaired vehicles with remaining <= 48 (excluded):',int((~repaired & remaining.le(48)).sum()))
    counts('train last eligible only',train.loc[eligible,'y'])
    print('Train last eligible time_step by class:\n',train.loc[eligible].groupby('y').time_step.agg(['count','median']).to_string())
    train.to_csv(out/'train_last_readout_label_audit.csv',index=False)
    if args.predictions.exists():
        pr=read(args.predictions)
        yc=next((c for c in ('y_true','true_class','class_label','label') if c in pr),None)
        pc=next((c for c in ('predicted_class','y_pred','pred_class') if c in pr),None)
        if yc:
            pr[yc]=pr[yc].astype(str).str.replace('C','',regex=False).astype(int)
            counts('test predictions labels',pr[yc])
            if 'vehicle_id' in pr:
                check=pr[['vehicle_id',yc]].merge(test,on='vehicle_id',validate='one_to_one')
                print('Test prediction vs label mismatch:',int((check[yc]!=check.y).sum()))
        if pc: print('Predicted class counts:\n',pr[pc].value_counts().to_string())
        for c in ('Risk48','risk48_score','Risk6','risk6_score'):
            if c in pr and yc:
                print(f'{c} mean by true class:\n',pr.groupby(yc)[c].agg(['count','mean','median']).to_string())
    print('\nAudit saved to:',out)
    print('NOTE: Train last-readout labels are NOT the same distribution as stage-sampled training rows.')
if __name__=='__main__': main()
