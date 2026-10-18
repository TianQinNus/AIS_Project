#!/usr/bin/env python3
"""C4-focused development ablation. No test-set access. Reuses prior feature implementation."""
import argparse, json, math, gc
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
import fleetcare_sampling_ablation as base

SEED=42

def spread(df, k):
    if df.empty: return df.copy()
    pieces=[]
    for _,g in df.groupby('vehicle_id',sort=False):
        g=g.sort_values('time_step')
        ids=np.unique(np.round(np.linspace(0,len(g)-1,min(k,len(g)))).astype(int))
        pieces.append(g.iloc[ids])
    return pd.concat(pieces,ignore_index=True) if pieces else df.iloc[:0].copy()

def select_training(op,tte,c4_k,neg_k=2):
    m=op[['vehicle_id','time_step']].merge(tte,on='vehicle_id',how='left',validate='many_to_one',sort=False)
    if m[['in_study_repair','length_of_study_time_step']].isna().any().any():raise ValueError('Missing TTE')
    remaining=(m.length_of_study_time_step-m.time_step).to_numpy()
    repair=m.in_study_repair.to_numpy()==1
    eligible=(repair&(remaining>=0))|(~repair&(remaining>48))
    pool=pd.DataFrame({'row_id':np.flatnonzero(eligible),'vehicle_id':m.vehicle_id.to_numpy()[eligible],
      'time_step':m.time_step.to_numpy()[eligible],
      'class_label':np.where(repair[eligible],base.tte_label(remaining[eligible]),0),
      'is_repair':repair[eligible]})
    positive=spread(pool[pool.class_label==4],c4_k)
    # Negatives: include earlier states from repaired vehicles, not only unrepaired vehicles.
    repaired_neg=spread(pool[(pool.is_repair)&(pool.class_label!=4)],neg_k)
    unrepaired_neg=pool[~pool.is_repair].sort_values(['vehicle_id','time_step']).groupby('vehicle_id',sort=False).tail(1)
    out=pd.concat([positive,repaired_neg,unrepaired_neg],ignore_index=True)
    if out.row_id.duplicated().any():raise ValueError('Duplicate readout selected')
    out['binary_label']=(out.class_label==4).astype(int)
    return out

def metadata_for_lags(meta,op,lags):
    m=dict(meta);m['lags']=list(lags)
    raw=meta['raw_cols'];hist,singles=base.detect_groups(raw)
    one=op.iloc[:1].copy().reset_index(drop=True)
    x,names,_,_=base.build_features(one,pd.DataFrame({'row_id':[0]}),raw,hist,singles,lags)
    m['feature_names']=list(names)+['spec:'+s for s in meta['spec_columns']]
    m['n_features']=len(m['feature_names'])
    return m

def scores(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    out={'ap':float(average_precision_score(y,p)),'auc':float(roc_auc_score(y,p)),
         'positives':int(y.sum()),'prevalence':float(y.mean())}
    for fraction in [.01,.02,.05,.10]:
        k=math.ceil(len(y)*fraction);order=np.argsort(-p,kind='stable')[:k]
        found=int(y[order].sum());key=f'top{int(fraction*100)}'
        out[key]={'inspected':k,'found':found,'recall':float(found/y.sum()),'precision':float(found/k)}
    return out

def model():
    return XGBClassifier(objective='binary:logistic',n_estimators=220,max_depth=4,
      learning_rate=.06,min_child_weight=5,subsample=.8,colsample_bytree=.8,
      reg_lambda=2.,tree_method='hist',eval_metric='logloss',random_state=SEED,n_jobs=4)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('raw'))
    p.add_argument('--model-dir',type=Path,default=Path('processed/final_model_xgboost'))
    p.add_argument('--output-dir',type=Path,default=Path('processed/c4_overhaul'))
    p.add_argument('--batch-vehicles',type=int,default=250)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    meta=json.loads((a.model_dir/'meta.json').read_text())
    root=a.data_dir
    tr=base.read(root/'train_operational_readouts.csv').sort_values(['vehicle_id','time_step'],kind='stable').reset_index(drop=True)
    va=base.read(root/'validation_operational_readouts.csv').sort_values(['vehicle_id','time_step'],kind='stable').reset_index(drop=True)
    tte=base.read(root/'train_tte.csv')
    trs=base.read(root/'train_specifications.csv');vas=base.read(root/'validation_specifications.csv')
    last=va.groupby('vehicle_id',sort=False).tail(1)[['vehicle_id','time_step']].reset_index(drop=True)
    last=last.merge(base.labels(root/'validation_labels.csv'),on='vehicle_id',validate='one_to_one')
    yva=(last.class_label.to_numpy()==4).astype(int)
    folds=list(StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED).split(np.zeros(len(yva)),yva))
    configurations=[('c4_3_lag5_20',3,[5,20]),('c4_10_lag5_20',10,[5,20]),
                    ('c4_3_lag1_3_5',3,[1,3,5]),('c4_10_lag1_3_5',10,[1,3,5])]
    results=[];winner=None;best=None
    for name,k,lags in configurations:
        print('\n===',name,'===',flush=True)
        m=metadata_for_lags(meta,tr,lags)
        selected=select_training(tr,tte,k)
        print('Training:',len(selected),'rows; C4:',int(selected.binary_label.sum()),
              '; repaired negative:',int(((selected.is_repair)&(selected.binary_label==0)).sum()),flush=True)
        xt=base.make_features(tr,trs,selected,m,a.batch_vehicles)
        xv=base.make_features(va,vas,last,m,a.batch_vehicles)
        yt=selected.binary_label.to_numpy()
        oof=np.zeros(len(yva),dtype=np.float32)
        for i,(fit,hold) in enumerate(folds,1):
            clf=model()
            clf.fit(np.concatenate([xt,xv[fit]]),np.r_[yt,yva[fit]])
            oof[hold]=clf.predict_proba(xv[hold])[:,1]
            print(' fold',i,'/3 complete',flush=True)
            del clf;gc.collect()
        result={'name':name,'c4_snapshots':k,'lags':lags,'train_rows':len(selected),**scores(yva,oof)}
        print(json.dumps(result,indent=2),flush=True)
        pd.DataFrame({'vehicle_id':last.vehicle_id,'true_c4':yva,'risk6':oof}).to_csv(a.output_dir/(name+'_oof.csv'),index=False)
        results.append(result)
        # Business target: Recall@Top5, then AP, then Recall@Top2.
        key=(result['top5']['recall'],result['ap'],result['top2']['recall'])
        if best is None or key>best:best=key;winner=(name,k,lags)
        del xt,xv,selected,oof;gc.collect()
    (a.output_dir/'development_results.json').write_text(json.dumps(results,indent=2))
    name,k,lags=winner
    print('\nWINNER:',name,'(DEVELOPMENT ONLY)',flush=True)
    m=metadata_for_lags(meta,tr,lags)
    chosen=select_training(tr,tte,k)
    xt=base.make_features(tr,trs,chosen,m,a.batch_vehicles)
    xv=base.make_features(va,vas,last,m,a.batch_vehicles)
    clf=model();clf.fit(np.concatenate([xt,xv]),np.r_[chosen.binary_label.to_numpy(),yva])
    joblib.dump(clf,a.output_dir/'c4_candidate.joblib')
    m.update({'task':'binary_c4','sampling':'time_spaced_c4','c4_k':k,'negative_repair_k':2,
              'selection_metric':'c4_recall_top5','evaluation':'development_oof_only'})
    (a.output_dir/'c4_candidate_meta.json').write_text(json.dumps(m,indent=2))
    print('Saved candidate model and metadata. No test data accessed; original model untouched.',flush=True)

if __name__=='__main__':main()
