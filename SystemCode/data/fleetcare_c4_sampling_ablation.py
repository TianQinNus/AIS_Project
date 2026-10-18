#!/usr/bin/env python3
"""Standalone FleetCare C4 sampling ablation; no project-local imports.

Install: pip install numpy pandas scikit-learn xgboost
Run: python fleetcare_c4_sampling_ablation.py --data-dir raw --out-dir processed/c4_ablation

IMPORTANT: Feature reconstruction follows documented Rate/Prop/Mix/Ratio/Spec
blocks, but cannot be guaranteed byte-for-byte identical to test21.build_features.
All configurations in THIS experiment use identical feature engineering and OOF folds.
The official validation labels have been reused during project development: these
are exploratory cross-fitting results, not an independent final test.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier

SEED = 42
CLASSES = list(range(5))
CONFIGS = {
    'baseline_3_3_3_3_3_random': ([3,3,3,3,3], 'random'),
    'c4_5_random': ([3,3,3,3,5], 'random'),
    'c4_10_random': ([3,3,3,3,10], 'random'),
    'c3_5_c4_10_random': ([3,3,3,5,10], 'random'),
    'c4_10_time_spaced': ([3,3,3,3,10], 'spaced'),
}

def read_csv(root, name):
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}')
    return pd.read_csv(path)

def tte_label(t):
    """C4 <=6, C3 <=12, C2 <=24, C1 <=48, otherwise C0."""
    t = np.asarray(t)
    return np.select([t <= 6, t <= 12, t <= 24, t <= 48], [4,3,2,1], default=0).astype(np.int8)

def prepare_data(root):
    tr = read_csv(root,'train_operational_readouts.csv').sort_values(['vehicle_id','time_step'],kind='stable').reset_index(drop=True)
    va = read_csv(root,'validation_operational_readouts.csv').sort_values(['vehicle_id','time_step'],kind='stable').reset_index(drop=True)
    ts = read_csv(root,'train_specifications.csv')
    vs = read_csv(root,'validation_specifications.csv')
    tte = read_csv(root,'train_tte.csv')
    vl = read_csv(root,'validation_labels.csv')
    cols = [c for c in tr if c not in ('vehicle_id','time_step')]
    if set(cols) != set(c for c in va if c not in ('vehicle_id','time_step')):
        raise ValueError('Train/validation operational columns differ')
    for label, s in [('train',tr),('validation',va)]:
        if s[['vehicle_id','time_step']].isna().any().any(): raise ValueError(f'{label} missing vehicle/time')
    m = tr[['vehicle_id','time_step']].merge(tte,on='vehicle_id',how='left',validate='many_to_one',sort=False)
    if m[['in_study_repair','length_of_study_time_step']].isna().any().any():
        raise ValueError('Missing train TTE metadata')
    remain = m['length_of_study_time_step'].to_numpy() - m['time_step'].to_numpy()
    is_rep = m['in_study_repair'].to_numpy() == 1
    eligible = (is_rep & (remain >= 0)) | (~is_rep & (remain > 48))
    pool = pd.DataFrame({'row_id':np.flatnonzero(eligible),
                         'vehicle_id':m['vehicle_id'].to_numpy()[eligible],
                         'time_step':m['time_step'].to_numpy()[eligible],
                         'is_repair':is_rep[eligible],
                         'class_label':np.where(is_rep[eligible],tte_label(remain[eligible]),0)})
    last = va.groupby('vehicle_id',sort=False).tail(1).copy()
    last['row_id'] = last.index
    last = last[['vehicle_id','row_id','time_step']].merge(vl,on='vehicle_id',how='left',validate='one_to_one')
    if last['class_label'].isna().any(): raise ValueError('Missing validation labels')
    last['class_label'] = last['class_label'].astype(int)
    if not set(last['class_label'].unique()).issubset(CLASSES): raise ValueError('Unexpected labels')
    return tr,va,ts,vs,pool,last,cols

def sample_train(pool, ks, method, k_nonrepair=2):
    chunks=[]
    for (vid,cls), group in pool[pool.is_repair].groupby(['vehicle_id','class_label'],sort=False):
        k=min(len(group),int(ks[int(cls)]))
        if method=='random':
            g=group.sample(n=k,random_state=SEED)
        else:
            g=group.sort_values('time_step',kind='stable')
            indices=np.unique(np.rint(np.linspace(0,len(g)-1,k)).astype(int))
            g=g.iloc[indices]
        g=g.copy(); g['sw']=1.0/len(g); chunks.append(g)
    for vid,group in pool[~pool.is_repair].groupby('vehicle_id',sort=False):
        g=group.sample(n=min(len(group),k_nonrepair),random_state=SEED).copy()
        g['sw']=1.0/len(g); chunks.append(g)
    if not chunks: raise ValueError('No eligible training rows')
    return pd.concat(chunks,ignore_index=True)

def groups_from_columns(cols):
    """Histogram groups use the prefix before the final underscore."""
    from collections import defaultdict
    candidate=defaultdict(list)
    for c in cols:
        if '_' in c: candidate[c.rsplit('_',1)[0]].append(c)
    hist={k:v for k,v in candidate.items() if len(v)>1}
    hist_cols={c for v in hist.values() for c in v}
    singles=[c for c in cols if c not in hist_cols]
    return hist,singles

def engineer(op, cols, lags=(5,20)):
    """Compute features for every source readout once, preserving chronological order.

    RATE: per-exposure histogram totals and singleton counters.
    PROP: histogram proportions.
    MIX: lagged nonnegative histogram increments as proportions.
    RATIO: recent rate / lifetime rate, for each rate base and lag.
    """
    hist,singles=groups_from_columns(cols)
    names=[]; blocks=[]
    x=op[cols].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(dtype=np.float32)
    idx={c:i for i,c in enumerate(cols)}
    time=pd.to_numeric(op.time_step,errors='coerce').fillna(0).to_numpy(dtype=np.float64)
    vehicle=op.vehicle_id.to_numpy()
    grp=op.groupby('vehicle_id',sort=False)
    lag_idx={}
    for lag in lags:
        ix=grp.cumcount().to_numpy() - lag
        starts=grp.cumcount().to_numpy()
        pos=np.arange(len(op))-lag
        lag_idx[lag]=np.where(starts>=lag,pos,-1)
    bases=[]; base_names=[]
    for group, members in hist.items():
        ids=[idx[c] for c in members]
        arr=np.maximum(x[:,ids],0)
        total=arr.sum(axis=1)
        bases.append(total);base_names.append('hist_total:'+group)
        p=np.divide(arr,total[:,None],out=np.zeros_like(arr),where=total[:,None]>1e-12)
        blocks.append(p);names.extend(['prop:'+c for c in members])
        for lag in lags:
            prior=lag_idx[lag]; valid=prior>=0
            delta=np.zeros_like(arr)
            delta[valid]=np.maximum(arr[valid]-arr[prior[valid]],0)
            den=delta.sum(axis=1)
            mix=np.divide(delta,den[:,None],out=np.zeros_like(delta),where=den[:,None]>1e-12)
            blocks.append(mix);names.extend([f'mix{lag}:'+c for c in members])
    for c in singles:
        bases.append(np.maximum(x[:,idx[c]],0));base_names.append('counter:'+c)
    b=np.column_stack(bases).astype(np.float32)
    rate=b/np.maximum(time[:,None]+1,1)
    blocks.append(rate);names.extend(['rate:'+n for n in base_names])
    for lag in lags:
        prior=lag_idx[lag];valid=prior>=0
        recent=np.zeros_like(b)
        dt=np.maximum(time[valid]-time[prior[valid]],1e-6)
        recent[valid]=np.maximum(b[valid]-b[prior[valid]],0)/dt[:,None]
        ratio=np.divide(recent,rate+1e-6,out=np.zeros_like(recent),where=(rate+1e-6)>0)
        ratio=np.clip(ratio,0,1000)
        blocks.append(ratio);names.extend([f'ratio{lag}:'+n for n in base_names])
    out=np.column_stack(blocks).astype(np.float32)
    out=np.nan_to_num(out,nan=0,posinf=0,neginf=0)
    return out,names,{'hist_groups':len(hist),'hist_bins':sum(map(len,hist.values())),'single_counters':len(singles)}

def specs(train_spec,val_spec, train_ids,val_ids):
    a=train_spec.copy();b=val_spec.copy()
    if a.vehicle_id.duplicated().any() or b.vehicle_id.duplicated().any(): raise ValueError('Duplicate spec ID')
    both=pd.concat([a,b],ignore_index=True)
    cats=[c for c in both if c!='vehicle_id' and not pd.api.types.is_numeric_dtype(both[c])]
    both=pd.get_dummies(both,columns=cats,dtype=np.float32)
    cols=[c for c in both if c!='vehicle_id']
    # Prefer one-hot all eight anonymous spec fields (including integer-coded categories).
    # If they are already numerical encoded categories, do not treat as continuous magnitudes.
    raw=pd.concat([a,b],ignore_index=True)
    raw_cols=[c for c in raw if c!='vehicle_id']
    raw[raw_cols]=raw[raw_cols].astype('string').fillna('MISSING')
    enc=pd.get_dummies(raw,columns=raw_cols,dtype=np.float32).set_index('vehicle_id')
    xtr=enc.reindex(train_ids).fillna(0).to_numpy(dtype=np.float32)
    xva=enc.reindex(val_ids).fillna(0).to_numpy(dtype=np.float32)
    return xtr,xva,list(enc.columns)

def model():
    return XGBClassifier(objective='multi:softprob',num_class=5,n_estimators=150,
        learning_rate=0.08,max_depth=5,min_child_weight=3,subsample=0.8,
        colsample_bytree=0.8,reg_alpha=0,reg_lambda=1,eval_metric='mlogloss',
        tree_method='hist',random_state=SEED,n_jobs=4)

def metric_row(y,p,method):
    pos=(y>0).astype(int); c4=(y==4).astype(int)
    risk48=1-p[:,0];risk6=p[:,4]
    n=len(y);k=max(1,math.ceil(.05*n)); idx=np.argsort(-risk48,kind='stable')[:k]
    k6=max(1,math.ceil(.05*n));ix6=np.argsort(-risk6,kind='stable')[:k6]
    def ap(yy,score):return float(average_precision_score(yy,score)) if yy.sum() else np.nan
    def auc(yy,score):return float(roc_auc_score(yy,score)) if len(np.unique(yy))==2 else np.nan
    # Risk48 Top20% is a transparent attention proxy, NOT the actual rule engine.
    ix20=np.argsort(-risk48,kind='stable')[:math.ceil(.20*n)]
    return dict(method=method,vehicles=n,positives=int(pos.sum()),c4_total=int(c4.sum()),
        risk48_ap=ap(pos,risk48),risk48_auc=auc(pos,risk48),
        risk6_ap=ap(c4,risk6),risk6_auc=auc(c4,risk6),
        recall_top5=float(pos[idx].sum()/max(pos.sum(),1)),
        precision_top5=float(pos[idx].mean()),found_top5=int(pos[idx].sum()),
        c4_recall_top5_risk6=float(c4[ix6].sum()/max(c4.sum(),1)),
        c4_found_top5_risk6=int(c4[ix6].sum()),
        c4_coverage_top20_risk48_proxy=float(c4[ix20].sum()/max(c4.sum(),1)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=Path('raw'))
    ap.add_argument('--out-dir',type=Path,default=Path('processed/c4_ablation'))
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--configs',nargs='*',choices=list(CONFIGS),default=list(CONFIGS))
    ap.add_argument('--val-weight',type=float,default=10)
    ap.add_argument('--lags',default='5,20')
    args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    tr,va,ts,vs,pool,last,cols=prepare_data(args.data_dir)
    lags=[int(v) for v in args.lags.split(',')]
    print('Engineering features from full readout histories...',flush=True)
    tr_f,tr_names,diagnostic=engineer(tr,cols,lags)
    va_f,va_names,_=engineer(va,cols,lags)
    if tr_names!=va_names:raise ValueError('Feature mismatch')
    tr_s,va_s,spec_names=specs(ts,vs,tr.vehicle_id.to_numpy(),va.vehicle_id.to_numpy())
    Xtr=np.hstack([tr_f,tr_s]);Xva=np.hstack([va_f,va_s]);del tr_f,va_f,tr_s,va_s
    print('Features:',Xtr.shape[1],'train rows:',len(tr),'val vehicles:',len(last),'blocks:',diagnostic,flush=True)
    print('NOTE: standalone feature reconstruction; not guaranteed identical to test21.py',flush=True)
    # Identical held-out vehicle folds for every experiment.
    splitter=StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=SEED)
    yval=last.class_label.to_numpy(dtype=int)
    folds=list(splitter.split(np.zeros(len(last)),yval))
    # No validation vehicle in its own fold's training partition.
    all_metrics=[];all_preds=[];sample_rows=[]
    for name in args.configs:
        ks,method=CONFIGS[name]
        selected=sample_train(pool,ks,method)
        sample_rows.append(dict(method=name,train_samples=len(selected),c0=int((selected.class_label==0).sum()),
            c1=int((selected.class_label==1).sum()),c2=int((selected.class_label==2).sum()),
            c3=int((selected.class_label==3).sum()),c4=int((selected.class_label==4).sum()),
            total_train_weight=float(selected.sw.sum())))
        sx=Xtr[selected.row_id.to_numpy(dtype=int)];sy=selected.class_label.to_numpy(dtype=int);sw=selected.sw.to_numpy(dtype=float)
        p=np.zeros((len(last),5),dtype=np.float32)
        for fold,(fit,held) in enumerate(folds,1):
            fit_rows=last.iloc[fit].row_id.to_numpy(dtype=int)
            hold_rows=last.iloc[held].row_id.to_numpy(dtype=int)
            x=np.vstack([sx,Xva[fit_rows]])
            y=np.r_[sy,yval[fit]]
            w=np.r_[sw,np.full(len(fit),args.val_weight)]
            m=model();m.fit(x,y,sample_weight=w)
            p[held]=m.predict_proba(Xva[hold_rows])
            print(f'{name}: fold {fold}/{args.folds} complete',flush=True)
            del x,y,w,m
        all_metrics.append(metric_row(yval,p,name))
        pred=pd.DataFrame({'method':name,'vehicle_id':last.vehicle_id.to_numpy(),
            'class_label':yval,'risk48':1-p[:,0],'risk6':p[:,4]})
        for i in CLASSES:pred[f'p_c{i}']=p[:,i]
        all_preds.append(pred)
    metrics=pd.DataFrame(all_metrics);counts=pd.DataFrame(sample_rows)
    preds=pd.concat(all_preds,ignore_index=True)
    metrics.to_csv(args.out_dir/'metrics.csv',index=False)
    counts.to_csv(args.out_dir/'sampling_counts.csv',index=False)
    preds.to_csv(args.out_dir/'oof_predictions.csv',index=False)
    # Compare paired ranking selections and C4 recovery at same fleet capacity.
    ref=preds[preds.method==args.configs[0]].copy().set_index('vehicle_id')
    comparison=[]
    for name in args.configs[1:]:
        q=preds[preds.method==name].set_index('vehicle_id').loc[ref.index]
        y=ref.class_label.to_numpy(); k=math.ceil(.05*len(y))
        rsel=set(ref.iloc[np.argsort(-ref.risk48.to_numpy())[:k]].index)
        qsel=set(q.iloc[np.argsort(-q.risk48.to_numpy())[:k]].index)
        truth=set(ref.index[ref.class_label>0]);c4=set(ref.index[ref.class_label==4])
        comparison.append(dict(method=name,baseline_top5_positives=len(rsel&truth),
            candidate_top5_positives=len(qsel&truth),net_positive_gain=len(qsel&truth)-len(rsel&truth),
            candidate_only_positives=len((qsel-rsel)&truth),baseline_only_positives=len((rsel-qsel)&truth),
            candidate_only_c4=len((qsel-rsel)&c4)))
    pd.DataFrame(comparison).to_csv(args.out_dir/'paired_top5_changes.csv',index=False)
    (args.out_dir/'run_info.json').write_text(json.dumps({'configs':args.configs,'folds':args.folds,
        'val_weight':args.val_weight,'lags':lags,'feature_count':int(Xtr.shape[1]),
        'feature_blocks':diagnostic,'spec_feature_count':len(spec_names),
        'limitations':['Exploratory validation reused during development',
        'Feature formulas reconstructed; exact original test21.build_features unavailable',
        'C4 top20% risk48 coverage is a proxy, NOT the actual rule engine']},indent=2))
    print('\n=== OOF METRICS ===\n'+metrics.round(5).to_string(index=False))
    print('\n=== SAMPLING COUNTS ===\n'+counts.to_string(index=False))
    print('\nSaved to',args.out_dir.resolve())

if __name__=='__main__':main()
