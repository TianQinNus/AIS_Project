#!/usr/bin/env python3
"""FleetCare targeted repair experiment. Single file; never overwrites original model.
Choose models ONLY with validation OOF; test is post-hoc confirmation (already inspected).
"""
import argparse, json, gc, math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score
SEED=42
def detect_groups(cols, min_size=2, max_size=40):

    """按列名前缀(最后一个下划线之前)识别直方图组; 其余视为单列计数器。"""

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

def read(path):
    print('[READ]',path,flush=True)
    return pd.read_csv(path)

def labels(path):
    df=read(path)
    col=next((c for c in ['class_label','label','state','class','target','y_true'] if c in df.columns),None)
    if col is None: raise ValueError(f'No class label column: {list(df.columns)}')
    df=df[['vehicle_id',col]].rename(columns={col:'class_label'})
    if df.vehicle_id.duplicated().any(): raise ValueError('Duplicate labels')
    df['class_label']=df.class_label.astype(str).str.replace('C','',regex=False).astype(int)
    return df

def tte_label(t):
    return np.select([t<=6,t<=12,t<=24,t<=48],[4,3,2,1],default=0).astype(int)

def sample_training(op,tte,k=3,k_nonrepair=2):
    m=op[['vehicle_id','time_step']].merge(tte,on='vehicle_id',how='left',validate='many_to_one',sort=False)
    if m[['in_study_repair','length_of_study_time_step']].isna().any().any():raise ValueError('Missing TTE')
    remaining=m.length_of_study_time_step.to_numpy()-m.time_step.to_numpy()
    repaired=m.in_study_repair.to_numpy()==1
    eligible=(repaired & (remaining>=0)) | (~repaired & (remaining>48))
    pool=pd.DataFrame({'row_id':np.flatnonzero(eligible),'vehicle_id':m.vehicle_id.to_numpy()[eligible],
        'class_label':np.where(repaired[eligible],tte_label(remaining[eligible]),0), 'is_repair':repaired[eligible]})
    chunks=[]
    for flag,keys,limit in [(True,['vehicle_id','class_label'],k),(False,['vehicle_id'],k_nonrepair)]:
        p=pool[pool.is_repair==flag].sample(frac=1,random_state=SEED)
        p=p[p.groupby(keys,sort=False).cumcount()<limit].copy()
        p['sw']=1/p.groupby(keys,sort=False).row_id.transform('size')
        chunks.append(p)
    return pd.concat(chunks,ignore_index=True)

def make_features(op,spec,selected,meta,batch=250):
    raw_cols=meta['raw_cols']; hist,singles=detect_groups(raw_cols)
    op=op.sort_values(['vehicle_id','time_step'],kind='stable').reset_index(drop=True)
    if op.duplicated(['vehicle_id','time_step']).any():raise ValueError('Duplicate vehicle/time')
    ids=selected.vehicle_id.drop_duplicates().tolist()
    blocks=[]; features_expected=list(meta['feature_names'])
    for off in range(0,len(ids),batch):
        batch_ids=ids[off:off+batch]
        sub=op[op.vehicle_id.isin(batch_ids)].copy().reset_index(drop=True)
        pos=sub[['vehicle_id','time_step']].reset_index().rename(columns={'index':'row_id'})
        want=selected[selected.vehicle_id.isin(batch_ids)]
        chosen=want[['vehicle_id','time_step']].merge(pos,on=['vehicle_id','time_step'],how='left',validate='one_to_one')
        if chosen.row_id.isna().any():raise ValueError('Selected readout not found')
        xf,names,_,_=build_features(sub,chosen,raw_cols,hist,singles,meta['lags'])
        xs=spec_for(spec,chosen.vehicle_id.to_numpy(),meta['spec_columns'])
        feature_names=list(names)+[f'spec:{n}' for n in meta['spec_columns']]
        if feature_names!=features_expected:raise ValueError('Feature order differs from saved model metadata')
        x=np.hstack([xf,xs]).astype(np.float32)
        block=want.merge(chosen[['vehicle_id','time_step','row_id']],on=['vehicle_id','time_step'],validate='one_to_one')
        # chosen preserves selected order, so x is in want order
        blocks.append((want.index.to_numpy(),x))
        if off%2000==0:print('  engineered',min(off+batch,len(ids)),'/',len(ids),'vehicles',flush=True)
        del sub,xf,xs,x;gc.collect()
    result=np.empty((len(selected),len(features_expected)),dtype=np.float32)
    for idx,x in blocks:
        result[selected.index.get_indexer(idx)]=x
    return result

def make_model():
    return XGBClassifier(objective='multi:softprob',num_class=5,n_estimators=150,
        max_depth=5,learning_rate=.08,min_child_weight=3,subsample=.8,
        colsample_bytree=.8,reg_alpha=0.,reg_lambda=1.,tree_method='hist',
        eval_metric='mlogloss',random_state=SEED,n_jobs=4)

def metrics(y,p):
    risk48=p[:,1:].sum(axis=1); risk6=p[:,4]
    def calc(target,s):
        target=np.asarray(target,dtype=bool); k=math.ceil(.05*len(target)); ids=np.argsort(-s,kind='stable')[:k]
        return {'ap':float(average_precision_score(target,s)),
          'auc':float(roc_auc_score(target,s)),
          'found_top5':int(target[ids].sum()),
          'recall_top5':float(target[ids].sum()/target.sum()) if target.sum() else None}
    return {'risk48':calc(y>0,risk48),'risk6':calc(y==4,risk6),
      'accuracy':float(accuracy_score(y,p.argmax(axis=1))),
      'macro_f1':float(f1_score(y,p.argmax(axis=1),labels=[0,1,2,3,4],average='macro',zero_division=0)),
      'c4_class_recall':float(((p.argmax(axis=1)==4)&(y==4)).sum()/(y==4).sum()) if (y==4).sum() else None}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=Path('raw'))
    ap.add_argument('--model-dir',type=Path,default=Path('processed/final_model_xgboost'))
    ap.add_argument('--output-dir',type=Path,default=Path('processed/model_repair_experiment'))
    ap.add_argument('--folds',type=int,default=3)
    ap.add_argument('--batch-vehicles',type=int,default=250)
    ap.add_argument('--skip-test',action='store_true',help='Run only development CV; do not access test data')
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    meta=json.loads((args.model_dir/'meta.json').read_text())
    if 'feature_names' not in meta:raise ValueError('meta.json missing feature_names')
    root=args.data_dir
    tr=read(root/'train_operational_readouts.csv').sort_values(['vehicle_id','time_step'],kind='stable').reset_index(drop=True)
    va=read(root/'validation_operational_readouts.csv').sort_values(['vehicle_id','time_step'],kind='stable').reset_index(drop=True)
    trspec=read(root/'train_specifications.csv');vaspec=read(root/'validation_specifications.csv')
    pool=sample_training(tr,read(root/'train_tte.csv'))
    trsel=pool[['vehicle_id','row_id','class_label','sw']].copy()
    trsel['time_step']=tr.time_step.to_numpy()[trsel.row_id.to_numpy()]
    va_last=va.groupby('vehicle_id',sort=False).tail(1)[['vehicle_id','time_step']].reset_index(drop=True)
    va_last=va_last.merge(labels(root/'validation_labels.csv'),on='vehicle_id',validate='one_to_one')
    trsel=trsel.reset_index(drop=True);va_last=va_last.reset_index(drop=True)
    print('Building train feature matrix',flush=True)
    xtr=make_features(tr,trspec,trsel,meta,args.batch_vehicles)
    print('Building validation feature matrix',flush=True)
    xva=make_features(va,vaspec,va_last,meta,args.batch_vehicles)
    ytr=trsel.class_label.to_numpy();yva=va_last.class_label.to_numpy();wtr=trsel.sw.to_numpy()
    names=meta['feature_names']; rate_idx=np.array([i for i,n in enumerate(names) if n.startswith('rate:')])
    all_idx=np.arange(len(names));no_rate=np.array([i for i in all_idx if i not in set(rate_idx)])
    print('Features:',len(all_idx),'no-rate:',len(no_rate),'train:',len(ytr),'validation:',len(yva),flush=True)
    configs=[('full_423_valweight10',all_idx,10.),('no_rate_valweight10',no_rate,10.),
             ('no_rate_valweight1',no_rate,1.),('full_423_valweight1',all_idx,1.)]
    splitter=StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=SEED)
    folds=list(splitter.split(xva,yva))
    results=[]
    for name,idx,val_weight in configs:
        print('\n===',name,'===',flush=True)
        oof=np.empty((len(yva),5),dtype=np.float32)
        for fold,(fit,hold) in enumerate(folds,1):
            model=make_model()
            xx=np.concatenate([xtr[:,idx],xva[fit][:,idx]])
            yy=np.r_[ytr,yva[fit]]
            ww=np.r_[wtr,np.full(len(fit),val_weight)]
            model.fit(xx,yy,sample_weight=ww)
            oof[hold]=model.predict_proba(xva[hold][:,idx]);del xx,yy,ww,model;gc.collect()
            print('fold',fold,'/',len(folds),'done',flush=True)
        result={'method':name,**metrics(yva,oof)}
        results.append(result)
        print(json.dumps(result,indent=2),flush=True)
        pd.DataFrame({'vehicle_id':va_last.vehicle_id,'y_true':yva,
            'risk48':oof[:,1:].sum(axis=1),'risk6':oof[:,4]}).to_csv(args.output_dir/f'{name}_oof.csv',index=False)
    # Predeclare selection: primary Risk48 AP, then Risk48 recall@5, then Risk6 AP.
    best=max(results,key=lambda r:(r['risk48']['ap'],r['risk48']['recall_top5'],r['risk6']['ap']))
    print('\nWINNER BY VALIDATION ONLY:',best['method'],flush=True)
    (args.output_dir/'validation_results.json').write_text(json.dumps(results,indent=2))
    if args.skip_test:
        print('Skipped test as requested');return
    name,idx,vw=next(c for c in configs if c[0]==best['method'])
    model=make_model()
    model.fit(np.concatenate([xtr[:,idx],xva[:,idx]]),np.r_[ytr,yva],
        sample_weight=np.r_[wtr,np.full(len(yva),vw)])
    joblib.dump(model,args.output_dir/'candidate_model.joblib')
    candidate_meta={**meta,'feature_names':[names[i] for i in idx],
        'n_features':len(idx),'feature_subset_indices':idx.tolist(),
        'source_model_dir':str(args.model_dir),'experiment':name}
    (args.output_dir/'candidate_meta.json').write_text(json.dumps(candidate_meta,indent=2))
    del xtr,xva;gc.collect()
    te=read(root/'test_operational_readouts.csv').sort_values(['vehicle_id','time_step'],kind='stable').reset_index(drop=True)
    tespec=read(root/'test_specifications.csv')
    te_last=te.groupby('vehicle_id',sort=False).tail(1)[['vehicle_id','time_step']].reset_index(drop=True)
    te_last=te_last.merge(labels(root/'test_labels.csv'),on='vehicle_id',validate='one_to_one')
    xtest=make_features(te,tespec,te_last,meta,args.batch_vehicles)
    p=model.predict_proba(xtest[:,idx]);yt=te_last.class_label.to_numpy()
    test_result={'method':name,'warning':'Test was previously inspected; this is post-hoc, not fresh independent confirmation.',**metrics(yt,p)}
    print('\n=== POST-HOC TEST RESULT ===\n',json.dumps(test_result,indent=2),flush=True)
    (args.output_dir/'posthoc_test_metrics.json').write_text(json.dumps(test_result,indent=2))
    pd.DataFrame({'vehicle_id':te_last.vehicle_id,'y_true':yt,'y_pred':p.argmax(axis=1),
       'risk48':p[:,1:].sum(axis=1),'risk6':p[:,4]}).to_csv(args.output_dir/'candidate_test_predictions.csv',index=False)
    print('Saved to:',args.output_dir)
if __name__=='__main__':main()
