#!/usr/bin/env python3
"""C4 OOF rank ensemble; optional post-hoc test. No fitting or DB writes.
Place next to fleetcare_sampling_ablation.py and fleetcare_c4_overhaul.py.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score, roc_auc_score
import fleetcare_sampling_ablation as base


def evaluate(y, score):
    y=np.asarray(y,dtype=int); score=np.asarray(score,dtype=float)
    if not np.isfinite(score).all(): raise ValueError('Nonfinite scores')
    out={'ap':float(average_precision_score(y,score)), 'auc':float(roc_auc_score(y,score))}
    for frac in (.01,.02,.05,.10):
        k=math.ceil(len(y)*frac); ix=np.argsort(-score,kind='stable')[:k]
        found=int(y[ix].sum())
        out[f'top{round(frac*100)}']={'inspected':k,'found':found,'recall':found/int(y.sum()),'precision':found/k}
    return out

def ranks(x):
    return pd.Series(x).rank(method='average',pct=True).to_numpy(float)

def read_oof(path, score_col, label_col, ids):
    if not path.is_file(): raise FileNotFoundError(path)
    d=pd.read_csv(path)
    if d.vehicle_id.duplicated().any(): raise ValueError('Duplicate vehicle_id in '+str(path))
    if score_col not in d or label_col not in d: raise ValueError(f'Missing {score_col}/{label_col} in {path}')
    d=d.set_index('vehicle_id').reindex(ids)
    if d[[score_col,label_col]].isna().any().any(): raise ValueError('Missing OOF rows in '+str(path))
    return d[score_col].to_numpy(float),d[label_col].to_numpy(int)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=Path('raw'))
    ap.add_argument('--model-dir',type=Path,default=Path('processed/final_model_xgboost'))
    ap.add_argument('--c4-dir',type=Path,default=Path('processed/c4_overhaul'))
    ap.add_argument('--binary-dir',type=Path,default=Path('processed/objective_overhaul'))
    ap.add_argument('--sampling-dir',type=Path,default=Path('processed/sampling_ablation'))
    ap.add_argument('--output-dir',type=Path,default=Path('processed/c4_ensemble'))
    ap.add_argument('--evaluate-test',action='store_true')
    ap.add_argument('--batch-vehicles',type=int,default=250)
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    # Align all development OOF by vehicle ID, never by row order.
    c4_files=sorted(a.c4_dir.glob('c4_*_oof.csv'))
    if not c4_files: raise FileNotFoundError('No c4_*_oof.csv in '+str(a.c4_dir))
    d0=pd.read_csv(c4_files[0]); ids=d0.vehicle_id.to_numpy()
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate OOF IDs')
    sources={}
    for path in c4_files:
        score,y=read_oof(path,'risk6','true_c4',ids)
        sources[path.stem]=score
    p=a.binary_dir/'binary6_w1_oof.csv'
    if p.is_file():
        score,by=read_oof(p,'risk_score','y_true',ids)
        if not np.array_equal(y,by==4):raise ValueError('binary6 OOF labels disagree')
        sources['binary6_w1']=score
    p=a.sampling_dir/'recent_eligible_oof.csv'
    if p.is_file():
        score,sy=read_oof(p,'risk6','true_class',ids)
        if not np.array_equal(y,sy==4):raise ValueError('recent eligible OOF labels disagree')
        sources['recent_eligible_risk6']=score
    print('OOF sources:',list(sources),flush=True)
    results=[]
    for name,s in sources.items():
        results.append({'name':name,'weights':{name:1.},**evaluate(y,s)})
    # Fixed, small rank-weight grid; ties resolved by AP. No test-driven weight selection.
    names=list(sources)
    rr={n:ranks(v) for n,v in sources.items()}
    from itertools import combinations
    for n1,n2 in combinations(names,2):
        for w in (.25,.5,.75):
            s=w*rr[n1]+(1-w)*rr[n2]
            results.append({'name':f'{n1}__{n2}__{w:.2f}', 'weights':{n1:w,n2:1-w},**evaluate(y,s)})
    # Three-model blends: small finite candidate set, no combinatorial search.
    if {'c4_10_lag1_3_5','binary6_w1','recent_eligible_risk6'}.issubset(rr):
        for w in ((.5,.3,.2),(.4,.4,.2),(.3,.5,.2),(.3,.3,.4)):
            keys=('c4_10_lag1_3_5','binary6_w1','recent_eligible_risk6')
            s=sum(v*rr[k] for k,v in zip(keys,w))
            results.append({'name':'tri_'+''.join(str(int(v*10)) for v in w),
                            'weights':dict(zip(keys,w)),**evaluate(y,s)})
    results.sort(key=lambda d:(d['top5']['found'],d['ap'],d['top2']['found']),reverse=True)
    (a.output_dir/'development_comparison.json').write_text(json.dumps(results,indent=2))
    print('\n=== DEVELOPMENT OOF TOP CANDIDATES ===',flush=True)
    for r in results[:12]:print(f"{r['name']:<60} AP={r['ap']:.5f} top1={r['top1']['found']} top2={r['top2']['found']} top5={r['top5']['found']} top10={r['top10']['found']}")
    saved_names={'c4_10_lag1_3_5','binary6_w1','recent_eligible_risk6'}
    runnable=[r for r in results if set(r['weights']).issubset(saved_names)]
    if not runnable: raise ValueError('No candidate can be evaluated with currently saved models')
    winner=runnable[0]
    if winner['name']!=results[0]['name']:
        print('NOTE: absolute OOF winner requires an unsaved C4 variant; selecting best runnable saved-model candidate instead.')
    (a.output_dir/'ensemble_selection.json').write_text(json.dumps(winner,indent=2))
    print('\nWINNER (DEVELOPMENT ONLY):',json.dumps(winner,indent=2),flush=True)
    if not a.evaluate_test:
        print('No test data accessed. Add --evaluate-test for POST-HOC test comparison.')
        return
    meta=json.loads((a.model_dir/'meta.json').read_text())
    c4meta=json.loads((a.c4_dir/'c4_candidate_meta.json').read_text())
    op=base.read(a.data_dir/'test_operational_readouts.csv').sort_values(['vehicle_id','time_step'],kind='stable').reset_index(drop=True)
    spec=base.read(a.data_dir/'test_specifications.csv')
    last=op.groupby('vehicle_id',sort=False).tail(1)[['vehicle_id','time_step']].reset_index(drop=True)
    last=last.merge(base.labels(a.data_dir/'test_labels.csv'),on='vehicle_id',validate='one_to_one')
    yt=(last.class_label.to_numpy()==4).astype(int)
    # Reuse identical features for models sharing metadata; short lag model has its own metadata.
    need=set(winner['weights'])
    preds={}
    full_needed='binary6_w1' in need or 'recent_eligible_risk6' in need
    if full_needed:
        xfull=base.make_features(op,spec,last,meta,a.batch_vehicles)
        if 'binary6_w1' in need:
            m=joblib.load(a.binary_dir/'binary_risk6.joblib')
            if m.n_features_in_!=xfull.shape[1]:raise ValueError('Binary model feature mismatch')
            preds['binary6_w1']=m.predict_proba(xfull)[:,1]
        if 'recent_eligible_risk6' in need:
            m=joblib.load(a.sampling_dir/'sampling_candidate_model.joblib')
            if m.n_features_in_!=xfull.shape[1]:raise ValueError('Sampling model feature mismatch')
            preds['recent_eligible_risk6']=m.predict_proba(xfull)[:,4]
        del xfull
    for n in need:
        if n.startswith('c4_'):
            if n!='c4_10_lag1_3_5':
                raise ValueError('Only saved c4_10_lag1_3_5 model available; winner uses '+n+
                    '. Run development comparison, then train/save this C4 configuration before Test.')
            xc4=base.make_features(op,spec,last,c4meta,a.batch_vehicles)
            m=joblib.load(a.c4_dir/'c4_candidate.joblib')
            if m.n_features_in_!=xc4.shape[1]:raise ValueError('C4 feature mismatch')
            preds[n]=m.predict_proba(xc4)[:,1]
            del xc4
    if len(need)==1:
        testscore=next(iter(preds.values()))
    else:
        testscore=sum(w*ranks(preds[n]) for n,w in winner['weights'].items())
    testresult={'winner':winner['name'],'weights':winner['weights'],**evaluate(yt,testscore),
        'warning':'Post-hoc test. Previously inspected test set is not fresh independent confirmation.'}
    print('\n=== POST-HOC TEST ===\n'+json.dumps(testresult,indent=2),flush=True)
    (a.output_dir/'posthoc_test_metrics.json').write_text(json.dumps(testresult,indent=2))
    pd.DataFrame({'vehicle_id':last.vehicle_id,'true_c4':yt,'ensemble_score':testscore,
       **{n:preds[n] for n in preds}}).to_csv(a.output_dir/'posthoc_test_predictions.csv',index=False)

if __name__=='__main__':main()
