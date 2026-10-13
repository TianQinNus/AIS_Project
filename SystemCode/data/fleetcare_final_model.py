"""FleetCare final inference: XGBoost + 5-point replay + stage-aware rules + SHAP.

Place beside fleetcare_historical_replay.py, test_26_temporal_rules.py,
test_29_oof_temporal_safety.py, test_30_stage_aware_rules.py,
final_model_xgboost.py, and test_20_normalized.py.

Dry run: python fleetcare_final_integrated.py --model-dir processed/final_xgboost --dry-run
Persist: python fleetcare_final_integrated.py --model-dir processed/final_xgboost
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

from fleetcare_historical_replay import load_model, replay, summarize
from test_26_temporal_rules import enrich, decide as decide_v2
from test_29_oof_temporal_safety import ml_floor, combine, LEVEL
from test_30_stage_aware_rules import add_stage_features, decide_stage

HORIZONS = (48, 24, 12, 6)
PRIORITY_RANK = {'NORMAL':0, 'LOW':1, 'MEDIUM':2, 'REVIEW':3, 'HIGH':4, 'CRITICAL':5}
PRIORITY_FOR_ACTION = {'CONTINUE':'NORMAL', 'MONITOR':'LOW', 'ENHANCED_MONITOR':'MEDIUM',
                       'INSPECT':'HIGH', 'PRIORITY_INSPECT':'CRITICAL',
                       'MANUAL_REVIEW':'REVIEW', 'COLLECT_MORE_DATA':'REVIEW'}

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
    from test_20_normalized import build_features, detect_groups
    from final_model_xgboost import spec_for
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
    """Return one COLLECT_MORE_DATA decision for each one-readout vehicle."""
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
    rows["action"] = "COLLECT_MORE_DATA"
    rows["evidence"] = "Only one operational readout is available; temporal change cannot be assessed."
    rows["reason"] = "A second readout is required before the predictive model and temporal rules can be applied."
    rows["rule_trace"] = "TRD | history_count=1 | action=COLLECT_MORE_DATA"
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
        analysis=build_analysis_dataframe(decisions)
        shap_df=calculate_top_shap(model,meta,operational,specifications,decisions,
                                   top_n=top_shap,batch_vehicles=batch_vehicles)
        if len(shap_df)!=len(decisions)*top_shap:
            raise ValueError('Unexpected SHAP row count')
        if output_dir:
            folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
            history.to_csv(folder/'prediction_history.csv',index=False)
            decisions.to_csv(folder/'prediction_decisions.csv',index=False)
            shap_df.to_csv(folder/'prediction_shap.csv',index=False)
            analysis.to_csv(folder/'analysis_preview.csv',index=False)
            print('Saved CSV previews to:',folder)
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
    ap.add_argument('--op-csv',type=Path)
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

if __name__=='__main__': main()
