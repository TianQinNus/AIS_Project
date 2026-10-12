"""Read-only deployment checks. Never modifies business or knowledge tables."""
import argparse
import json
import os
from sqlalchemy import text
from database import create_postgres_engine
from knowledge_rag import KnowledgeRetriever
from rag import get_vehicle_analysis

ANALYSIS_FIELDS=set("analysis_id vehicle_id prediction_at time_step history_count risk48_score risk24_score risk12_score risk24_percentile rule_trace hybrid_ranking_score priority_score state confidence_score risk48_percentile risk12_percentile risk6_score risk6_percentile history_quality risk6_persistence risk6_deterioration_percentile class_worsening rule_id priority action evidence reason".split())
SHAP_FIELDS=set("analysis_id feature_name feature_value shap_value contribution_rank".split())


def check_setup(*,engine=None,skip_model=False,vehicle_id=None):
    owned=engine is None
    report={"checks":[],"ready_for_search":False,"ready_for_vehicle_queries":False,
            "llm_environment_configured":bool(os.getenv("OPENAI_API_KEY") and os.getenv("FLEETCARE_LLM_MODEL")),
            "live_llm_tested":False}
    def record(name,passed,**details):
        report["checks"].append({"name":name,"passed":bool(passed),**details})
    try:
        engine=engine if engine is not None else create_postgres_engine()
        with engine.connect() as db:
            db.execute(text("SET TRANSACTION READ ONLY"))
            record("database_connection",True,version=db.execute(text("SELECT version()")).scalar_one())
            extension=db.execute(text("SELECT e.extversion,n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid=e.extnamespace WHERE e.extname='vector'")).first()
            record("pgvector_in_public",bool(extension and extension[1]=="public"),extension_version=extension[0] if extension else None)
            columns={}
            for table in ["analysis","prediction_shap"]:
                columns[table]=set(db.execute(text("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=:table"),{"table":table}).scalars())
            missing={"analysis":sorted(ANALYSIS_FIELDS-columns["analysis"]),"prediction_shap":sorted(SHAP_FIELDS-columns["prediction_shap"])}
            business_ok=not any(missing.values())
            record("business_schema",business_ok,missing_fields=missing)
        try:
            with engine.connect() as db:
                db.execute(text("SET TRANSACTION READ ONLY"))
                version=db.execute(text("SELECT active_version FROM rag.kb_state WHERE id=1")).scalar_one()
                metadata=db.execute(text("SELECT metadata FROM rag.kb_versions WHERE version=:v"),{"v":version}).scalar_one()
                count=db.execute(text("SELECT count(*) FROM rag.kb_chunks WHERE version=:v AND public.vector_dims(embedding)=384"),{"v":version}).scalar_one()
                record("knowledge_version",metadata.get("schema_version")==3 and count==metadata.get("chunk_count"),chunks=count,version=version)
        except Exception as exc:
            record("knowledge_version",False,error_type=type(exc).__name__,next_step="Run migrate_knowledge.py with an ingestion account")
        if business_ok:
            try:
                if vehicle_id is None:
                    with engine.connect() as db:
                        vehicle_id=db.execute(text("SELECT vehicle_id FROM public.analysis ORDER BY analysis_id DESC LIMIT 1")).scalar()
                result=get_vehicle_analysis(vehicle_id=int(vehicle_id),engine=engine) if vehicle_id is not None else {"success":False}
                report["ready_for_vehicle_queries"]=result.get("success",False)
                record("vehicle_lookup",report["ready_for_vehicle_queries"],vehicle_id=vehicle_id,shap_rows=len(result.get("shap",[])))
            except Exception as exc:
                record("vehicle_lookup",False,error_type=type(exc).__name__)
        if not skip_model:
            try:
                retrieval=KnowledgeRetriever(engine=engine).search("Brake air leaks inspection",kind="maintenance")
                report["ready_for_search"]=bool(retrieval["hits"])
                record("local_model_and_search",report["ready_for_search"],backend=retrieval["retrieval_backend"],hits=len(retrieval["hits"]))
            except Exception as exc:
                record("local_model_and_search",False,error_type=type(exc).__name__,next_step="Check knowledge tables and run prepare_embedding_model.py --allow-download")
    except Exception as exc:
        record("database_configuration_or_connection",False,error_type=type(exc).__name__,next_step="Check PostgreSQL host/port/database/account")
    finally:
        if owned and engine is not None:
            engine.dispose()
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--skip-model",action="store_true")
    parser.add_argument("--vehicle-id",type=int)
    args=parser.parse_args()
    result=check_setup(skip_model=args.skip_model,vehicle_id=args.vehicle_id)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if all(c["passed"] for c in result["checks"]) else 1)
