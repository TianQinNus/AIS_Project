"""Read-only integrity check for the latest six Test54 prediction dates."""
from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


def main():
    kwargs = {
        "username": os.getenv("PGUSER", os.getenv("USER", "")),
        "database": os.getenv("PGDATABASE", "fleetcare"),
    }
    if os.getenv("PGPASSWORD") is not None:
        kwargs["password"] = os.getenv("PGPASSWORD")
    if os.getenv("PGHOST"):
        kwargs["host"] = os.getenv("PGHOST")
        kwargs["port"] = int(os.getenv("PGPORT", "5432"))
    engine = create_engine(URL.create("postgresql+psycopg2", **kwargs))
    daily_sql = text("""
        WITH target_days AS (
            SELECT DISTINCT (prediction_at AT TIME ZONE 'Asia/Singapore')::date AS day
            FROM public.analysis
            WHERE rule_trace LIKE '%TEST54_MODEL_FIRST%'
            ORDER BY day DESC
            LIMIT 6
        )
        SELECT (a.prediction_at AT TIME ZONE 'Asia/Singapore')::date AS day,
               COUNT(*) AS rows,
               COUNT(DISTINCT a.vehicle_id) AS vehicles,
               COUNT(*) FILTER (
                   WHERE a.rule_trace LIKE '%TEST54_MODEL_FIRST%'
               ) AS test54_rows
        FROM public.analysis a
        JOIN target_days d
          ON (a.prediction_at AT TIME ZONE 'Asia/Singapore')::date = d.day
        GROUP BY 1
        ORDER BY 1
    """)
    shap_sql = text("""
        WITH target_days AS (
            SELECT DISTINCT (prediction_at AT TIME ZONE 'Asia/Singapore')::date AS day
            FROM public.analysis
            WHERE rule_trace LIKE '%TEST54_MODEL_FIRST%'
            ORDER BY day DESC
            LIMIT 6
        )
        SELECT COUNT(ps.*) AS shap_rows,
               COUNT(DISTINCT ps.analysis_id) AS analyses_with_shap
        FROM public.analysis a
        JOIN target_days d
          ON (a.prediction_at AT TIME ZONE 'Asia/Singapore')::date = d.day
        LEFT JOIN public.prediction_shap ps USING (analysis_id)
        WHERE a.rule_trace LIKE '%TEST54_MODEL_FIRST%'
    """)
    with engine.connect() as connection:
        print(pd.read_sql(daily_sql, connection).to_string(index=False))
        print(pd.read_sql(shap_sql, connection).to_string(index=False))
    engine.dispose()


if __name__ == "__main__":
    main()
