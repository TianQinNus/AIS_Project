from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo
import gc
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, delete, func, insert, select, text
from sqlalchemy.engine import URL



SGT = ZoneInfo("Asia/Singapore")
MODEL_FILE = "final_model.joblib"
MODEL_COMPONENTS = ("c4_xgboost", "c4_lightgbm", "stage_xgboost")
PRIORITY_VALUE = {
    "NORMAL": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "REVIEW": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def load_final_model(model_dir: Path):
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_version") != "FINAL_TEST54":
        raise ValueError("The selected model directory is not FINAL_TEST54")
    model_path = model_dir / MODEL_FILE
    models = joblib.load(model_path)
    if set(models) != set(MODEL_COMPONENTS):
        raise ValueError(
            f"{MODEL_FILE} must contain only {MODEL_COMPONENTS}; found {tuple(models)}"
        )
    return models, metadata


def detect_groups(columns: list[str], min_size: int = 2, max_size: int = 40):
    buckets: dict[str, list[int]] = {}
    for index, column in enumerate(columns):
        key = str(column).rsplit("_", 1)[0] if "_" in str(column) else str(column)
        buckets.setdefault(key, []).append(index)
    histograms = [(key, np.asarray(indices, dtype=int)) for key, indices in buckets.items()
                  if min_size <= len(indices) <= max_size]
    used = {index for _, indices in histograms for index in indices}
    singles = np.asarray([index for index in range(len(columns)) if index not in used], dtype=int)
    return histograms, singles


def vehicle_start(vehicle_ids: np.ndarray) -> np.ndarray:
    changed = np.r_[True, vehicle_ids[1:] != vehicle_ids[:-1]]
    starts = np.flatnonzero(changed)
    return starts[np.cumsum(changed) - 1]


def safe_divide(numerator, denominator):
    a = np.asarray(numerator, dtype=np.float64)
    b = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b
    return np.where(np.isfinite(result) & (b > 0), result, np.nan)


def make_engine(args):
    kwargs = {"username": args.username, "database": args.database}
    if args.password is not None:
        kwargs["password"] = args.password
    if args.host:
        kwargs.update(host=args.host, port=args.port)
    return create_engine(URL.create("postgresql+psycopg2", **kwargs), pool_pre_ping=True)


def dynamic_features(op: pd.DataFrame, selected: pd.DataFrame, raw_columns: list[str],
                     lags: tuple[int, ...]):
    histograms, singles = detect_groups(raw_columns)
    values = op[raw_columns].to_numpy(dtype=np.float32)
    vehicles = op["vehicle_id"].to_numpy()
    times = op["time_step"].to_numpy(dtype=np.float64)
    starts = vehicle_start(vehicles)
    positions = selected["row_id"].to_numpy(dtype=int)
    current = values[positions].astype(np.float64)
    current_time = times[positions]
    age = current_time + 1.0
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def add(array, labels):
        array = np.asarray(array, dtype=np.float32)
        arrays.append(array[:, None] if array.ndim == 1 else array)
        names.extend(labels)

    lifetime = safe_divide(current[:, singles], age[:, None])
    add(lifetime, [f"rate:{raw_columns[index]}" for index in singles])
    group_totals = {}
    for key, indices in histograms:
        total = current[:, indices].sum(axis=1)
        group_totals[key] = total
        add(safe_divide(current[:, indices], total[:, None]),
            [f"prop:{raw_columns[index]}" for index in indices])
        add(safe_divide(total, age), [f"rate:{key}_total"])
    for lag in lags:
        prior = np.maximum(positions - lag, starts[positions])
        elapsed = current_time - times[prior]
        difference = current - values[prior].astype(np.float64)
        difference = np.where(difference < 0, np.nan, difference)
        recent = safe_divide(difference[:, singles], elapsed[:, None])
        add(safe_divide(recent, lifetime),
            [f"ratio{lag}:{raw_columns[index]}" for index in singles])
        for key, indices in histograms:
            total_change = difference[:, indices].sum(axis=1)
            add(safe_divide(difference[:, indices], total_change[:, None]),
                [f"mix{lag}:{raw_columns[index]}" for index in indices])
            add(safe_divide(safe_divide(total_change, elapsed),
                            safe_divide(group_totals[key], age)),
                [f"ratio{lag}:{key}_total"])
    del values
    gc.collect()
    return np.hstack(arrays).astype(np.float32), names




def specification_features(specifications: pd.DataFrame, vehicle_ids: np.ndarray,
                           schema: list[str]) -> np.ndarray:
    if specifications["vehicle_id"].duplicated().any():
        raise ValueError("Duplicate vehicle_id in specifications")
    features = specifications.set_index("vehicle_id")
    categorical = [column for column in features
                   if not pd.api.types.is_numeric_dtype(features[column])]
    features = pd.get_dummies(features, columns=categorical, dtype=np.float32)
    aligned = features.reindex(columns=schema, fill_value=0.0).reindex(vehicle_ids)
    if aligned.isna().all(axis=1).any():
        raise ValueError("Missing specification row for one or more vehicles")
    return aligned.to_numpy(dtype=np.float32)


def build_matrix(op: pd.DataFrame, selected: pd.DataFrame, specifications: pd.DataFrame,
                 schema: list[str], raw_columns: list[str], lags=(1, 3, 5)):
    dynamic, names = dynamic_features(op, selected, raw_columns, tuple(lags))
    static = specification_features(specifications, selected["vehicle_id"].to_numpy(), schema)
    feature_names = names + [f"spec:{column}" for column in schema]
    return np.hstack([dynamic, static]).astype(np.float32), feature_names


def load_active_fleet(engine, metadata):
    quoted_raw = ", ".join(f'r."{column}"' for column in metadata["raw_columns"])
    operational_query = text(f"""
        SELECT r.vehicle_id, r.time_step, {quoted_raw}
        FROM public.truck_readout r
        JOIN public.truck t USING (vehicle_id)
        WHERE t.is_active IS TRUE
          AND COALESCE(t.in_maintenance, FALSE) IS FALSE
        ORDER BY r.vehicle_id, r.time_step
    """)
    specification_query = text("""
        SELECT vehicle_id,
               spec_0 AS "Spec_0", spec_1 AS "Spec_1",
               spec_2 AS "Spec_2", spec_3 AS "Spec_3",
               spec_4 AS "Spec_4", spec_5 AS "Spec_5",
               spec_6 AS "Spec_6", spec_7 AS "Spec_7"
        FROM public.truck
        WHERE is_active IS TRUE
          AND COALESCE(in_maintenance, FALSE) IS FALSE
        ORDER BY vehicle_id
    """)
    with engine.connect() as connection:
        operational = pd.read_sql(operational_query, connection)
        specifications = pd.read_sql(specification_query, connection)
    if specifications.empty:
        raise RuntimeError("No active vehicles outside maintenance were found")
    operational = operational.sort_values(
        ["vehicle_id", "time_step"], kind="stable"
    ).reset_index(drop=True)
    if operational.duplicated(["vehicle_id", "time_step"]).any():
        raise ValueError("Duplicate vehicle_id/time_step rows in truck_readout")
    return operational, specifications


def snapshot_cut(operational: pd.DataFrame, offset: int) -> pd.DataFrame:
    position = operational.groupby("vehicle_id", sort=False).cumcount()
    total = operational.groupby("vehicle_id", sort=False)["vehicle_id"].transform("size")
    return operational.loc[position < (total - offset)].reset_index(drop=True)


def latest_rows(cut: pd.DataFrame) -> pd.DataFrame:
    if cut.empty:
        return pd.DataFrame(columns=["row_id", "vehicle_id", "time_step"])
    selected = cut[["vehicle_id", "time_step"]].copy()
    selected["row_id"] = np.arange(len(cut), dtype=int)
    return selected.groupby("vehicle_id", sort=False).tail(1).reset_index(drop=True)


def test54_scores(matrix: np.ndarray, models: dict):
    xgb_probability = models["c4_xgboost"].predict_proba(matrix)[:, 1]
    lgb_probability = models["c4_lightgbm"].predict_proba(matrix)[:, 1]
    xgb_rank = percentile_rank(xgb_probability)
    lgb_rank = percentile_rank(lgb_probability)
    final_rank = percentile_rank((xgb_rank + lgb_rank) / 2.0)
    raw_score = (xgb_probability + lgb_probability) / 2.0
    return raw_score, final_rank, xgb_rank, lgb_rank


def score_snapshot(cut, specifications, models, metadata, snapshot_order, offset):
    selected = latest_rows(cut)
    all_ids = specifications["vehicle_id"].astype(int).tolist()
    matrix = None
    names = None
    if selected.empty:
        scored = pd.DataFrame(columns=["vehicle_id"])
    else:
        matrix, names = build_matrix(
            cut,
            selected,
            specifications,
            metadata["spec_columns"],
            metadata["raw_columns"],
            tuple(metadata["lags"]),
        )
        if names != metadata["feature_names"]:
            raise ValueError("Inference feature schema does not match FINAL_TEST54")
        raw_score, final_rank, xgb_rank, lgb_rank = test54_scores(matrix, models)
        stage_probability = models["stage_xgboost"].predict_proba(matrix)
        percentile = final_rank * 100.0
        priority_rank = pd.Series(final_rank).rank(
            ascending=False, method="first"
        ).to_numpy(int)
        top1_count = max(1, int(np.ceil(len(selected) * 0.01)))
        top5_count = max(1, int(np.ceil(len(selected) * 0.05)))
        scored = pd.DataFrame({
            "vehicle_id": selected["vehicle_id"].astype(int),
            "time_step": selected["time_step"].astype(float),
            "state": [f"C{value}" for value in stage_probability.argmax(axis=1)],
            "confidence_score": stage_probability.max(axis=1),
            "risk6_score": stage_probability[:, 4],
            "risk12_score": stage_probability[:, 3:5].sum(axis=1),
            "risk24_score": stage_probability[:, 2:5].sum(axis=1),
            "risk48_score": stage_probability[:, 1:5].sum(axis=1),
            "risk6_percentile": percentile,
            "risk12_percentile": percentile_rank(
                stage_probability[:, 3:5].sum(axis=1)
            ) * 100.0,
            "risk24_percentile": percentile_rank(
                stage_probability[:, 2:5].sum(axis=1)
            ) * 100.0,
            "risk48_percentile": percentile_rank(
                stage_probability[:, 1:5].sum(axis=1)
            ) * 100.0,
            "c4_priority_rank": priority_rank,
            "scored_fleet_size": len(selected),
            "is_top1": priority_rank <= top1_count,
            "is_top5": priority_rank <= top5_count,
            "xgb_rank": xgb_rank,
            "lgb_rank": lgb_rank,
            "model_disagreement": np.abs(xgb_rank - lgb_rank),
            "row_id": selected["row_id"].astype(int),
        })
    scored = pd.DataFrame({"vehicle_id": all_ids}).merge(
        scored, on="vehicle_id", how="left"
    )
    scored["snapshot_order"] = int(snapshot_order)
    scored["source_offset"] = int(offset)
    readout_counts = cut.groupby("vehicle_id").size()
    scored["source_readout_index"] = scored["vehicle_id"].map(
        readout_counts
    ).fillna(0).astype(int)
    for column in ("is_top1", "is_top5"):
        scored[column] = scored[column].fillna(False).astype(bool)
    missing = scored["state"].isna()
    scored.loc[missing, "state"] = "NO_DATA"
    scored.loc[missing, "confidence_score"] = np.nan
    return scored, matrix, names


def add_temporal_evidence(history: pd.DataFrame, profile: dict) -> pd.DataFrame:
    data = history.sort_values(["vehicle_id", "snapshot_order"], kind="stable").copy()
    grouped = data.groupby("vehicle_id", sort=False)
    thresholds = {
        6: profile["monitor_percentile"],
        12: 97.0,
        24: 90.0,
        48: 80.0,
    }
    for horizon, threshold in thresholds.items():
        percentile_column = f"risk{horizon}_percentile"
        trend_column = f"risk{horizon}_trend"
        rising_column = f"is_risk{horizon}_rising"
        persistence_column = f"risk{horizon}_persistence"
        persistent_column = f"is_risk{horizon}_persistent"
        deterioration_column = f"risk{horizon}_deterioration_percentile"
        data[trend_column] = grouped[percentile_column].diff()
        data[rising_column] = data[trend_column].gt(0)
        positive_trend = data[trend_column].clip(lower=0)
        data[deterioration_column] = positive_trend.groupby(
            data["snapshot_order"]
        ).rank(pct=True, method="average") * 100.0
        high = data[percentile_column].ge(threshold)
        data[persistence_column] = high.groupby(
            data["vehicle_id"], sort=False
        ).transform(lambda values: values.rolling(5, min_periods=1).mean())
        data[persistent_column] = (
            data.groupby("vehicle_id", sort=False).cumcount().ge(1)
            & data[persistence_column].ge(profile["persistence"])
        )
    data["fast_risk6_deterioration"] = (
        data["is_risk6_rising"]
        & data["risk6_deterioration_percentile"].ge(95.0)
    )
    data["risk6_volatility"] = grouped["risk6_percentile"].transform(
        lambda values: values.rolling(5, min_periods=3).std()
    )
    data["volatility_percentile"] = data.groupby("snapshot_order")[
        "risk6_volatility"
    ].rank(pct=True, method="average") * 100.0
    data["disagreement_percentile"] = data.groupby("snapshot_order")[
        "model_disagreement"
    ].rank(pct=True, method="average") * 100.0
    return data


def apply_rule(row: pd.Series, profile: dict):
    percentile = float(row["risk6_percentile"])
    count = int(row["history_count"])
    if bool(row["is_top1"]):
        return "TR1", "CRITICAL", "PRIORITY_INSPECT", "The model places this vehicle in the fleet's top 1% for near-term C4 risk."
    if bool(row["is_top5"]):
        return "TR2", "HIGH", "INSPECT", "The model places this vehicle in the fleet's top 5% for near-term C4 risk."
    if count == 0:
        return "TRD", "REVIEW", "COLLECT_MORE_DATA", "No valid prediction is available; collect additional vehicle readings."
    if count < 2:
        return "TR0", "REVIEW", "COLLECT_MORE_DATA", "Only one valid prediction is available; collect more readings before escalation."
    if (
        row["volatility_percentile"] >= profile["volatility_percentile"]
        and row["disagreement_percentile"] >= profile["disagreement_percentile"]
    ):
        return "TRV", "REVIEW", "MANUAL_REVIEW", "Recent predictions fluctuate sharply and the two C4 models disagree."
    if (
        percentile >= profile["rescue_risk6_percentile"]
        and row["risk12_percentile"] >= profile["rescue_risk12_percentile"]
        and row["risk12_deterioration_percentile"]
        >= profile["rescue_risk12_deterioration_percentile"]
    ):
        return (
            "TR2B",
            "HIGH",
            "INSPECT",
            "Near-term C4 rank and 12-unit risk are both high, while the latest "
            "Risk12 increase is elevated relative to the fleet.",
        )
    if (
        percentile >= profile["enhanced_monitor_percentile"]
        and (row["is_risk6_persistent"] or row["is_risk6_rising"])
    ):
        return "TR4", "MEDIUM", "ENHANCED_MONITOR", "The vehicle has high 24-unit risk with persistence or upward movement."
    if percentile >= profile["monitor_percentile"] and row["fast_risk6_deterioration"]:
        return "TR3", "MEDIUM", "ENHANCED_MONITOR", "Near-term risk is elevated and deteriorating faster than 95% of the fleet."
    if (
        percentile >= profile["monitor_percentile"]
        and (row["is_risk6_persistent"] or row["is_risk6_rising"])
    ):
        return "TR5", "LOW", "MONITOR", "The vehicle has elevated 48-unit risk with persistence or upward movement."
    return "TR6", "NORMAL", "CONTINUE", "The model found no condition requiring escalation."


def finish_analysis(history: pd.DataFrame, profile: dict) -> pd.DataFrame:
    data = add_temporal_evidence(history, profile)
    decisions = data.apply(
        lambda row: apply_rule(row, profile), axis=1, result_type="expand"
    )
    decisions.columns = ["rule_id", "priority", "action", "reason"]
    data = pd.concat([data, decisions], axis=1)
    data["history_quality"] = np.select(
        [data["history_count"].lt(2), data["history_count"].lt(3)],
        ["INSUFFICIENT", "LIMITED"],
        default="ADEQUATE",
    )
    data["trend_direction"] = np.select(
        [data["risk6_trend"].gt(0), data["risk6_trend"].lt(0)],
        ["RISING", "FALLING"],
        default="STABLE",
    )
    data["class_worsening"] = data.groupby("vehicle_id")["state"].transform(
        lambda values: pd.to_numeric(
            values.str.extract(r"(\d+)")[0], errors="coerce"
        ).diff().gt(0)
    )
    data["hybrid_ranking_score"] = data["risk6_percentile"] / 100.0
    data["priority_score"] = (
        data["priority"].map(PRIORITY_VALUE) * 100.0
        + data["risk6_percentile"].fillna(0.0)
    )
    data["ranking"] = data.groupby("snapshot_order")["priority_score"].rank(
        ascending=False, method="first"
    ).astype("Int64")
    data["evidence"] = data.apply(
        lambda row: (
            f"C4 risk rank P{row.risk6_percentile:.1f}; "
            f"Risk12 P{row.risk12_percentile:.1f}; "
            f"Risk12 deterioration P{row.risk12_deterioration_percentile:.1f}; "
            f"Risk24 P{row.risk24_percentile:.1f}; "
            f"Risk48 P{row.risk48_percentile:.1f}; "
            f"history={int(row.history_count)}/5; trend={row.trend_direction}"
        ),
        axis=1,
    )
    data["rule_trace"] = data.apply(
        lambda row: (
            f"{row.rule_id} | source=latest_readout | "
            f"history={int(row.history_count)}/5 | "
            f"c4_rank={row.risk6_percentile:.1f} | "
            f"risk12_rank={row.risk12_percentile:.1f} | "
            f"risk12_deterioration={row.risk12_deterioration_percentile:.1f} | "
            f"{row.action}"
        ),
        axis=1,
    )
    return data


def top_shap_rows(payloads, models, top_n):
    output = []
    booster = models["c4_xgboost"].get_booster()
    xgboost = __import__("xgboost")
    for prediction_at, scored, matrix, names in payloads:
        if matrix is None or scored.empty:
            continue
        contributions = booster.predict(xgboost.DMatrix(matrix), pred_contribs=True)
        for index, vehicle_id in enumerate(scored["vehicle_id"].astype(int)):
            top = np.argsort(np.abs(contributions[index, :-1]))[-top_n:][::-1]
            for rank, feature_index in enumerate(top, 1):
                value = float(matrix[index, feature_index])
                output.append({
                    "vehicle_id": vehicle_id,
                    "prediction_at": prediction_at,
                    "feature_name": names[feature_index],
                    "feature_value": value if np.isfinite(value) else None,
                    "shap_value": float(contributions[index, feature_index]),
                    "contribution_rank": rank,
                })
    return pd.DataFrame(output)


def write_postgres(
    engine,
    analysis,
    shap_rows,
    replace_existing=False,
    analysis_table="analysis",
    shap_table="prediction_shap",
):
    metadata = MetaData()
    target = Table(analysis_table, metadata, schema="public", autoload_with=engine)
    shap_target = Table(shap_table, metadata, schema="public", autoload_with=engine)
    prediction_times = sorted(
        set(pd.to_datetime(analysis["prediction_at"], utc=True).dt.to_pydatetime())
    )
    vehicles = sorted(analysis["vehicle_id"].astype(int).unique().tolist())
    database_columns = [
        column for column in analysis.columns
        if column in target.c and column != "analysis_id"
    ]
    id_map = {}
    with engine.begin() as connection:
        existing_query = select(target.c.analysis_id).where(
            target.c.vehicle_id.in_(vehicles),
            target.c.prediction_at.in_(prediction_times),
        )
        existing = [row[0] for row in connection.execute(existing_query)]
        if existing and not replace_existing:
            raise RuntimeError(
                f"Found {len(existing)} existing rows on target dates; use --replace-existing"
            )
        if existing:
            connection.execute(delete(target).where(target.c.analysis_id.in_(existing)))
        records = analysis[database_columns].astype(object).where(
            pd.notna(analysis[database_columns]), None
        )
        for start in range(0, len(records), 500):
            inserted = connection.execute(
                insert(target).returning(
                    target.c.analysis_id, target.c.vehicle_id, target.c.prediction_at
                ),
                records.iloc[start:start + 500].to_dict("records"),
            ).fetchall()
            for row in inserted:
                key = (int(row.vehicle_id), row.prediction_at.astimezone(SGT).date())
                if key in id_map:
                    raise RuntimeError(f"Duplicate inserted prediction key {key}")
                id_map[key] = int(row.analysis_id)
        shaped = []
        for row in shap_rows.itertuples(index=False):
            key = (int(row.vehicle_id), pd.Timestamp(row.prediction_at).date())
            if key not in id_map:
                raise RuntimeError(f"No analysis row for SHAP key {key}")
            shaped.append({
                "analysis_id": id_map[key],
                "feature_name": row.feature_name,
                "feature_value": row.feature_value,
                "shap_value": row.shap_value,
                "contribution_rank": int(row.contribution_rank),
            })
        for start in range(0, len(shaped), 1000):
            connection.execute(insert(shap_target), shaped[start:start + 1000])
    return len(analysis), len(shaped), len(existing)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the frozen FINAL_TEST54 ranker on recent database readouts"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=script_dir / "final_model_test54",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir /  "final_test54_six_day_inference",
    )
    parser.add_argument("--prediction-at", default=None)
    parser.add_argument("--history-days", type=int, default=1)
    parser.add_argument("--top-shap", type=int, default=5)
    parser.add_argument("--username", default=os.getenv("PGUSER", os.getenv("USER", "")))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD"))
    parser.add_argument("--host", default=os.getenv("PGHOST"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--database", default=os.getenv("PGDATABASE", "fleetcare"))
    parser.add_argument("--analysis-table", default="analysis")
    parser.add_argument("--shap-table", default="prediction_shap")
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    if args.history_days < 1:
        parser.error("--history-days must be positive")
    if args.top_shap < 1:
        parser.error("--top-shap must be positive")
    if args.replace_existing and not args.write_db:
        parser.error("--replace-existing requires --write-db")

    prediction_at = (
        pd.Timestamp(args.prediction_at)
        if args.prediction_at
        else pd.Timestamp.now(tz=SGT)
    )
    if prediction_at.tzinfo is None:
        prediction_at = prediction_at.tz_localize(SGT)
    else:
        prediction_at = prediction_at.tz_convert(SGT)

    models, metadata = load_final_model(args.model_dir)
    engine = make_engine(args)
    try:
        operational, specifications = load_active_fleet(engine, metadata)
        print(
            f"Model={metadata['model_version']} file={args.model_dir / MODEL_FILE}",
            flush=True,
        )
        print(
            f"Eligible fleet: {len(specifications):,} active vehicles; "
            f"{len(operational):,} readouts",
            flush=True,
        )
        frames = []
        payloads = []
        maximum_offset = args.history_days + 3
        for snapshot_order, offset in enumerate(
            range(maximum_offset, -1, -1), start=1
        ):
            cut = snapshot_cut(operational, offset)
            scored, matrix, names = score_snapshot(
                cut, specifications, models, metadata, snapshot_order, offset
            )
            frames.append(scored)
            valid = scored[scored["row_id"].notna()].sort_values(
                "row_id", kind="stable"
            ).reset_index(drop=True)
            if offset < args.history_days:
                output_at = prediction_at - pd.Timedelta(days=offset)
                payloads.append((output_at, valid, matrix, names))
            print(
                f"Snapshot {snapshot_order}/{maximum_offset + 1}: "
                f"offset={offset}, scored={len(valid):,}",
                flush=True,
            )

        history = pd.concat(frames, ignore_index=True).sort_values(
            ["vehicle_id", "snapshot_order"], kind="stable"
        ).reset_index(drop=True)
        history["history_count"] = history["state"].ne("NO_DATA").astype(int).groupby(
            history["vehicle_id"], sort=False
        ).cumsum()
        completed = finish_analysis(history, metadata["rule_profile"])
        analysis = completed.loc[
            completed["source_offset"].lt(args.history_days)
        ].copy().reset_index(drop=True)
        analysis["prediction_at"] = analysis["source_offset"].map(
            lambda offset: prediction_at - pd.Timedelta(days=int(offset))
        )
        shap_rows = top_shap_rows(payloads, models, args.top_shap)
        analysis = analysis.drop(
            columns=[
                "row_id", "xgb_rank", "lgb_rank", "model_disagreement",
                "risk6_volatility", "volatility_percentile",
                "disagreement_percentile", "fast_risk6_deterioration",
                "snapshot_order",
            ],
            errors="ignore",
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        analysis.to_csv(args.output_dir / "analysis_preview.csv", index=False)
        shap_rows.to_csv(args.output_dir / "prediction_shap_preview.csv", index=False)
        print(
            f"Final predictions: {len(analysis):,} rows across "
            f"{args.history_days} day(s)",
            flush=True,
        )
        print(analysis["action"].value_counts().to_string(), flush=True)
        if args.write_db:
            saved, saved_shap, replaced = write_postgres(
                engine,
                analysis,
                shap_rows,
                args.replace_existing,
                args.analysis_table,
                args.shap_table,
            )
            print(
                f"Committed {saved:,} analysis rows and {saved_shap:,} SHAP rows; "
                f"replaced {replaced:,} existing rows.",
                flush=True,
            )
        else:
            print("DRY RUN: database unchanged", flush=True)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
