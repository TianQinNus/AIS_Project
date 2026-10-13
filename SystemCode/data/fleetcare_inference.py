"""FleetCare production inference: one latest prediction per vehicle.

This production entry point uses the final trained model, feature engineering
and temporal rule policy. It consumes every currently available readout and
persists only the latest decision per vehicle.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HORIZONS = (48, 24, 12, 6)
CLASS_NAMES = {i: f"C{i}" for i in range(5)}
CLASS_DESCRIPTIONS = {
    0: "No repair predicted within 48 operational units",
    1: "Repair proximity: 24-48 operational units",
    2: "Repair proximity: 12-24 operational units",
    3: "Repair proximity: 6-12 operational units",
    4: "Repair proximity: 0-6 operational units",
}
PRIORITY_RANK = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "REVIEW": 3, "HIGH": 4, "CRITICAL": 5}
HIGH_PERCENTILES = {48: 80, 24: 85, 12: 90, 6: 95}
PRIORITY_RISK6_PERCENTILE = 99.5
INSPECT_RISK12_PERCENTILE = 99.5
UNSTABLE_CONFIDENCE_THRESHOLD = 0.35
UNSTABLE_MARGIN_THRESHOLD = 0.03
LARGE_CLASS_JUMP = 2
MIN_LARGE_JUMPS = 3


def create_history_features(df, feature_cols):
    """Reproduce the final model's training-time history features."""
    df = df.sort_values(["vehicle_id", "time_step"]).reset_index(drop=True).copy()
    if df.duplicated(["vehicle_id", "time_step"]).any():
        raise ValueError("Duplicate (vehicle_id, time_step) records; resolve before inference")
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing raw features: {sorted(missing)}")

    group = df.groupby("vehicle_id", sort=False)
    df["time_gap"] = group["time_step"].diff().astype(np.float32)
    safe_gap = df["time_gap"].replace(0, np.nan)
    previous = group[feature_cols].shift(1).astype(np.float32)
    previous.columns = [f"{column}__prev" for column in feature_cols]
    current = df[feature_cols].to_numpy(dtype=np.float32)
    previous_values = previous.to_numpy()
    delta = pd.DataFrame(
        current - previous_values,
        columns=[f"{column}__delta" for column in feature_cols],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = (current - previous_values) / safe_gap.to_numpy(dtype=np.float32)[:, None]
    rate = pd.DataFrame(rates, columns=[f"{column}__rate" for column in feature_cols])
    parts = [df, previous, delta, rate]

    for lag in (3, 5):
        old = group[feature_cols].shift(lag).to_numpy(dtype=np.float32)
        parts.append(
            pd.DataFrame(current - old, columns=[f"{column}__change{lag}" for column in feature_cols])
        )
    for window in (3, 5):
        rolling = (
            group[feature_cols]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .reindex(df.index)
            .astype(np.float32)
        )
        rolling.columns = [f"{column}__mean{window}" for column in feature_cols]
        parts.append(rolling)
        std = (
            group[feature_cols]
            .rolling(window=window, min_periods=2)
            .std()
            .reset_index(level=0, drop=True)
            .reindex(df.index)
            .astype(np.float32)
        )
        std.columns = [f"{column}__std{window}" for column in feature_cols]
        parts.append(std)
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


def _slope(frame, column):
    if len(frame) < 2:
        return 0.0
    x = frame["time_step"].to_numpy(dtype=float)
    y = frame[column].to_numpy(dtype=float)
    if np.ptp(x) == 0 or not np.isfinite(y).all():
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _historical_features(group):
    group = group.sort_values("time_step")
    recent = group.tail(5)
    last_three = group.tail(3)
    current = group.iloc[-1]
    history_count = len(group)
    classes = recent["predicted_class"].astype(int).tolist()
    class_jumps = np.abs(np.diff(np.asarray(classes, dtype=int)))
    output = {
        "vehicle_id": current["vehicle_id"],
        "time_step": current["time_step"],
        "history_count": history_count,
        "history_quality": "STRONG" if history_count >= 5 else "MODERATE" if history_count >= 3 else "LIMITED",
        "recent_class_history": " -> ".join(f"C{value}" for value in classes),
        "class_change": classes[-1] - classes[0],
        "class_stability": float(np.mean(np.asarray(classes) == classes[-1])),
        "max_recent_class_jump": int(class_jumps.max()) if len(class_jumps) else 0,
        "large_class_jump_count": int((class_jumps >= LARGE_CLASS_JUMP).sum()),
        "previous_class": classes[-2] if history_count >= 2 else np.nan,
        "time_gap": float(group.iloc[-1]["time_step"] - group.iloc[-2]["time_step"])
        if history_count >= 2
        else np.nan,
    }
    for horizon in HORIZONS:
        risk = f"Risk{horizon}"
        previous_risk = float(group.iloc[-2][risk]) if history_count >= 2 else np.nan
        current_risk = float(current[risk])
        output[f"previous_{risk}"] = previous_risk
        output[f"{risk}_change"] = current_risk - previous_risk if history_count >= 2 else 0.0
        output[f"{risk}_slope"] = (
            (current_risk - previous_risk) / output["time_gap"]
            if history_count >= 2 and output["time_gap"] > 0
            else 0.0
        )
        output[f"{risk}_trend"] = _slope(recent, risk)
        output[f"recent_{risk}_mean"] = float(last_three[risk].mean())
    return output


def _temporal_rule(row):
    predicted_class = int(row["predicted_class"])
    high = {h: bool(row[f"Risk{h}_percentile"] >= HIGH_PERCENTILES[h]) for h in HORIZONS}
    persistent = {h: bool(row[f"is_risk{h}_persistent"]) for h in HORIZONS}
    fast = {h: bool(row[f"fast_Risk{h}_deterioration"]) for h in HORIZONS}
    worsening = bool(row["is_class_worsening"])
    limited = bool(row["limited_history"])
    uncertain = bool(row["is_prediction_uncertain"])
    volatile = bool(row["is_prediction_volatile"])
    confidence = float(row["confidence"])
    margin = float(row["probability_margin"])
    extreme_risk6 = bool(row["Risk6_percentile"] >= PRIORITY_RISK6_PERCENTILE)
    extreme_risk12 = bool(row["Risk12_percentile"] >= INSPECT_RISK12_PERCENTILE)

    evidence = [f"current state C{predicted_class}"]
    for horizon in HORIZONS:
        if high[horizon]:
            evidence.append(f"Risk{horizon} above fleet P{HIGH_PERCENTILES[horizon]}")
        if persistent[horizon]:
            evidence.append(f"Risk{horizon} persistently elevated")
        if fast[horizon]:
            evidence.append(f"Risk{horizon} increasing unusually fast")
    if worsening:
        evidence.append("recent class worsening")
    if uncertain:
        evidence.append(f"uncertain class prediction (confidence={confidence:.3f}, margin={margin:.3f})")
    if volatile:
        evidence.append(
            f"repeated class volatility (max jump={int(row['max_recent_class_jump'])}, "
            f"large jumps={int(row['large_class_jump_count'])})"
        )

    if limited and predicted_class >= 3:
        rule, priority, action = "TR0", "REVIEW", "MANUAL_REVIEW"
        reason = "The current C3/C4 prediction has fewer than three historical observations; review before escalation."
    elif uncertain and volatile:
        rule, priority, action = "TRV", "REVIEW", "MANUAL_REVIEW"
        reason = "Repeated large class changes with low-confidence, near-tied probabilities; the decision state is marked unstable for manual review."
    elif predicted_class == 4 and extreme_risk6 and (persistent[6] or fast[6] or fast[12]):
        rule, priority, action = "TR1", "CRITICAL", "PRIORITY_INSPECT"
        reason = "C4 is supported by Risk6 in the fleet's top 0.5% and persistent or rapidly deteriorating near-term evidence."
    elif predicted_class >= 3 and extreme_risk12 and (persistent[12] or fast[12] or fast[6] or worsening):
        rule, priority, action = "TR2", "HIGH", "INSPECT"
        reason = "C3/C4 is supported by Risk12 in the fleet's top 0.5% together with persistence or recent deterioration."
    elif predicted_class < 3 and extreme_risk6 and persistent[6] and fast[6]:
        rule, priority, action = "TR3", "HIGH", "INSPECT"
        reason = "Although the hard class is below C3, Risk6 is in the fleet's top 0.5%, persistent and rapidly deteriorating."
    elif predicted_class == 2 or (high[24] and (persistent[24] or fast[24])):
        rule, priority, action = "TR4", "MEDIUM", "ENHANCED_MONITOR"
        reason = "C2 or elevated Risk24 with persistent or deteriorating medium-term evidence supports enhanced monitoring."
    elif predicted_class == 1 or (high[48] and (persistent[48] or fast[48])):
        rule, priority, action = "TR5", "LOW", "MONITOR"
        reason = "C1 or elevated Risk48 with persistent or deteriorating early-risk evidence supports monitoring."
    else:
        rule, priority, action = "TR6", "NORMAL", "CONTINUE"
        reason = "No combined class, horizon-risk and temporal evidence supports a higher maintenance action."

    return pd.Series(
        {
            "rule_id": rule,
            "priority": priority,
            "action": action,
            "reason": reason,
            "evidence": "; ".join(evidence),
            "evidence_score": sum(high.values()) + sum(persistent.values()) + sum(fast.values()) + int(worsening),
            "rule_trace": (
                f"{rule} | class=C{predicted_class} | confidence={confidence:.3f} | margin={margin:.3f} | "
                f"max_jump={int(row['max_recent_class_jump'])} | "
                f"large_jumps={int(row['large_class_jump_count'])} | "
                + " | ".join(f"Risk{h}_pct={row[f'Risk{h}_percentile']:.1f}" for h in HORIZONS)
                + f" | action={action}"
            ),
        }
    )


def _prediction_date(value=None):
    if value is not None:
        return pd.Timestamp(value).normalize().date()
    return pd.Timestamp.now(tz="Asia/Singapore").date()


class FleetCarePredictor:
    """Run the final model once and return the latest decision per vehicle."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "fleetcare_vehicle_aware_history_xgb.joblib")
        self.feature_columns = joblib.load(self.model_dir / "fleetcare_history_feature_columns.joblib")
        self.raw_feature_columns = joblib.load(self.model_dir / "fleetcare_raw_feature_columns.joblib")

    def predict(self, operational, specifications, as_of_date=None):
        if not {"vehicle_id", "time_step"}.issubset(operational.columns):
            raise ValueError("operational requires vehicle_id and time_step")
        if "vehicle_id" not in specifications or specifications["vehicle_id"].duplicated().any():
            raise ValueError("specifications requires one unique row per vehicle_id")

        operational = operational.sort_values(["vehicle_id", "time_step"]).reset_index(drop=True).copy()
        if operational.duplicated(["vehicle_id", "time_step"]).any():
            raise ValueError("Duplicate (vehicle_id, time_step) rows")

        sizes = operational.groupby("vehicle_id")["vehicle_id"].transform("size")
        no_history_count = operational.loc[sizes == 1, "vehicle_id"].nunique()
        eligible = operational.loc[sizes >= 2].copy()
        if eligible.empty:
            raise ValueError("No vehicles have at least 2 readouts")

        prediction_at = _prediction_date(as_of_date)
        print(f"Running latest inference for {eligible['vehicle_id'].nunique():,} vehicles on {prediction_at}")

        features = create_history_features(eligible, self.raw_feature_columns)
        features = features.merge(specifications, on="vehicle_id", how="left", validate="many_to_one")
        features = features.sort_values(["vehicle_id", "time_step"]).reset_index(drop=True)

        x = features.drop(columns=["vehicle_id"], errors="ignore").reindex(columns=self.feature_columns)
        if x.columns.tolist() != list(self.feature_columns):
            raise RuntimeError("Model feature order mismatch")

        proba = np.asarray(self.model.predict_proba(x), dtype=float)
        if proba.shape != (len(features), 5) or not np.isfinite(proba).all():
            raise ValueError(f"Invalid five-class probability matrix: {proba.shape}")
        if not np.allclose(proba.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Class probabilities do not sum to one")

        features["predicted_class"] = np.argmax(proba, axis=1)
        features["confidence"] = np.max(proba, axis=1)
        for cls in range(5):
            features[f"P_C{cls}"] = proba[:, cls]
        sorted_proba = np.sort(proba, axis=1)
        features["probability_margin"] = sorted_proba[:, -1] - sorted_proba[:, -2]
        for horizon, start in ((48, 1), (24, 2), (12, 3), (6, 4)):
            features[f"Risk{horizon}"] = proba[:, start:].sum(axis=1)

        current_idx = features.groupby("vehicle_id", sort=False).tail(1).index
        current = features.loc[current_idx].copy()
        current["prediction_step"] = 1
        current["prediction_at"] = prediction_at

        temporal_rows = []
        for _, group in features.groupby("vehicle_id", sort=False):
            record = _historical_features(group)
            record["prediction_step"] = 1
            temporal_rows.append(record)
        temporal = pd.DataFrame(temporal_rows)

        output = current[
            ["vehicle_id", "prediction_step", "time_step", "predicted_class", "confidence", "probability_margin"]
            + [f"P_C{i}" for i in range(5)]
            + [f"Risk{h}" for h in HORIZONS]
        ].merge(
            temporal,
            on=["vehicle_id", "prediction_step", "time_step"],
            validate="one_to_one",
        )

        grouped_features = features.groupby("vehicle_id", sort=False)
        for horizon in HORIZONS:
            risk = f"Risk{horizon}"
            trend = f"{risk}_trend"
            output[f"{risk}_percentile"] = output[risk].rank(pct=True, method="average") * 100
            output[f"{trend}_percentile"] = output[trend].rank(pct=True, method="average") * 100
            output[f"fast_{risk}_deterioration"] = (
                (output[trend] > 0)
                & (output[f"{trend}_percentile"] >= 90)
                & (output["history_count"] >= 3)
            )

            cutoff = float(output[risk].quantile(HIGH_PERCENTILES[horizon] / 100))
            high_count = {
                vehicle_id: int((group[risk].tail(5) >= cutoff).sum())
                for vehicle_id, group in grouped_features
            }
            output[f"{risk}_high_count_5"] = output["vehicle_id"].map(high_count)
            output[f"{risk}_persistence"] = (
                output[f"{risk}_high_count_5"] / output["history_count"].clip(upper=5)
            )
            output[f"is_risk{horizon}_persistent"] = (
                (output[f"{risk}_persistence"] >= 0.60)
                & (output["history_count"] >= 3)
            )

        output["is_class_worsening"] = output["class_change"] > 0
        output["is_prediction_stable"] = output["class_stability"] >= 0.60
        output["limited_history"] = output["history_count"] < 3
        output["is_prediction_uncertain"] = (
            (output["confidence"] < UNSTABLE_CONFIDENCE_THRESHOLD)
            & (output["probability_margin"] < UNSTABLE_MARGIN_THRESHOLD)
        )
        output["is_prediction_volatile"] = (
            (output["max_recent_class_jump"] >= LARGE_CLASS_JUMP)
            & (output["large_class_jump_count"] >= MIN_LARGE_JUMPS)
        )
        output["fast_near_term_deterioration"] = (
            output["fast_Risk6_deterioration"] | output["fast_Risk12_deterioration"]
        )

        rising = sum((output[f"Risk{h}_trend"] > 0).astype(int) for h in HORIZONS)
        output["trend_direction"] = np.select(
            [rising >= 3, rising == 0],
            ["RISING", "STABLE_OR_FALLING"],
            default="MIXED",
        )

        reasoning = output.apply(_temporal_rule, axis=1)
        output = pd.concat([output.reset_index(drop=True), reasoning.reset_index(drop=True)], axis=1)
        output["predicted_state"] = output["predicted_class"].map(CLASS_NAMES)
        unstable = output["is_prediction_uncertain"] & output["is_prediction_volatile"]
        output.loc[unstable, "predicted_state"] = "UNSTABLE"
        output["state_description"] = output["predicted_class"].map(CLASS_DESCRIPTIONS)
        output.loc[unstable, "state_description"] = (
            "Unstable low-confidence class sequence; review the risk trajectory and source readouts"
        )
        output["priority_rank"] = output["priority"].map(PRIORITY_RANK)
        output["priority_score"] = output["priority_rank"] / 5 * 100
        output["hybrid_ranking_score"] = (
            0.40 * output["Risk6_percentile"]
            + 0.15 * output["Risk12_percentile"]
            + 0.10 * output["Risk24_percentile"]
            + 0.05 * output["Risk48_percentile"]
            + 0.15 * output["priority_score"]
            + 0.10 * output["Risk6_trend_percentile"]
            + 0.05 * output["Risk6_persistence"] * 100
        )

        output = output.sort_values("hybrid_ranking_score", ascending=False, kind="stable").reset_index(drop=True)
        output["ranking"] = np.arange(1, len(output) + 1)
        output["prediction_at"] = prediction_at

        self._selected_model_input = x.loc[current_idx].reset_index(drop=True)
        self._selected_keys = current[
            ["vehicle_id", "time_step", "prediction_step", "prediction_at"]
        ].reset_index(drop=True)

        expected = eligible["vehicle_id"].nunique()
        if len(output) != expected or output["vehicle_id"].duplicated().any():
            raise RuntimeError(f"Latest prediction validation failed: {len(output)} != {expected}")
        print(
            f"Predicted one latest row for {len(output):,} vehicles; "
            f"{no_history_count:,} one-readout vehicles require more data"
        )
        return output


def build_analysis_dataframe(result):
    """Map the final model output to the PostgreSQL analysis schema."""
    columns = {
        "vehicle_id": "vehicle_id",
        "prediction_at": "prediction_at",
        "time_step": "time_step",
        "predicted_state": "state",
        "confidence": "confidence_score",
        "history_quality": "history_quality",
        "history_count": "history_count",
        "trend_direction": "trend_direction",
        "is_class_worsening": "class_worsening",
        "rule_id": "rule_id",
        "priority": "priority",
        "action": "action",
        "evidence": "evidence",
        "reason": "reason",
        "rule_trace": "rule_trace",
        "hybrid_ranking_score": "hybrid_ranking_score",
        "priority_score": "priority_score",
        "ranking": "ranking",
    }
    for horizon in HORIZONS:
        columns.update(
            {
                f"Risk{horizon}": f"risk{horizon}_score",
                f"Risk{horizon}_percentile": f"risk{horizon}_percentile",
                f"Risk{horizon}_persistence": f"risk{horizon}_persistence",
                f"Risk{horizon}_trend": f"risk{horizon}_trend",
                f"Risk{horizon}_trend_percentile": f"risk{horizon}_deterioration_percentile",
                f"is_risk{horizon}_persistent": f"is_risk{horizon}_persistent",
                f"fast_Risk{horizon}_deterioration": f"fast_risk{horizon}_deterioration",
            }
        )
    return result[list(columns)].rename(columns=columns).copy()


def calculate_top_shap(predictor, result, top_n=5):
    """Calculate top feature contributions for each latest vehicle prediction."""
    import shap

    pipeline = predictor.model
    if not hasattr(pipeline, "steps"):
        raise TypeError("SHAP expects a sklearn Pipeline ending in XGBoost")
    preprocessor = pipeline[:-1]
    estimator = pipeline.steps[-1][1]
    transformed = preprocessor.transform(predictor._selected_model_input)
    try:
        names = [
            str(value).replace("numeric__", "").replace("categorical__", "")
            for value in preprocessor.get_feature_names_out()
        ]
    except (AttributeError, ValueError):
        names = [f"feature_{index}" for index in range(transformed.shape[1])]

    values = shap.TreeExplainer(estimator)(transformed).values
    values = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    if values.ndim == 2:
        values = values[:, :, None]
    elif values.ndim == 3 and values.shape[1] != transformed.shape[1] and values.shape[2] == transformed.shape[1]:
        values = np.transpose(values, (0, 2, 1))
    if values.ndim != 3 or values.shape[1] != transformed.shape[1] or values.shape[2] not in (1, 5):
        raise ValueError(f"Unexpected SHAP shape {values.shape}")

    predicted = result.set_index(["vehicle_id", "prediction_step"])["predicted_class"].to_dict()
    rows = []
    for row_index, key in predictor._selected_keys.iterrows():
        vehicle_id = int(key["vehicle_id"])
        prediction_step = int(key["prediction_step"])
        predicted_class = int(predicted[(vehicle_id, prediction_step)])
        contributions = values[
            row_index,
            :,
            predicted_class if values.shape[2] == 5 else 0,
        ]
        top_indices = np.argsort(np.abs(contributions))[-min(top_n, len(contributions)):][::-1]
        transformed_row = transformed[row_index]
        transformed_row = (
            transformed_row.toarray().ravel()
            if hasattr(transformed_row, "toarray")
            else np.asarray(transformed_row).ravel()
        )
        for rank, feature_index in enumerate(top_indices, 1):
            feature_value = float(transformed_row[feature_index])
            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "prediction_at": key["prediction_at"],
                    "feature_name": names[feature_index],
                    "feature_value": feature_value if np.isfinite(feature_value) else None,
                    "shap_value": float(contributions[feature_index]),
                    "contribution_rank": rank,
                }
            )
    return pd.DataFrame(rows)


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


def run_fleetcare_prediction(
    model_dir,
    username="tianqin-gg",
    password=None,
    host="127.0.0.1",
    port=5432,
    database="fleetcare",
    analysis_table="analysis",
    shap_table="prediction_shap",
    top_shap=5,
    persist=True,
    as_of_date=None,
):
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL

    engine = create_engine(
        URL.create(
            "postgresql+psycopg2",
            username=username,
            password=password if password is not None else os.environ.get("PGPASSWORD", ""),
            host=host,
            port=port,
            database=database,
        ),
        pool_pre_ping=True,
    )
    try:
        operational, specifications = load_from_db(engine)
        predictor = FleetCarePredictor(model_dir)
        result = predictor.predict(operational, specifications, as_of_date=as_of_date)
        analysis = build_analysis_dataframe(result)
        no_history = build_no_history_dataframe(operational, analysis.columns, as_of_date=as_of_date)
        if not no_history.empty:
            analysis = pd.concat([analysis, no_history], ignore_index=True)
            print(f"Added {len(no_history):,} COLLECT_MORE_DATA decisions")

        shap_df = calculate_top_shap(predictor, result, top_n=top_shap)
        if persist:
            analysis_rows, shap_rows = write_to_postgres(
                engine, analysis, shap_df, analysis_table, shap_table
            )
            print(f"Saved {analysis_rows:,} analysis rows and {shap_rows:,} SHAP rows")
        else:
            print("Dry run: no database writes")

        print("\nLatest action counts:")
        print(analysis["action"].value_counts(dropna=False).to_string())
        return analysis
    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="FleetCare latest prediction per vehicle")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--username", default=os.getenv("PGUSER", "tianqin-gg"))
    parser.add_argument("--password", default=None, help="Uses PGPASSWORD when omitted")
    parser.add_argument("--host", default=os.getenv("PGHOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--database", default=os.getenv("PGDATABASE", "fleetcare"))
    parser.add_argument("--analysis-table", default="analysis")
    parser.add_argument("--shap-table", default="prediction_shap")
    parser.add_argument("--top-shap", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--as-of-date", default=None, help="Prediction batch date YYYY-MM-DD")
    args = parser.parse_args()
    return run_fleetcare_prediction(
        model_dir=args.model_dir,
        username=args.username,
        password=args.password,
        host=args.host,
        port=args.port,
        database=args.database,
        analysis_table=args.analysis_table,
        shap_table=args.shap_table,
        top_shap=args.top_shap,
        persist=not args.dry_run,
        as_of_date=args.as_of_date,
    )


if __name__ == "__main__":
    main()
