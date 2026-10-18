"""Complete standalone FleetCare production inference for the latest vehicle readout."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sqlalchemy import MetaData, Table, create_engine, delete, func, insert, select, text
from sqlalchemy.engine import URL
from xgboost import XGBClassifier

SGT = ZoneInfo("Asia/Singapore")
HORIZONS = (48, 24, 12, 6)
PRIORITY = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4, "REVIEW": 2}
RULE_PROFILE = {
    "p6": 99.0, "p12": 97.0, "inspect6": 97.0,
    "p24": 90.0, "p48": 80.0, "persistence": 0.80,
    "fast_percentile": 85.0, "min_rising": 2,
    "review_near": 97.0, "volatility_percentile": 95.0,
    "uncertainty_percentile": 90.0,
}
MODEL_FILE = "fleetcare_c4_risk_ensemble.joblib"
MODEL_COMPONENTS = (
    "c4_xgboost",
    "c4_lightgbm",
    "c4_veto",
    "horizon_classifier",
)


def load_csv(folder: Path, name: str) -> pd.DataFrame:
    path = Path(folder) / name
    frame = pd.read_csv(path)
    if len(frame.columns) == 1:
        frame = pd.read_csv(path, sep=";")
    return frame


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


def fit_spec_schema(specifications: pd.DataFrame) -> list[str]:
    features = specifications.drop(columns="vehicle_id")
    categorical = [column for column in features
                   if not pd.api.types.is_numeric_dtype(features[column])]
    return list(pd.get_dummies(features, columns=categorical, dtype=np.float32).columns)


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


def tte_label(remaining: np.ndarray) -> np.ndarray:
    return np.select([remaining <= 6, remaining <= 12, remaining <= 24, remaining <= 48],
                     [4, 3, 2, 1], default=0).astype(np.int8)


def training_pool(op: pd.DataFrame, tte: pd.DataFrame) -> pd.DataFrame:
    merged = op[["vehicle_id", "time_step"]].merge(
        tte, on="vehicle_id", how="left", validate="many_to_one")
    remaining = (merged["length_of_study_time_step"] - merged["time_step"]).to_numpy()
    repaired = merged["in_study_repair"].to_numpy() == 1
    eligible = (repaired & (remaining >= 0)) | (~repaired & (remaining > 48))
    return pd.DataFrame({
        "row_id": np.flatnonzero(eligible),
        "vehicle_id": merged["vehicle_id"].to_numpy()[eligible],
        "time_step": merged["time_step"].to_numpy()[eligible],
        "is_repair": repaired[eligible],
        "class_label": np.where(repaired[eligible], tte_label(remaining[eligible]), 0),
    })


def _spaced(group: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    ordered = group.sort_values("time_step", kind="stable")
    if maximum is None or len(ordered) <= maximum:
        return ordered.copy()
    positions = np.unique(np.rint(np.linspace(0, len(ordered) - 1, maximum)).astype(int))
    return ordered.iloc[positions].copy()


def select_training(pool: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chunks = []
    repaired = pool[pool["is_repair"]]
    for (_, label), group in repaired.groupby(["vehicle_id", "class_label"], sort=False):
        chosen = _spaced(group, 10 if int(label) == 4 else 3)
        chosen["sw"] = 1.0 / len(chosen)
        chunks.append(chosen)
    for _, group in pool[~pool["is_repair"]].groupby("vehicle_id", sort=False):
        size = min(2, len(group))
        chosen = group.iloc[rng.choice(len(group), size=size, replace=False)].copy()
        chosen["sw"] = 1.0 / len(chosen)
        chunks.append(chosen)
    return pd.concat(chunks, ignore_index=True).sample(
        frac=1, random_state=seed).reset_index(drop=True)


def last_rows(op: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows = op[["vehicle_id", "time_step"]].copy()
    rows["row_id"] = np.arange(len(rows), dtype=int)
    rows = rows.groupby("vehicle_id", sort=False).tail(1)
    rows = rows.merge(labels[["vehicle_id", "class_label"]], on="vehicle_id",
                      how="left", validate="one_to_one")
    if rows["class_label"].isna().any():
        raise ValueError("Missing labels for last readout")
    return rows.reset_index(drop=True)


def class_weights(labels: np.ndarray, base: np.ndarray, exponent: float) -> np.ndarray:
    negative = float(base[labels == 0].sum())
    positive = float(base[labels == 1].sum())
    multiplier = (negative / max(positive, 1e-12)) ** exponent
    return base * np.where(labels == 1, multiplier, 1.0)


def multiclass_weights(labels: np.ndarray, base: np.ndarray, exponent: float = 0.25):
    totals = np.asarray([base[labels == value].sum() for value in range(5)], dtype=float)
    multiplier = np.power(totals.max() / np.maximum(totals, 1e-12), exponent)
    return base * multiplier[labels]


def make_xgb_c4(seed: int = 42):
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        n_estimators=300, learning_rate=0.04, max_depth=5, min_child_weight=5,
        gamma=0.05, subsample=0.8, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=3.0, random_state=seed, n_jobs=4)


def make_lgb_c4(seed: int = 42):
    return LGBMClassifier(
        objective="binary", n_estimators=400, learning_rate=0.035,
        num_leaves=15, min_child_samples=35, colsample_bytree=0.8,
        subsample=0.8, subsample_freq=1, reg_lambda=2.0,
        verbosity=-1, random_state=seed, n_jobs=4)


def make_veto(seed: int = 61):
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        n_estimators=350, learning_rate=0.03, max_depth=3, min_child_weight=10,
        gamma=0.1, subsample=0.85, colsample_bytree=0.7,
        reg_alpha=0.2, reg_lambda=6.0, random_state=seed, n_jobs=4)


def make_multiclass(seed: int = 42):
    return XGBClassifier(
        objective="multi:softprob", num_class=5, eval_metric="mlogloss",
        n_estimators=350, learning_rate=0.035, max_depth=5, min_child_weight=8,
        gamma=0.1, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.2, reg_lambda=5.0, tree_method="hist",
        random_state=seed, n_jobs=4)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def ensemble_c4_risk_score(x: np.ndarray, models: dict):
    xgb_raw = models["c4_xgboost"].predict_proba(x)[:, 1]
    lgb_raw = models["c4_lightgbm"].predict_proba(x)[:, 1]
    veto_raw = models["c4_veto"].predict_proba(x)[:, 1]
    pair = (percentile_rank(xgb_raw) + percentile_rank(lgb_raw)) / 2
    base_rank = percentile_rank(pair)
    veto_rank = percentile_rank(veto_raw)
    return np.minimum(base_rank, veto_rank), base_rank, veto_rank


def ranking_metrics(labels: np.ndarray, score: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int8)
    order = np.argsort(-score, kind="stable")
    result = {
        "c4_average_precision": float(average_precision_score(labels, score)),
        "c4_roc_auc": float(roc_auc_score(labels, score)),
        "vehicles": int(len(labels)), "c4_total": int(labels.sum()),
    }
    for percent in (1, 2, 5, 10, 20):
        count = max(1, int(np.ceil(len(labels) * percent / 100)))
        found = int(labels[order[:count]].sum())
        result[f"top{percent}_vehicles"] = count
        result[f"top{percent}_c4_found"] = found
        result[f"precision_at_{percent}pct"] = found / count
        result[f"recall_at_{percent}pct"] = found / max(int(labels.sum()), 1)
    return result


def load_package(model_dir: Path):
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    model_path = model_dir / MODEL_FILE
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing FleetCare C4 risk model: {model_path}")
    models = joblib.load(model_path)
    if not isinstance(models, dict):
        raise TypeError(f"{MODEL_FILE} must contain a model-component dictionary")
    missing = [name for name in MODEL_COMPONENTS if name not in models]
    extra = [name for name in models if name not in MODEL_COMPONENTS]
    if missing or extra:
        raise ValueError(f"Invalid {MODEL_FILE}: missing={missing}, unexpected={extra}")
    return models, metadata


def make_engine(args):
    kwargs = {"username": args.username, "database": args.database}
    if args.password is not None:
        kwargs["password"] = args.password
    if args.host:
        kwargs.update(host=args.host, port=args.port)
    return create_engine(URL.create("postgresql+psycopg2", **kwargs), pool_pre_ping=True)


def load_active_fleet(engine, metadata):
    raw_columns = metadata["raw_columns"]
    quoted_raw = ", ".join(f'r."{column}"' for column in raw_columns)
    op_query = text(f"""
        SELECT r.vehicle_id, r.time_step, {quoted_raw}
        FROM public.truck_readout r
        JOIN public.truck t USING (vehicle_id)
        WHERE t.is_active IS TRUE
          AND COALESCE(t.in_maintenance, FALSE) IS FALSE
        ORDER BY r.vehicle_id, r.time_step
    """)
    spec_columns = [column for column in metadata["spec_columns"] if not column.startswith("Spec_")]
    del spec_columns  # one-hot schema is reconstructed from the eight source specification fields.
    spec_query = text("""
        SELECT vehicle_id, spec_0 AS "Spec_0", spec_1 AS "Spec_1",
               spec_2 AS "Spec_2", spec_3 AS "Spec_3", spec_4 AS "Spec_4",
               spec_5 AS "Spec_5", spec_6 AS "Spec_6", spec_7 AS "Spec_7"
        FROM public.truck
        WHERE is_active IS TRUE
          AND COALESCE(in_maintenance, FALSE) IS FALSE
        ORDER BY vehicle_id
    """)
    with engine.connect() as connection:
        operational = pd.read_sql(op_query, connection)
        specifications = pd.read_sql(spec_query, connection)
    if specifications.empty:
        raise RuntimeError("No trucks satisfy is_active=TRUE and in_maintenance IS NOT TRUE")
    operational = operational.sort_values(["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
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


def score_snapshot(cut, specifications, models, metadata, snapshot_order, offset):
    selected = latest_rows(cut)
    all_ids = specifications["vehicle_id"].astype(int).tolist()
    if selected.empty:
        scored = pd.DataFrame(columns=["vehicle_id"])
    else:
        x, names = build_matrix(
            cut, selected, specifications, metadata["spec_columns"],
            metadata["raw_columns"], tuple(metadata["lags"]),
        )
        if names != metadata["feature_names"]:
            raise ValueError("Inference feature schema does not match the saved C4 risk model package")
        risk6, base_rank, veto_rank = ensemble_c4_risk_score(x, models)
        probabilities = models["horizon_classifier"].predict_proba(x)
        scored = pd.DataFrame({
            "vehicle_id": selected["vehicle_id"].astype(int),
            "time_step": selected["time_step"].astype(float),
            "state": [f"C{value}" for value in probabilities.argmax(axis=1)],
            "confidence_score": probabilities.max(axis=1),
            # Keep the Test39 ensemble rank internally for temporal rules.
            "risk6_score": risk6,
            # Persisted as analysis.risk6_score after the rules are evaluated so
            # the stored horizons remain nested probabilities for the UI.
            "risk6_probability": probabilities[:, 4],
            "risk12_score": probabilities[:, 3:].sum(axis=1),
            "risk24_score": probabilities[:, 2:].sum(axis=1),
            "risk48_score": probabilities[:, 1:].sum(axis=1),
            "base_rank": base_rank,
            "veto_rank": veto_rank,
            "model_disagreement": np.abs(base_rank - veto_rank),
            "row_id": selected["row_id"].astype(int),
        })
        for horizon in HORIZONS:
            scored[f"risk{horizon}_percentile"] = percentile_rank(
                scored[f"risk{horizon}_score"].to_numpy()) * 100
    scored = pd.DataFrame({"vehicle_id": all_ids}).merge(scored, on="vehicle_id", how="left")
    scored["snapshot_order"] = int(snapshot_order)
    scored["source_offset"] = offset
    scored["source_readout_index"] = scored["vehicle_id"].map(
        cut.groupby("vehicle_id").size()).fillna(0).astype(int)
    # Reassigned after all snapshots are concatenated: rules count prior risk
    # observations, not the number of raw telemetry rows.
    scored["history_count"] = 0
    missing = scored["state"].isna()
    scored.loc[missing, "state"] = "NO_DATA"
    scored.loc[missing, "confidence_score"] = np.nan
    return scored, cut, x if not selected.empty else None, names if not selected.empty else None


def add_temporal_evidence(history: pd.DataFrame, profile: dict) -> pd.DataFrame:
    data = history.sort_values(["vehicle_id", "snapshot_order"], kind="stable").copy()
    for horizon in HORIZONS:
        score = f"risk{horizon}_score"
        percentile = f"risk{horizon}_percentile"
        previous = data.groupby("vehicle_id", sort=False)[score].shift(1)
        data[f"risk{horizon}_trend"] = data[score] - previous
        data[f"is_risk{horizon}_rising"] = data[f"risk{horizon}_trend"] > 0
        cutoff = data.groupby("snapshot_order")[f"risk{horizon}_trend"].transform(
            lambda values: values.clip(lower=0).quantile(profile["fast_percentile"] / 100))
        data[f"fast_risk{horizon}_deterioration"] = (
            data[f"is_risk{horizon}_rising"] &
            (data[f"risk{horizon}_trend"] >= cutoff) &
            data[f"risk{horizon}_trend"].gt(0)
        )
        threshold = profile[{48: "p48", 24: "p24", 12: "p12", 6: "p6"}[horizon]]
        high = data[percentile].ge(threshold)
        rolling_fraction = high.groupby(data["vehicle_id"], sort=False).transform(
            lambda values: values.rolling(5, min_periods=1).mean())
        data[f"risk{horizon}_persistence"] = rolling_fraction
        data[f"is_risk{horizon}_persistent"] = (
            data.groupby("vehicle_id", sort=False).cumcount().ge(1) &
            rolling_fraction.ge(profile["persistence"])
        )
        positive = data[f"risk{horizon}_trend"].clip(lower=0)
        data[f"risk{horizon}_deterioration_percentile"] = positive.groupby(
            data["snapshot_order"]).rank(pct=True, method="average") * 100

    data["risk6_volatility"] = data.groupby("vehicle_id", sort=False)["risk6_score"].transform(
        lambda values: values.rolling(5, min_periods=3).std())
    data["volatility_percentile"] = data.groupby("snapshot_order")["risk6_volatility"].rank(
        pct=True, method="average") * 100
    data["uncertainty_score"] = 1 - data["confidence_score"]
    disagreement_pct = data.groupby("snapshot_order")["model_disagreement"].rank(
        pct=True, method="average") * 100
    uncertainty_pct = data.groupby("snapshot_order")["uncertainty_score"].rank(
        pct=True, method="average") * 100
    data["independent_uncertainty_flag"] = (
        disagreement_pct.ge(profile["uncertainty_percentile"]) &
        uncertainty_pct.ge(profile["uncertainty_percentile"])
    )
    return data


def apply_rule(row, profile):
    count = int(row["history_count"])
    if count < 2:
        return "TRD", "REVIEW", "COLLECT_MORE_DATA", "Fewer than two available risk observations."
    if count < 3 and (row["risk6_percentile"] >= profile["review_near"] or
                      row["risk12_percentile"] >= profile["review_near"]):
        return "TR0", "REVIEW", "MANUAL_REVIEW", "High near-term rank with fewer than three observations."
    if (row["volatility_percentile"] >= profile["volatility_percentile"] and
            bool(row["independent_uncertainty_flag"])):
        return "TRV", "REVIEW", "MANUAL_REVIEW", "High volatility and an independent uncertainty flag."
    if (row["risk6_percentile"] >= profile["p6"] and
            (row["is_risk6_persistent"] or row["fast_risk6_deterioration"] or
             row["fast_risk12_deterioration"])):
        return "TR1", "CRITICAL", "PRIORITY_INSPECT", "Extreme Risk6 with persistence or fast deterioration."
    if (row["risk12_percentile"] >= profile["p12"] and
            (row["is_risk12_persistent"] or row["is_risk12_rising"] or
             row["fast_risk6_deterioration"])):
        return "TR2", "HIGH", "INSPECT", "High Risk12 with supporting temporal evidence."
    rising = sum(bool(row[f"is_risk{horizon}_rising"]) for horizon in HORIZONS)
    if row["risk6_percentile"] >= profile["inspect6"] and rising >= profile["min_rising"]:
        return "TR3", "HIGH", "INSPECT", "High Risk6 with at least two rising horizons."
    if (row["risk24_percentile"] >= profile["p24"] and
            (row["is_risk24_persistent"] or row["is_risk24_rising"])):
        return "TR4", "MEDIUM", "ENHANCED_MONITOR", "High Risk24 with persistence or upward movement."
    if (row["risk48_percentile"] >= profile["p48"] and
            (row["is_risk48_persistent"] or row["is_risk48_rising"])):
        return "TR5", "LOW", "MONITOR", "High Risk48 with persistence or upward movement."
    return "TR6", "NORMAL", "CONTINUE", "No escalation condition was met."


def finish_analysis(history, profile):
    data = add_temporal_evidence(history, profile)
    decisions = data.apply(lambda row: apply_rule(row, profile), axis=1, result_type="expand")
    decisions.columns = ["rule_id", "priority", "action", "reason"]
    data = pd.concat([data, decisions], axis=1)
    data["history_quality"] = np.select(
        [data["history_count"].lt(2), data["history_count"].lt(3)],
        ["INSUFFICIENT", "LIMITED"], default="ADEQUATE")
    data["trend_direction"] = np.select(
        [data["risk6_trend"].gt(0), data["risk6_trend"].lt(0)],
        ["RISING", "FALLING"], default="STABLE")
    data["class_worsening"] = data.groupby("vehicle_id")["state"].transform(
        lambda values: pd.to_numeric(values.str.extract(r"(\d+)")[0], errors="coerce").diff().gt(0))
    data["hybrid_ranking_score"] = data["risk6_score"]
    data["priority_score"] = data["priority"].map(PRIORITY) * 100 + data["risk6_percentile"].fillna(0)
    data["ranking"] = data.groupby("snapshot_order")["priority_score"].rank(
        ascending=False, method="first").astype("Int64")
    data["evidence"] = data.apply(
        lambda row: (f"Risk6 P{row.risk6_percentile:.1f}; Risk12 P{row.risk12_percentile:.1f}; "
                     f"Risk24 P{row.risk24_percentile:.1f}; history={int(row.history_count)}"), axis=1)
    data["rule_trace"] = data.apply(
        lambda row: (f"{row.rule_id} | C4_RISK_ENSEMBLE | source=latest_readout | "
                     f"history={int(row.history_count)}/5 | "
                     f"p6={row.risk6_percentile:.1f} p12={row.risk12_percentile:.1f} "
                     f"p24={row.risk24_percentile:.1f} p48={row.risk48_percentile:.1f} | {row.action}"), axis=1)
    return data


def top_shap_rows(snapshot_payloads, analysis, models, metadata, top_n):
    output = []
    booster = models["c4_xgboost"].get_booster()
    for prediction_at, scored, matrix, names in snapshot_payloads:
        if matrix is None or len(scored) == 0:
            continue
        contributions = booster.predict(__import__("xgboost").DMatrix(matrix), pred_contribs=True)
        for index, vehicle_id in enumerate(scored["vehicle_id"].astype(int)):
            top = np.argsort(np.abs(contributions[index, :-1]))[-top_n:][::-1]
            for rank, feature_index in enumerate(top, 1):
                value = float(matrix[index, feature_index])
                output.append({
                    "vehicle_id": vehicle_id, "prediction_at": prediction_at,
                    "feature_name": names[feature_index],
                    "feature_value": value if np.isfinite(value) else None,
                    "shap_value": float(contributions[index, feature_index]),
                    "contribution_rank": rank,
                })
    return pd.DataFrame(output)


def write_postgres(engine, analysis, shap_rows, replace_existing=False,
                   analysis_table="analysis", shap_table="prediction_shap"):
    metadata = MetaData()
    target = Table(analysis_table, metadata, schema="public", autoload_with=engine)
    shap_target = Table(shap_table, metadata, schema="public", autoload_with=engine)
    dates = sorted(set(pd.to_datetime(analysis["prediction_at"]).dt.date))
    vehicles = sorted(analysis["vehicle_id"].astype(int).unique().tolist())
    database_columns = [column for column in analysis.columns
                        if column in target.c and column != "analysis_id"]
    id_map = {}
    with engine.begin() as connection:
        existing_query = select(target.c.analysis_id).where(
            target.c.vehicle_id.in_(vehicles),
            func.date(func.timezone("Asia/Singapore", target.c.prediction_at)).in_(dates),
        )
        existing = [row[0] for row in connection.execute(existing_query)]
        if existing and not replace_existing:
            raise RuntimeError(
                f"Found {len(existing)} existing rows on target dates. Add --replace-existing to replace only this batch.")
        if existing:
            connection.execute(delete(target).where(target.c.analysis_id.in_(existing)))
        records = analysis[database_columns].astype(object).where(pd.notna(analysis[database_columns]), None)
        for start in range(0, len(records), 500):
            inserted = connection.execute(
                insert(target).returning(target.c.analysis_id, target.c.vehicle_id, target.c.prediction_at),
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
                raise RuntimeError(f"No inserted analysis row for SHAP key {key}")
            shaped.append({
                "analysis_id": id_map[key], "feature_name": row.feature_name,
                "feature_value": row.feature_value, "shap_value": row.shap_value,
                "contribution_rank": int(row.contribution_rank),
            })
        for start in range(0, len(shaped), 1000):
            connection.execute(insert(shap_target), shaped[start:start + 1000])
    return len(analysis), len(shaped), len(existing)


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=("FleetCare production inference: score the latest database readout "
                     "using up to five recent readouts per vehicle"))
    parser.add_argument("--model-dir", type=Path,
                        default=script_dir / "fleetcare_c4_risk_model")
    parser.add_argument("--output-dir", type=Path,
                        default=script_dir / "fleetcare_c4_risk_inference")
    parser.add_argument(
        "--prediction-at", default=None,
        help="ISO timestamp of the newest output; defaults to current Singapore time",
    )
    parser.add_argument(
        "--history-days", type=int, default=1,
        help="Number of consecutive daily outputs ending at --prediction-at (default: 1)",
    )
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
    if args.top_shap < 1:
        parser.error("--top-shap must be positive")
    if args.history_days < 1:
        parser.error("--history-days must be positive")
    if args.replace_existing and not args.write_db:
        parser.error("--replace-existing requires --write-db")

    prediction_at = (pd.Timestamp(args.prediction_at)
                     if args.prediction_at else pd.Timestamp.now(tz=SGT))
    if prediction_at.tzinfo is None:
        prediction_at = prediction_at.tz_localize(SGT)
    else:
        prediction_at = prediction_at.tz_convert(SGT)
    engine = make_engine(args)
    try:
        models, metadata = load_package(args.model_dir)
        operational, specifications = load_active_fleet(engine, metadata)
        print(f"Eligible fleet: {len(specifications):,} active, non-maintenance vehicles; "
              f"{len(operational):,} readouts", flush=True)
        frames = []
        output_payloads = []
        # Five temporal observations are required for the earliest requested
        # output. For D output days this needs offsets D+3 ... 0.
        maximum_offset = args.history_days + 3
        for snapshot_order, offset in enumerate(range(maximum_offset, -1, -1), start=1):
            cut = snapshot_cut(operational, offset)
            scored, _, matrix, names = score_snapshot(
                cut, specifications, models, metadata, snapshot_order, offset)
            frames.append(scored)
            valid = scored[scored["row_id"].notna()].sort_values(
                "row_id", kind="stable").reset_index(drop=True)
            if offset < args.history_days:
                output_at = prediction_at - pd.Timedelta(days=offset)
                output_payloads.append((output_at, valid, matrix, names))
            label = "latest" if offset == 0 else f"latest-{offset} readouts"
            print(f"History snapshot {snapshot_order}/{maximum_offset + 1} ({label}): "
                  f"scored={len(valid):,}; unavailable={len(scored)-len(valid):,}", flush=True)
        raw_history = pd.concat(frames, ignore_index=True)
        raw_history = raw_history.sort_values(
            ["vehicle_id", "snapshot_order"], kind="stable").reset_index(drop=True)
        valid_risk = raw_history["state"].ne("NO_DATA").astype(int)
        raw_history["history_count"] = valid_risk.groupby(
            raw_history["vehicle_id"], sort=False).cumsum()
        profile = metadata["rule_profile"]
        history_with_decisions = finish_analysis(raw_history, profile)
        analysis = history_with_decisions.loc[
            history_with_decisions["source_offset"].lt(args.history_days)
        ].copy().reset_index(drop=True)
        # The rule engine above intentionally uses the Test39 ensemble rank.
        # The database/API contract uses risk6_score as the true P(C4), while
        # risk6_percentile retains the ensemble-based fleet priority percentile.
        analysis["risk6_score"] = analysis.pop("risk6_probability")
        analysis["prediction_at"] = analysis["source_offset"].map(
            lambda offset: prediction_at - pd.Timedelta(days=int(offset)))
        shap_rows = top_shap_rows(
            output_payloads,
            analysis, models, metadata, args.top_shap,
        )
        analysis = analysis.drop(columns=["row_id", "base_rank", "veto_rank", "model_disagreement",
                                          "uncertainty_score", "volatility_percentile",
                                          "independent_uncertainty_flag", "risk6_volatility",
                                          "snapshot_order"], errors="ignore")
        # args.output_dir.mkdir(parents=True, exist_ok=True)
        # analysis.to_csv(args.output_dir / "analysis_preview.csv", index=False)
        # shap_rows.to_csv(args.output_dir / "prediction_shap_preview.csv", index=False)
        print(f"Final predictions: {len(analysis):,} rows across {args.history_days} day(s), "
              f"ending at {prediction_at.isoformat()}", flush=True)
        print("Action counts:\n" + analysis["action"].value_counts().to_string(), flush=True)
        if args.write_db:
            saved, saved_shap, replaced = write_postgres(
                engine, analysis, shap_rows, args.replace_existing,
                args.analysis_table, args.shap_table)
            print(f"Committed {saved:,} analysis rows and {saved_shap:,} SHAP rows; "
                  f"replaced {replaced:,} existing target rows.", flush=True)
        else:
            print("DRY RUN: database unchanged; add --write-db after reviewing previews.", flush=True)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
