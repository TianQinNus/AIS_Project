"""Test 60: validate the temporal rule engine on Test54 C4 rankings.

Test54 is trained on Train only for Validation histories.  The existing frozen
rule profile is evaluated unchanged, then alternative profiles are selected on
Validation.  Only after selection are the Train+Validation Test54 and horizon
models used to score Test histories.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fleetcare_c4_risk_inference import (
    build_matrix, fit_spec_schema, load_csv, percentile_rank,
)
from test_40_rule_engine_search import (
    apply_rules, evaluate, make_history_rows, profiles,
)
from test_52_endpoint_aligned_model import read_operational, train_endpoints
from test_54_endpoint_feature_parameter_search import fit_pair


FROZEN_PROFILE = {
    "name": "proposed_pers0.8_fast85_rise2",
    "p6": 99.0,
    "p12": 97.0,
    "inspect6": 97.0,
    "p24": 90.0,
    "p48": 80.0,
    "persistence": 0.8,
    "fast_percentile": 85.0,
    "min_rising": 2,
    "review_near": 97.0,
    "volatility_percentile": 95.0,
    "uncertainty_percentile": 90.0,
}


def grouped_rank(values: np.ndarray, group: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"value": values, "group": group})
    return frame.groupby("group", sort=False)["value"].rank(pct=True).to_numpy(float)


def score_history(op, spec, labels, schema, raw_columns, feature_names,
                  c4_models, horizon_model):
    selected = make_history_rows(op)
    x, names = build_matrix(op, selected, spec, schema, raw_columns)
    if names != feature_names:
        raise ValueError("History feature schema differs from Test54")
    step = selected["step_from_latest"].to_numpy()
    xgb_rank = grouped_rank(c4_models["xgb"].predict_proba(x)[:, 1], step)
    lgb_rank = grouped_rank(c4_models["lgb"].predict_proba(x)[:, 1], step)
    risk6 = grouped_rank((xgb_rank + lgb_rank) / 2.0, step)
    probability = horizon_model.predict_proba(x)
    risk12 = np.maximum(
        grouped_rank(probability[:, 3] + probability[:, 4], step), risk6
    )
    risk24 = np.maximum(
        grouped_rank(probability[:, 2:5].sum(axis=1), step), risk12
    )
    risk48 = np.maximum(grouped_rank(1.0 - probability[:, 0], step), risk24)
    latest_class = labels.set_index("vehicle_id")["class_label"]
    selected["class_label"] = selected["vehicle_id"].map(latest_class).astype(int)
    # temporal_summary uses these two columns to construct an independent
    # model-disagreement flag.  Here they are the two Test54 components.
    selected["base_rank"] = xgb_rank
    selected["veto_rank"] = lgb_rank
    selected["Risk6"] = risk6
    selected["Risk12"] = risk12
    selected["Risk24"] = risk24
    selected["Risk48"] = risk48
    return selected


def report_by_rule(decisions):
    result = decisions.groupby(["rule_id", "action"], dropna=False).agg(
        vehicles=("vehicle_id", "size"),
        c4=("class_label", lambda values: int((values == 4).sum())),
    ).reset_index()
    result["c4_rate"] = result["c4"] / result["vehicles"]
    return result


def main():
    root = Path(__file__).resolve().parent
    raw = root / "raw"
    out = root / "processed" / "test_60_test54_rule_engine"
    out.mkdir(parents=True, exist_ok=True)

    train_op = read_operational(raw, "train")
    validation_op = read_operational(raw, "validation")
    train_spec = load_csv(raw, "train_specifications.csv")
    validation_spec = load_csv(raw, "validation_specifications.csv")
    validation_labels = load_csv(raw, "validation_labels.csv")
    tte = load_csv(raw, "train_tte.csv")
    raw_columns = [c for c in train_op if c not in ("vehicle_id", "time_step")]
    schema = fit_spec_schema(train_spec)
    endpoints = train_endpoints(train_op, tte)
    x_train, feature_names = build_matrix(
        train_op, endpoints, train_spec, schema, raw_columns
    )
    print("Training Train-only Test54 for Validation...", flush=True)
    validation_c4_models = fit_pair(
        x_train, endpoints["class_label"].to_numpy(int), "FLEXIBLE"
    )
    validation_horizon_model = joblib.load(
        root / "processed" / "test_40_rule_engine_search" /
        "auxiliary_multiclass.joblib"
    )
    validation_history = score_history(
        validation_op, validation_spec, validation_labels, schema, raw_columns,
        feature_names, validation_c4_models, validation_horizon_model,
    )
    validation_history.to_csv(out / "validation_risk_history.csv", index=False)

    frozen_validation_decisions = apply_rules(validation_history, FROZEN_PROFILE)
    frozen_validation = evaluate(frozen_validation_decisions)
    frozen_validation_decisions.to_csv(
        out / "validation_frozen_decisions.csv", index=False
    )

    rows, decisions = [], {}
    for profile in profiles():
        current = apply_rules(validation_history, profile)
        metric = evaluate(current)
        rows.append({"profile": profile["name"], **metric})
        decisions[profile["name"]] = current
    search = pd.DataFrame(rows)
    eligible = search[
        search["immediate_capacity"].between(0.01, 0.05, inclusive="both")
        & (search["attention_total"] / search["vehicles"] <= 0.30)
    ].copy()
    if eligible.empty:
        raise RuntimeError("No rule profile met workload constraints")
    eligible = eligible.sort_values(
        ["attention_gain_same_n", "immediate_gain_same_n", "attention_c4",
         "immediate_c4", "attention_precision"],
        ascending=False,
    )
    winner_name = str(eligible.iloc[0]["profile"])
    winner_profile = next(p for p in profiles() if p["name"] == winner_name)
    winner_validation = eligible.iloc[0].to_dict()
    search.to_csv(out / "validation_rule_search.csv", index=False)
    decisions[winner_name].to_csv(out / "validation_winner_decisions.csv", index=False)
    print("Frozen Validation:", json.dumps(frozen_validation, indent=2), flush=True)
    print("Validation-selected profile:", winner_name, flush=True)
    print(eligible.head(12).to_string(index=False), flush=True)

    # Test remains untouched until the profile above has been frozen.
    test_op = read_operational(raw, "test")
    test_spec = load_csv(raw, "test_specifications.csv")
    test_labels = load_csv(raw, "test_labels.csv")
    final_test54 = joblib.load(
        root / "processed" / "test_54_endpoint_feature_parameter" /
        "fleetcare_c4_endpoint_tuned_candidate.joblib"
    )["models"]
    final_horizon = joblib.load(
        root / "processed" / "test_43_refit_train_validation" /
        "fleetcare_c4_risk_ensemble.joblib"
    )["horizon_classifier"]
    test_history = score_history(
        test_op, test_spec, test_labels, schema, raw_columns, feature_names,
        final_test54, final_horizon,
    )
    test_history.to_csv(out / "test_risk_history.csv", index=False)
    frozen_test_decisions = apply_rules(test_history, FROZEN_PROFILE)
    selected_test_decisions = apply_rules(test_history, winner_profile)
    frozen_test = evaluate(frozen_test_decisions)
    selected_test = evaluate(selected_test_decisions)
    frozen_test_decisions.to_csv(out / "test_frozen_decisions.csv", index=False)
    selected_test_decisions.to_csv(out / "test_selected_decisions.csv", index=False)
    report_by_rule(frozen_test_decisions).to_csv(
        out / "test_frozen_by_rule.csv", index=False
    )
    report_by_rule(selected_test_decisions).to_csv(
        out / "test_selected_by_rule.csv", index=False
    )
    result = {
        "test54_integration": (
            "Test54 XGB/LGB mean rank supplies Risk6 percentile; the existing "
            "five-class horizon model supplies Risk12/24/48; up to five recent "
            "readouts supply persistence, trends, volatility, and disagreement."
        ),
        "frozen_existing_profile": FROZEN_PROFILE,
        "frozen_existing_validation": frozen_validation,
        "validation_selected_profile": winner_profile,
        "validation_selected_metrics": winner_validation,
        "frozen_existing_test": frozen_test,
        "validation_selected_test": selected_test,
        "selection_rule": (
            "Validation only; immediate workload 1%-5%, attention workload <=30%; "
            "maximize attention C4 gain over Test54 at the same N, then immediate "
            "gain, C4 counts, and precision."
        ),
        "test_note": (
            "The profile was frozen before Test scoring, but official Test is not "
            "a pristine holdout because earlier project experiments inspected it."
        ),
    }
    (out / "results.json").write_text(
        json.dumps(result, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
