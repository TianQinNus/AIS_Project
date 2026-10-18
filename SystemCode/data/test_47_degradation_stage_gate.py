"""Test 47: stage-gated C4 ranking using the frozen Train+Validation package.

Validation selected a simple 70% C4 rank plus 30% Risk48 stage-entry rank.
Risk48 is the five-class model probability that maintenance is within 48 time
units. The geometric blend is fixed before this script scores Test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fleetcare_c4_risk_inference import (
    MODEL_FILE,
    build_matrix,
    ensemble_c4_risk_score,
    last_rows,
    load_csv,
    percentile_rank,
    ranking_metrics,
)


BASE_WEIGHT = 0.70


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here / "raw")
    parser.add_argument(
        "--model-dir", type=Path,
        default=here / "processed" / "test_43_refit_train_validation",
    )
    parser.add_argument(
        "--current-model-dir", type=Path,
        default=here / "processed" / "fleetcare_c4_risk_model",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=here / "processed" / "test_47_degradation_stage_gate",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((args.model_dir / "metadata.json").read_text())
    models = joblib.load(args.model_dir / MODEL_FILE)
    test_op = load_csv(args.data_dir, "test_operational_readouts.csv").sort_values(
        ["vehicle_id", "time_step"], kind="stable"
    ).reset_index(drop=True)
    test_spec = load_csv(args.data_dir, "test_specifications.csv")
    test_labels = load_csv(args.data_dir, "test_labels.csv")
    selected = last_rows(test_op, test_labels)
    x_test, names = build_matrix(
        test_op, selected, test_spec, metadata["spec_columns"],
        metadata["raw_columns"],
    )
    if names != metadata["feature_names"]:
        raise ValueError("Test feature schema differs")

    c4_rank, _, _ = ensemble_c4_risk_score(x_test, models)
    probabilities = models["horizon_classifier"].predict_proba(x_test)
    risk48_probability = probabilities[:, 1:].sum(axis=1)
    stage_rank = percentile_rank(risk48_probability)
    final_score = np.power(np.clip(c4_rank, 1e-6, 1), BASE_WEIGHT) * np.power(
        np.clip(stage_rank, 1e-6, 1), 1 - BASE_WEIGHT
    )
    y = (selected["class_label"].to_numpy(int) == 4).astype(np.int8)
    report = ranking_metrics(y, final_score)
    current = json.loads(
        (args.current_model_dir / "evaluation_results.json").read_text()
    )["test"]
    test43 = json.loads(
        (args.model_dir / "comparison.json").read_text()
    )["candidate_train_plus_validation"]
    result = {
        "selection_protocol": (
            "Validation selected a simple geometric blend of 70% C4 rank and "
            "30% Risk48 stage-entry rank."
        ),
        "current_train_only": current,
        "test43_c4_rank": test43,
        "test47_stage_gated_rank": report,
        "selected_for_deployment": False,
        "test_note": (
            "Test did not select the gate or weight in this experiment, but "
            "had been inspected in earlier project development."
        ),
    }
    predictions = pd.DataFrame({
        "vehicle_id": selected["vehicle_id"].to_numpy(),
        "actual_class": selected["class_label"].to_numpy(int),
        "c4_rank": c4_rank,
        "risk48_probability": risk48_probability,
        "risk48_stage_rank": stage_rank,
        "c4_risk_score": final_score,
        "ranking": pd.Series(final_score).rank(
            ascending=False, method="first"
        ).astype(int),
    }).sort_values("ranking")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "model_name": "FleetCare C4 Degradation-Stage Gate Candidate",
            "source_model": str(args.model_dir / MODEL_FILE),
            "c4_rank_weight": BASE_WEIGHT,
            "risk48_stage_rank_weight": 1 - BASE_WEIGHT,
            "formula": "c4_rank^0.70 * percentile_rank(P(within 48))^0.30",
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
