"""Test 53: endpoint-aligned ensemble expansion.

Starting from Test 52, compare three complementary endpoint signals:
multi-seed tree ranks, explicit five-interval sequence trees and LambdaRank.
Only a small predeclared set of blends is considered. Validation selects one;
the selected components are then refit on Train plus Validation before Test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fleetcare_c4_risk_inference import (
    build_matrix,
    fit_spec_schema,
    last_rows,
    load_csv,
    percentile_rank,
    ranking_metrics,
)
from test_48_interval_sequence_search import sequence_matrix
from test_51_lambdarank_extension import fit_ranker as fit_lambdarank
from test_52_endpoint_aligned_model import (
    fit_pair,
    pair_rank,
    read_operational,
    train_endpoints,
)


SEEDS = (42, 77, 2026)
BLENDS = {
    "base": {"base": 1.0},
    "multiseed": {"multiseed": 1.0},
    "sequence": {"sequence": 1.0},
    "ranker": {"ranker": 1.0},
    "base75_sequence25": {"base": 0.75, "sequence": 0.25},
    "base75_ranker25": {"base": 0.75, "ranker": 0.25},
    "multiseed75_sequence25": {"multiseed": 0.75, "sequence": 0.25},
    "multiseed75_ranker25": {"multiseed": 0.75, "ranker": 0.25},
    "base60_sequence20_ranker20": {
        "base": 0.60, "sequence": 0.20, "ranker": 0.20,
    },
    "multi60_sequence20_ranker20": {
        "multiseed": 0.60, "sequence": 0.20, "ranker": 0.20,
    },
}


def geometric_blend(signals, weights):
    score = np.ones(len(next(iter(signals.values()))), dtype=float)
    for name, weight in weights.items():
        score *= np.power(np.clip(signals[name], 1e-6, 1.0), weight)
    return score


def fit_multiseed(x, labels):
    return [fit_pair(x, labels, 0.75, seed) for seed in SEEDS]


def multiseed_rank(x, packages):
    ranks = []
    for package in packages:
        ranks.append(percentile_rank(package["xgb"].predict_proba(x)[:, 1]))
        ranks.append(percentile_rank(package["lgb"].predict_proba(x)[:, 1]))
    return percentile_rank(np.vstack(ranks).mean(axis=0))


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here / "raw")
    parser.add_argument(
        "--output-dir", type=Path,
        default=here / "processed" / "test_53_endpoint_ensemble",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_op = read_operational(args.data_dir, "train")
    validation_op = read_operational(args.data_dir, "validation")
    train_spec = load_csv(args.data_dir, "train_specifications.csv")
    validation_spec = load_csv(args.data_dir, "validation_specifications.csv")
    tte = load_csv(args.data_dir, "train_tte.csv")
    validation_labels = load_csv(args.data_dir, "validation_labels.csv")
    raw_columns = [c for c in train_op if c not in ("vehicle_id", "time_step")]
    schema = fit_spec_schema(train_spec)
    endpoints = train_endpoints(train_op, tte)
    validation_selected = last_rows(validation_op, validation_labels)
    labels = endpoints["class_label"].to_numpy(int)
    y_validation = (
        validation_selected["class_label"].to_numpy(int) == 4
    ).astype(np.int8)

    print("Building endpoint base and interval-sequence matrices...", flush=True)
    x_base, base_names = build_matrix(
        train_op, endpoints, train_spec, schema, raw_columns
    )
    xv_base, validation_base_names = build_matrix(
        validation_op, validation_selected, validation_spec, schema, raw_columns
    )
    x_sequence, sequence_names = sequence_matrix(
        train_op, endpoints, train_spec, schema, raw_columns
    )
    xv_sequence, validation_sequence_names = sequence_matrix(
        validation_op, validation_selected, validation_spec, schema, raw_columns
    )
    if base_names != validation_base_names or sequence_names != validation_sequence_names:
        raise ValueError("Validation feature schema differs")

    print("Training endpoint base and multi-seed components...", flush=True)
    multiseed_models = fit_multiseed(x_base, labels)
    base_signal = pair_rank(xv_base, multiseed_models[0])
    multiseed_signal = multiseed_rank(xv_base, multiseed_models)

    print("Training endpoint interval-sequence component...", flush=True)
    sequence_models = fit_pair(x_sequence, labels, 0.75, 42)
    sequence_signal = pair_rank(xv_sequence, sequence_models)

    print("Training endpoint LambdaRank component...", flush=True)
    ranker = fit_lambdarank(
        x_base, labels, np.ones(len(labels)),
        endpoints["time_step"].to_numpy(float), "R31_W00",
    )
    ranker_signal = percentile_rank(ranker.predict(xv_base))
    signals = {
        "base": base_signal,
        "multiseed": multiseed_signal,
        "sequence": sequence_signal,
        "ranker": ranker_signal,
    }
    candidates = []
    for name, weights in BLENDS.items():
        report = ranking_metrics(y_validation, geometric_blend(signals, weights))
        candidates.append({"candidate": name, **report})
    table = pd.DataFrame(candidates).sort_values(
        ["top5_c4_found", "top2_c4_found", "c4_average_precision", "c4_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    table.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    winner = table.iloc[0].to_dict()
    winner_name = winner["candidate"]
    winner_weights = BLENDS[winner_name]
    print("Validation endpoint ensemble leaders:", flush=True)
    print(table.to_string(index=False), flush=True)

    # Refit only components needed by the frozen Validation winner.
    validation_selected["sw"] = 1.0
    fit_labels = np.concatenate([
        labels, validation_selected["class_label"].to_numpy(int)
    ])
    xfit_base = np.vstack([x_base, xv_base]).astype(np.float32)
    xfit_sequence = np.vstack([x_sequence, xv_sequence]).astype(np.float32)
    fit_time = np.concatenate([
        endpoints["time_step"].to_numpy(float),
        validation_selected["time_step"].to_numpy(float),
    ])
    final_components = {}
    print(f"Refitting Validation winner {winner_name} before Test...", flush=True)
    if "base" in winner_weights:
        final_components["base"] = fit_pair(xfit_base, fit_labels, 0.75, 42)
    if "multiseed" in winner_weights:
        final_components["multiseed"] = fit_multiseed(xfit_base, fit_labels)
    if "sequence" in winner_weights:
        final_components["sequence"] = fit_pair(
            xfit_sequence, fit_labels, 0.75, 42
        )
    if "ranker" in winner_weights:
        final_components["ranker"] = fit_lambdarank(
            xfit_base, fit_labels, np.ones(len(fit_labels)), fit_time, "R31_W00"
        )

    test_op = read_operational(args.data_dir, "test")
    test_spec = load_csv(args.data_dir, "test_specifications.csv")
    test_labels = load_csv(args.data_dir, "test_labels.csv")
    test_selected = last_rows(test_op, test_labels)
    xt_base, test_base_names = build_matrix(
        test_op, test_selected, test_spec, schema, raw_columns
    )
    if base_names != test_base_names:
        raise ValueError("Test base schema differs")
    xt_sequence = None
    if "sequence" in winner_weights:
        xt_sequence, test_sequence_names = sequence_matrix(
            test_op, test_selected, test_spec, schema, raw_columns
        )
        if sequence_names != test_sequence_names:
            raise ValueError("Test sequence schema differs")
    test_signals = {}
    if "base" in winner_weights:
        test_signals["base"] = pair_rank(xt_base, final_components["base"])
    if "multiseed" in winner_weights:
        test_signals["multiseed"] = multiseed_rank(
            xt_base, final_components["multiseed"]
        )
    if "sequence" in winner_weights:
        test_signals["sequence"] = pair_rank(
            xt_sequence, final_components["sequence"]
        )
    if "ranker" in winner_weights:
        test_signals["ranker"] = percentile_rank(
            final_components["ranker"].predict(xt_base)
        )
    final_score = geometric_blend(test_signals, winner_weights)
    y_test = (test_selected["class_label"].to_numpy(int) == 4).astype(np.int8)
    test_report = ranking_metrics(y_test, final_score)
    test52 = json.loads(
        (here / "processed" / "test_52_endpoint_aligned" / "results.json").read_text()
    )["test52_endpoint_aligned"]
    result = {
        "selection_rule": (
            "Validation lexicographic: Top-5 C4, Top-2 C4, AP, then AUC"
        ),
        "validation_winner": winner,
        "winner_weights": winner_weights,
        "test52_reference": test52,
        "test53_endpoint_ensemble": test_report,
        "selected_for_deployment": False,
        "test_note": (
            "Test was loaded only after Validation selection in this experiment, "
            "but had been inspected in earlier project work."
        ),
    }
    predictions = pd.DataFrame({
        "vehicle_id": test_selected["vehicle_id"].to_numpy(),
        "actual_class": test_selected["class_label"].to_numpy(int),
        "c4_risk_score": final_score,
        "ranking": pd.Series(final_score).rank(
            ascending=False, method="first"
        ).astype(int),
    })
    for name, values in test_signals.items():
        predictions[f"{name}_score"] = values
    predictions.sort_values("ranking").to_csv(
        args.output_dir / "test_predictions.csv", index=False
    )
    joblib.dump(
        {"components": final_components, "weights": winner_weights},
        args.output_dir / "fleetcare_c4_endpoint_ensemble_candidate.joblib",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "model_name": "FleetCare Endpoint Ensemble Candidate",
            "model_file": "fleetcare_c4_endpoint_ensemble_candidate.joblib",
            "base_features": len(base_names),
            "sequence_features": len(sequence_names),
            "raw_columns": raw_columns,
            "spec_columns": schema,
            "base_feature_names": base_names,
            "sequence_feature_names": sequence_names,
            "winner": winner,
            "winner_weights": winner_weights,
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
