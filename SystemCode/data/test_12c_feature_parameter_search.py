from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

import test_11c_generalization_robust_xgboost as test11


SCREENING_SEED = 42
FINAL_SEEDS = [42, 52, 62, 72, 82]
FINALIST_COUNT = 5

PARAMETER_SETS = {
    "P1_regularized": {
        "n_estimators": 300,
        "learning_rate": 0.04,
        "max_depth": 4,
        "min_child_weight": 8,
        "gamma": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.65,
        "reg_alpha": 0.2,
        "reg_lambda": 5.0,
    },
    "P2_shallow": {
        "n_estimators": 450,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 10,
        "gamma": 0.2,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 8.0,
    },
    "P3_medium": {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "gamma": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
    },
    "P4_very_shallow": {
        "n_estimators": 550,
        "learning_rate": 0.025,
        "max_depth": 2,
        "min_child_weight": 12,
        "gamma": 0.2,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 10.0,
    },
    "P5_less_regularized": {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 3,
        "gamma": 0.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.5,
    },
}


def feature_columns(frame, feature_set):
    specs = [c for c in frame.columns if c.startswith("Spec_")]
    common = [
        c
        for c in frame.columns
        if c in {"time_gap", "history_count"}
        or c.startswith("actual_gap_")
        or c.startswith("readout_count_")
        or c.startswith("history_available_")
    ]
    rates = [c for c in frame.columns if "__rate_" in c]
    changes = [c for c in frame.columns if "__change_" in c]

    if feature_set == "F1_rate_all":
        selected = specs + common + rates
    elif feature_set == "F2_rate_change_all":
        selected = specs + common + rates + changes
    elif feature_set == "F3_rate_short":
        selected = specs + [
            c
            for c in common + rates
            if c in {"time_gap", "history_count"}
            or c.endswith("_6")
            or c.endswith("_12")
        ]
    elif feature_set == "F4_rate_long":
        selected = specs + [
            c
            for c in common + rates
            if c in {"time_gap", "history_count"}
            or c.endswith("_24")
            or c.endswith("_48")
        ]
    elif feature_set == "F5_rate_plus_short_change":
        selected = specs + common + rates + [
            c for c in changes if c.endswith("_6") or c.endswith("_12")
        ]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    return list(dict.fromkeys(selected))


FEATURE_SETS = [
    "F1_rate_all",
    "F2_rate_change_all",
    "F3_rate_short",
    "F4_rate_long",
    "F5_rate_plus_short_change",
]


def train_model(x_train, y_train, parameter_name, seed):
    parameters = PARAMETER_SETS[parameter_name]
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **parameters,
    )
    model.fit(x_train, y_train)
    return model


def decision_metrics(labels, probabilities):
    argmax_prediction = probabilities.argmax(axis=1)
    result = {
        "Accuracy": accuracy_score(labels, argmax_prediction),
        "MacroF1": f1_score(
            labels,
            argmax_prediction,
            labels=test11.CLASS_IDS,
            average="macro",
            zero_division=0,
        ),
    }
    for class_id in test11.CLASS_IDS:
        result[f"PredC{class_id}"] = int((argmax_prediction == class_id).sum())
    return result


def ranking_metrics(labels, probabilities):
    result = {}
    horizon_ap = []
    horizon_auc = []
    for horizon, positive_classes in test11.HORIZONS.items():
        binary = np.isin(labels, positive_classes).astype(int)
        score = probabilities[:, positive_classes].sum(axis=1)
        result[f"{horizon}_AP"] = float(average_precision_score(binary, score))
        result[f"{horizon}_AUC"] = float(roc_auc_score(binary, score))
        horizon_ap.append(result[f"{horizon}_AP"])
        horizon_auc.append(result[f"{horizon}_AUC"])
        k = max(1, int(np.ceil(len(labels) * 0.05)))
        selected = np.argsort(-score, kind="stable")[:k]
        result[f"{horizon}_Top5Recall"] = float(binary[selected].sum() / binary.sum())
        result[f"{horizon}_Top5Precision"] = float(binary[selected].mean())
    result["MeanHorizonAP"] = float(np.mean(horizon_ap))
    result["MeanHorizonAUC"] = float(np.mean(horizon_auc))
    return result


def build_seed_data(candidates, official, target_times, train_tte, train_spec, seed):
    base_ids, cal_ids, policy_ids = test11.split_vehicle_ids(train_tte, seed)
    train_sample = test11.vehicle_aware_training_sample(
        candidates[candidates["vehicle_id"].isin(base_ids)], seed
    )
    cal_snapshot = test11.landmark_snapshot(
        candidates[candidates["vehicle_id"].isin(cal_ids)], target_times, seed + 1000
    )
    policy_snapshot = test11.landmark_snapshot(
        candidates[candidates["vehicle_id"].isin(policy_ids)], target_times, seed + 2000
    )
    train_sample = test11.add_specs(train_sample, train_spec)
    cal_snapshot = test11.add_specs(cal_snapshot, train_spec)
    policy_snapshot = test11.add_specs(policy_snapshot, train_spec)
    return train_sample, cal_snapshot, policy_snapshot, official


def transform_feature_set(seed_data, feature_set):
    train_sample, cal_snapshot, policy_snapshot, official = seed_data
    frames = [test11.feature_frame(frame) for frame in seed_data]
    columns = feature_columns(frames[0], feature_set)
    frames = [frame.reindex(columns=columns) for frame in frames]
    preprocessor = test11.make_preprocessor(frames[0])
    transformed = [preprocessor.fit_transform(frames[0])]
    transformed.extend(preprocessor.transform(frame) for frame in frames[1:])
    labels = [
        train_sample["class_label"].astype(int).to_numpy(),
        cal_snapshot["class_label"].astype(int).to_numpy(),
        policy_snapshot["class_label"].astype(int).to_numpy(),
        official["class_label"].astype(int).to_numpy(),
    ]
    return transformed, labels, len(columns), preprocessor


def evaluate_combination(
    transformed,
    labels,
    feature_set,
    parameter_name,
    seed,
    include_official,
):
    x_train, x_cal, x_policy, x_official = transformed
    y_train, y_cal, y_policy, y_official = labels
    model = train_model(x_train, y_train, parameter_name, seed)
    raw_cal = test11.normalize_probabilities(model.predict_proba(x_cal))
    calibrator = test11.fit_calibrator(raw_cal, y_cal, seed)

    rows = []
    prediction_sets = [
        ("PolicyDev", y_policy, model.predict_proba(x_policy)),
    ]
    if include_official:
        prediction_sets.append(
            ("OfficialValidation", y_official, model.predict_proba(x_official))
        )

    for split_name, split_labels, raw_probabilities in prediction_sets:
        probabilities = test11.calibrated_proba(calibrator, raw_probabilities)
        row = {
            "Seed": seed,
            "Split": split_name,
            "FeatureSet": feature_set,
            "ParameterSet": parameter_name,
        }
        row.update(decision_metrics(split_labels, probabilities))
        row.update(ranking_metrics(split_labels, probabilities))
        rows.append(row)

    del model, calibrator
    gc.collect()
    return rows


def prepare_all_data():
    data = test11.load_data()
    raw_features = [
        c for c in data["train_op"].columns if c not in {"vehicle_id", "time_step"}
    ]
    train_op = data.pop("train_op")
    train_features = test11.make_generalization_features(train_op, raw_features)
    del train_op
    gc.collect()
    val_op = data.pop("val_op")
    val_features = test11.make_generalization_features(val_op, raw_features)
    del val_op
    gc.collect()
    candidates = test11.build_labelled_candidates(train_features, data["train_tte"])
    official = test11.official_last_snapshot(
        val_features, data["val_spec"], data["val_labels"]
    )
    del train_features, val_features
    gc.collect()
    return data, candidates, official


def main():
    output_dir = Path(test11.OUTPUT_DIR)
    data, candidates, official = prepare_all_data()
    target_times = official["time_step"].to_numpy()

    screening_rows = []
    screening_seed_data = build_seed_data(
        candidates,
        official,
        target_times,
        data["train_tte"],
        data["train_spec"],
        SCREENING_SEED,
    )

    for feature_set in FEATURE_SETS:
        print(f"Screening {feature_set}")
        transformed, labels, feature_count, preprocessor = transform_feature_set(
            screening_seed_data, feature_set
        )
        for parameter_name in PARAMETER_SETS:
            print(f"  {parameter_name}")
            rows = evaluate_combination(
                transformed,
                labels,
                feature_set,
                parameter_name,
                SCREENING_SEED,
                include_official=False,
            )
            rows[0]["FeatureCount"] = feature_count
            rows[0]["Parameters"] = json.dumps(PARAMETER_SETS[parameter_name])
            screening_rows.extend(rows)
        del transformed, labels, preprocessor
        gc.collect()

    screening = pd.DataFrame(screening_rows).sort_values(
        ["MeanHorizonAP", "Risk6_AP", "MacroF1"],
        ascending=[False, False, False],
    )
    screening.to_csv(output_dir / "model_12c_screening_results.csv", index=False)
    finalists = screening.head(FINALIST_COUNT)[["FeatureSet", "ParameterSet"]].copy()
    finalists.to_csv(output_dir / "model_12c_finalists.csv", index=False)
    finalist_pairs = list(finalists.itertuples(index=False, name=None))

    final_rows = []
    for seed in FINAL_SEEDS:
        print(f"Final evaluation seed {seed}")
        seed_data = build_seed_data(
            candidates,
            official,
            target_times,
            data["train_tte"],
            data["train_spec"],
            seed,
        )
        for feature_set in sorted(set(pair[0] for pair in finalist_pairs)):
            transformed, labels, feature_count, preprocessor = transform_feature_set(
                seed_data, feature_set
            )
            parameter_names = [p for f, p in finalist_pairs if f == feature_set]
            for parameter_name in parameter_names:
                rows = evaluate_combination(
                    transformed,
                    labels,
                    feature_set,
                    parameter_name,
                    seed,
                    include_official=True,
                )
                for row in rows:
                    row["FeatureCount"] = feature_count
                    row["Parameters"] = json.dumps(PARAMETER_SETS[parameter_name])
                final_rows.extend(rows)
            del transformed, labels, preprocessor
            gc.collect()
        del seed_data
        gc.collect()

    results = pd.DataFrame(final_rows)
    results.to_csv(output_dir / "model_12c_finalist_results_by_seed.csv", index=False)
    metric_columns = [
        c
        for c in results.select_dtypes(include=[np.number]).columns
        if c not in {"Seed", "FeatureCount"}
    ]
    summary = (
        results.groupby(["Split", "FeatureSet", "ParameterSet"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(output_dir / "model_12c_finalist_summary.csv", index=False)

    policy = results[results["Split"].eq("PolicyDev")]
    policy_ranking = (
        policy.groupby(["FeatureSet", "ParameterSet"])
        .agg(
            MeanHorizonAPMean=("MeanHorizonAP", "mean"),
            MeanHorizonAPStd=("MeanHorizonAP", "std"),
            MeanHorizonAUCMean=("MeanHorizonAUC", "mean"),
            MacroF1Mean=("MacroF1", "mean"),
            Risk6APMean=("Risk6_AP", "mean"),
        )
        .sort_values(
            ["MeanHorizonAPMean", "Risk6APMean", "MacroF1Mean"],
            ascending=[False, False, False],
        )
        .reset_index()
    )
    classification_ranking = (
        policy.groupby(["FeatureSet", "ParameterSet"])
        .agg(
            MacroF1Mean=("MacroF1", "mean"),
            MacroF1Std=("MacroF1", "std"),
        )
        .sort_values(["MacroF1Mean", "MacroF1Std"], ascending=[False, True])
        .reset_index()
    )
    ranking_winner = policy_ranking.iloc[0]
    classification_winner = classification_ranking.iloc[0]
    winner_frame = pd.DataFrame(
        [
            {
                "Objective": "RiskRanking",
                "SelectedFeatureSet": ranking_winner["FeatureSet"],
                "SelectedParameterSet": ranking_winner["ParameterSet"],
                "PolicyScoreMean": ranking_winner["MeanHorizonAPMean"],
                "PolicyScoreStd": ranking_winner["MeanHorizonAPStd"],
                "SelectionRule": "highest mean PolicyDev AP across Risk48/24/12/6",
            },
            {
                "Objective": "FiveClassClassification",
                "SelectedFeatureSet": classification_winner["FeatureSet"],
                "SelectedParameterSet": classification_winner["ParameterSet"],
                "PolicyScoreMean": classification_winner["MacroF1Mean"],
                "PolicyScoreStd": classification_winner["MacroF1Std"],
                "SelectionRule": "highest mean PolicyDev MacroF1",
            },
        ]
    )
    winner_frame.to_csv(output_dir / "model_12c_selected_model.csv", index=False)

    print("\nScreening leaders")
    print(
        screening[
            ["FeatureSet", "ParameterSet", "MeanHorizonAP", "Risk48_AP", "Risk6_AP", "MacroF1"]
        ].head(10)
    )
    print("\nMulti-seed PolicyDev ranking")
    print(policy_ranking)
    print("\nSelected model")
    print(winner_frame.to_string(index=False))


if __name__ == "__main__":
    main()


