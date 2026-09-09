from pathlib import Path
import gc

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


SEEDS = [42, 52, 62, 72, 82]
CALIBRATION_FRACTION = 0.15
POLICY_FRACTION = 0.15
CLASS_IDS = np.arange(5)
HORIZONS = {
    "Risk48": [1, 2, 3, 4],
    "Risk24": [2, 3, 4],
    "Risk12": [3, 4],
    "Risk6": [4],
}
TIME_WINDOWS = [6.0, 12.0, 24.0, 48.0]
INSPECTION_BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
EPS = 1e-7

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "raw"
OUTPUT_DIR = SCRIPT_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    files = {
        "train_op": DATA_DIR / "train_operational_readouts.csv",
        "train_spec": DATA_DIR / "train_specifications.csv",
        "train_tte": DATA_DIR / "train_tte.csv",
        "val_op": DATA_DIR / "validation_operational_readouts.csv",
        "val_spec": DATA_DIR / "validation_specifications.csv",
        "val_labels": DATA_DIR / "validation_labels.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    return {name: pd.read_csv(path) for name, path in files.items()}


def make_generalization_features(df, raw_features):
    # CHANGE 1A: collapse duplicate vehicle/time readouts so zero-time changes are not treated as motion.
    df = (
        df.sort_values(["vehicle_id", "time_step"])
        .drop_duplicates(["vehicle_id", "time_step"], keep="last")
    )
    df = df.sort_values(["vehicle_id", "time_step"]).reset_index(drop=True)
    grouped = df.groupby("vehicle_id", sort=False, observed=True)
    current = df[raw_features].to_numpy(dtype="float32")

    # CHANGE 1: use real 6/12/24/48 time-unit windows instead of the last 1/3/5 readouts.
    identifiers = df[["vehicle_id", "time_step"]].copy()
    identifiers["time_gap"] = grouped["time_step"].diff().astype("float32")
    identifiers["history_count"] = grouped.cumcount().astype("float32")
    feature_blocks = [identifiers]

    group_indices = [indices for indices in grouped.indices.values()]
    all_times = df["time_step"].to_numpy(dtype="float32")

    for window in TIME_WINDOWS:
        reference_rows = np.full(len(df), -1, dtype=np.int64)
        window_counts = np.zeros(len(df), dtype="float32")

        for indices in group_indices:
            times = all_times[indices]
            reference_local = np.searchsorted(
                times,
                times - window,
                side="right",
            ) - 1
            valid = reference_local >= 0
            reference_rows[indices[valid]] = indices[reference_local[valid]]

            window_start = np.searchsorted(
                times,
                times - window,
                side="left",
            )
            window_counts[indices] = (
                np.arange(len(indices), dtype="float32")
                - window_start.astype("float32")
                + 1.0
            )

        valid_rows = reference_rows >= 0
        delta = np.full(current.shape, np.nan, dtype="float32")
        rate = np.full(current.shape, np.nan, dtype="float32")
        actual_gap_all = np.full(len(df), np.nan, dtype="float32")
        delta[valid_rows] = current[valid_rows] - current[reference_rows[valid_rows]]
        actual_gap = (
            all_times[valid_rows] - all_times[reference_rows[valid_rows]]
        ).astype("float32")
        actual_gap_all[valid_rows] = actual_gap
        nonzero_gap = actual_gap > 0
        valid_positions = np.flatnonzero(valid_rows)
        rate_positions = valid_positions[nonzero_gap]
        rate[rate_positions] = delta[rate_positions] / actual_gap[nonzero_gap, None]

        window_name = str(int(window))
        delta_frame = pd.DataFrame(
            delta,
            columns=[f"{column}__change_{window_name}" for column in raw_features],
        )
        rate_frame = pd.DataFrame(
            rate,
            columns=[f"{column}__rate_{window_name}" for column in raw_features],
        )
        count_frame = pd.DataFrame(
            {
                f"actual_gap_{window_name}": actual_gap_all,
                f"readout_count_{window_name}": window_counts,
                f"history_available_{window_name}": valid_rows.astype("int8"),
            }
        )
        feature_blocks.extend([delta_frame, rate_frame, count_frame])
        del reference_rows, window_counts, delta, rate, actual_gap_all
        gc.collect()

    output = pd.concat(feature_blocks, axis=1)
    output.replace([np.inf, -np.inf], np.nan, inplace=True)
    del current, group_indices, feature_blocks
    gc.collect()
    return output


def tte_to_class(tte):
    if pd.isna(tte) or tte < 0:
        return np.nan
    if tte <= 6:
        return 4
    if tte <= 12:
        return 3
    if tte <= 24:
        return 2
    if tte <= 48:
        return 1
    return 0


def build_labelled_candidates(train_features, train_tte):
    labelled = train_features.merge(
        train_tte,
        on="vehicle_id",
        how="left",
        validate="many_to_one",
    )
    labelled["time_to_endpoint"] = (
        labelled["length_of_study_time_step"] - labelled["time_step"]
    )
    labelled["class_label"] = np.nan
    repair = labelled["in_study_repair"].eq(1)
    labelled.loc[repair, "class_label"] = (
        labelled.loc[repair, "time_to_endpoint"].map(tte_to_class)
    )
    safe_c0 = (
        labelled["in_study_repair"].eq(0)
        & labelled["time_to_endpoint"].gt(48)
    )
    labelled.loc[safe_c0, "class_label"] = 0
    labelled = labelled.dropna(subset=["class_label"]).copy()
    labelled["class_label"] = labelled["class_label"].astype("int8")
    return labelled


def make_split_strata(train_tte):
    # CHANGE 2: stratify by repair status and study-duration band, not repair alone.
    duration_band = pd.qcut(
        train_tte["length_of_study_time_step"],
        q=4,
        labels=False,
        duplicates="drop",
    ).astype(str)
    return train_tte["in_study_repair"].astype(str) + "_" + duration_band


def split_vehicle_ids(train_tte, seed):
    vehicle_ids = train_tte["vehicle_id"].to_numpy()
    strata = make_split_strata(train_tte).to_numpy()
    temp_fraction = CALIBRATION_FRACTION + POLICY_FRACTION
    base_ids, temp_ids, _, temp_strata = train_test_split(
        vehicle_ids,
        strata,
        test_size=temp_fraction,
        random_state=seed,
        stratify=strata,
    )
    cal_ids, policy_ids = train_test_split(
        temp_ids,
        train_size=CALIBRATION_FRACTION / temp_fraction,
        random_state=seed,
        stratify=temp_strata,
    )
    return set(base_ids), set(cal_ids), set(policy_ids)


def vehicle_aware_training_sample(df, seed):
    repair = df[df["in_study_repair"].eq(1)]
    nonrepair = df[df["in_study_repair"].eq(0) & df["class_label"].eq(0)]
    repair_sample = (
        repair.groupby(["vehicle_id", "class_label"], group_keys=False)
        .sample(n=1, random_state=seed)
    )
    nonrepair_sample = (
        nonrepair.groupby("vehicle_id", group_keys=False)
        .sample(n=1, random_state=seed)
    )
    return pd.concat([repair_sample, nonrepair_sample], ignore_index=True)


def official_last_snapshot(val_features, val_spec, val_labels):
    snapshot = (
        val_features.sort_values(["vehicle_id", "time_step"])
        .groupby("vehicle_id", sort=False)
        .tail(1)
    )
    return (
        snapshot.merge(val_spec, on="vehicle_id", validate="one_to_one")
        .merge(val_labels, on="vehicle_id", validate="one_to_one")
        .reset_index(drop=True)
    )


def landmark_snapshot(df, target_time_distribution, seed):
    # CHANGE 3: replace a uniformly random row with an external-like landmark time.
    rng = np.random.default_rng(seed)
    target_times = rng.choice(target_time_distribution, size=df["vehicle_id"].nunique())
    rows = []
    for target_time, (_, group) in zip(
        target_times,
        df.sort_values(["vehicle_id", "time_step"]).groupby("vehicle_id", sort=False),
    ):
        times = group["time_step"].to_numpy()
        position = np.searchsorted(times, target_time, side="right") - 1
        position = max(position, 0)
        rows.append(group.iloc[position])
    return pd.DataFrame(rows).reset_index(drop=True)


def add_specs(df, specs):
    return df.merge(specs, on="vehicle_id", how="left", validate="many_to_one")


def make_preprocessor(frame):
    categorical = [column for column in frame.columns if column.startswith("Spec_")]
    numeric = [column for column in frame.columns if column not in categorical]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                categorical,
            ),
        ]
    )


def feature_frame(df):
    excluded = {
        "vehicle_id",
        "time_step",
        "class_label",
        "length_of_study_time_step",
        "in_study_repair",
        "time_to_endpoint",
    }
    return df.drop(columns=[c for c in excluded if c in df.columns])


def normalize_probabilities(proba):
    proba = np.clip(np.asarray(proba, dtype="float64"), EPS, None)
    return proba / proba.sum(axis=1, keepdims=True)


def logit_features(proba):
    proba = normalize_probabilities(proba)
    logs = np.log(proba)
    return logs[:, :-1] - logs[:, [-1]]


def fit_calibrator(raw_proba, labels, seed):
    # CHANGE 4: calibration is fitted on a disjoint vehicle-level landmark snapshot.
    if set(np.unique(labels)) != set(CLASS_IDS):
        raise ValueError(
            f"Calibration landmark is missing classes: {sorted(set(CLASS_IDS) - set(labels))}"
        )
    calibrator = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=seed,
    )
    calibrator.fit(logit_features(raw_proba), labels)
    return calibrator


def calibrated_proba(calibrator, raw_proba):
    predicted = calibrator.predict_proba(logit_features(raw_proba))
    aligned = np.zeros((len(predicted), 5), dtype="float64")
    aligned[:, calibrator.classes_.astype(int)] = predicted
    return normalize_probabilities(aligned)


def safe_ap(y_true, score):
    return float(average_precision_score(y_true, score)) if np.unique(y_true).size == 2 else np.nan


def safe_auc(y_true, score):
    return float(roc_auc_score(y_true, score)) if np.unique(y_true).size == 2 else np.nan


def evaluate_horizons(y_true, proba, split_name, seed):
    rows = []
    for horizon, positive_classes in HORIZONS.items():
        binary = np.isin(y_true, positive_classes).astype(int)
        score = proba[:, positive_classes].sum(axis=1)
        rows.append(
            {
                "Seed": seed,
                "Split": split_name,
                "Horizon": horizon,
                "Prevalence": binary.mean(),
                "AveragePrecision": safe_ap(binary, score),
                "ROCAUC": safe_auc(binary, score),
            }
        )
    return rows


def evaluate_topk(y_true, proba, split_name, seed):
    rows = []
    for horizon, positive_classes in HORIZONS.items():
        binary = np.isin(y_true, positive_classes)
        score = proba[:, positive_classes].sum(axis=1)
        order = np.argsort(-score, kind="stable")
        for budget in INSPECTION_BUDGETS:
            k = max(1, int(np.ceil(len(y_true) * budget)))
            selected = order[:k]
            tp = int(binary[selected].sum())
            positives = int(binary.sum())
            c4_total = int((y_true == 4).sum())
            rows.append(
                {
                    "Seed": seed,
                    "Split": split_name,
                    "Horizon": horizon,
                    "Budget": budget,
                    "AlertCount": k,
                    "PrecisionAtK": tp / k,
                    "RecallAtK": tp / positives if positives else np.nan,
                    "LiftAtK": (tp / k) / binary.mean() if binary.mean() else np.nan,
                    "C4RecallAtK": (
                        int((y_true[selected] == 4).sum()) / c4_total if c4_total else np.nan
                    ),
                }
            )
    return rows


def summarize(df, groups):
    numeric = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in {"Seed"}
    ]
    return df.groupby(groups, dropna=False)[numeric].agg(["mean", "std"]).reset_index()


def main():
    data = load_data()
    raw_features = [
        column
        for column in data["train_op"].columns
        if column not in {"vehicle_id", "time_step"}
    ]

    print("Creating generalization-robust trend features...")
    train_op = data.pop("train_op")
    train_features = make_generalization_features(train_op, raw_features)
    del train_op
    gc.collect()
    val_op = data.pop("val_op")
    val_features = make_generalization_features(val_op, raw_features)
    del val_op
    gc.collect()
    candidates = build_labelled_candidates(train_features, data["train_tte"])
    official = official_last_snapshot(
        val_features,
        data["val_spec"],
        data["val_labels"],
    )
    target_times = official["time_step"].to_numpy()

    del train_features, val_features
    gc.collect()

    classification_rows = []
    calibration_rows = []
    horizon_rows = []
    topk_rows = []
    validation_predictions = []

    for seed in SEEDS:
        print(f"Seed {seed}")
        base_ids, cal_ids, policy_ids = split_vehicle_ids(data["train_tte"], seed)
        base_candidates = candidates[candidates["vehicle_id"].isin(base_ids)]
        cal_candidates = candidates[candidates["vehicle_id"].isin(cal_ids)]
        policy_candidates = candidates[candidates["vehicle_id"].isin(policy_ids)]

        train_sample = vehicle_aware_training_sample(base_candidates, seed)
        cal_snapshot = landmark_snapshot(cal_candidates, target_times, seed + 1000)
        policy_snapshot = landmark_snapshot(policy_candidates, target_times, seed + 2000)

        train_sample = add_specs(train_sample, data["train_spec"])
        cal_snapshot = add_specs(cal_snapshot, data["train_spec"])
        policy_snapshot = add_specs(policy_snapshot, data["train_spec"])

        x_train = feature_frame(train_sample)
        x_cal = feature_frame(cal_snapshot)
        x_policy = feature_frame(policy_snapshot)
        x_official = feature_frame(official)
        x_cal = x_cal.reindex(columns=x_train.columns)
        x_policy = x_policy.reindex(columns=x_train.columns)
        x_official = x_official.reindex(columns=x_train.columns)

        y_train = train_sample["class_label"].astype(int).to_numpy()
        y_cal = cal_snapshot["class_label"].astype(int).to_numpy()
        y_policy = policy_snapshot["class_label"].astype(int).to_numpy()
        y_official = official["class_label"].astype(int).to_numpy()

        preprocessor = make_preprocessor(x_train)
        xt_train = preprocessor.fit_transform(x_train)
        xt_cal = preprocessor.transform(x_cal)
        xt_policy = preprocessor.transform(x_policy)
        xt_official = preprocessor.transform(x_official)

        # CHANGE 5: stronger regularization; no additional class weights.
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            n_estimators=300,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=8,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.65,
            reg_alpha=0.2,
            reg_lambda=5.0,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(xt_train, y_train)

        raw_cal = normalize_probabilities(model.predict_proba(xt_cal))
        raw_policy = normalize_probabilities(model.predict_proba(xt_policy))
        raw_official = normalize_probabilities(model.predict_proba(xt_official))
        calibrator = fit_calibrator(raw_cal, y_cal, seed)
        policy_proba = calibrated_proba(calibrator, raw_policy)
        official_proba = calibrated_proba(calibrator, raw_official)

        for split_name, labels, proba in [
            ("PolicyDev", y_policy, policy_proba),
            ("OfficialValidation", y_official, official_proba),
        ]:
            prediction = proba.argmax(axis=1)
            classification_rows.append(
                {
                    "Seed": seed,
                    "Split": split_name,
                    "Accuracy": accuracy_score(labels, prediction),
                    "MacroF1": f1_score(labels, prediction, labels=CLASS_IDS, average="macro", zero_division=0),
                }
            )
            calibration_rows.append(
                {
                    "Seed": seed,
                    "Split": split_name,
                    "LogLoss": log_loss(labels, proba, labels=CLASS_IDS),
                }
            )
            horizon_rows.extend(evaluate_horizons(labels, proba, split_name, seed))
            topk_rows.extend(evaluate_topk(labels, proba, split_name, seed))

        prediction_frame = pd.DataFrame(
            {
                "vehicle_id": official["vehicle_id"].to_numpy(),
                "actual_class": y_official,
                "Seed": seed,
                **{f"P_C{i}": official_proba[:, i] for i in CLASS_IDS},
            }
        )
        validation_predictions.append(prediction_frame)

        del model, calibrator, preprocessor
        del xt_train, xt_cal, xt_policy, xt_official
        gc.collect()

    classification = pd.DataFrame(classification_rows)
    calibration = pd.DataFrame(calibration_rows)
    horizons = pd.DataFrame(horizon_rows)
    topk = pd.DataFrame(topk_rows)
    predictions = pd.concat(validation_predictions, ignore_index=True)
    probability_columns = [f"P_C{i}" for i in CLASS_IDS]
    averaged_predictions = (
        predictions.groupby(["vehicle_id", "actual_class"], as_index=False)[probability_columns]
        .mean()
    )
    averaged_predictions["predicted_class"] = (
        averaged_predictions[probability_columns].to_numpy().argmax(axis=1)
    )

    classification.to_csv(OUTPUT_DIR / "model_11c_classification_by_seed.csv", index=False)
    calibration.to_csv(OUTPUT_DIR / "model_11c_calibration_by_seed.csv", index=False)
    horizons.to_csv(OUTPUT_DIR / "model_11c_horizon_metrics_by_seed.csv", index=False)
    topk.to_csv(OUTPUT_DIR / "model_11c_topk_metrics_by_seed.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "model_11c_validation_predictions_by_seed.csv", index=False)
    averaged_predictions.to_csv(
        OUTPUT_DIR / "model_11c_validation_predictions_ensemble.csv",
        index=False,
    )
    summarize(classification, ["Split"]).to_csv(
        OUTPUT_DIR / "model_11c_classification_summary.csv",
        index=False,
    )
    summarize(calibration, ["Split"]).to_csv(
        OUTPUT_DIR / "model_11c_calibration_summary.csv",
        index=False,
    )
    summarize(horizons, ["Split", "Horizon"]).to_csv(
        OUTPUT_DIR / "model_11c_horizon_metrics_summary.csv",
        index=False,
    )
    summarize(topk, ["Split", "Horizon", "Budget"]).to_csv(
        OUTPUT_DIR / "model_11c_topk_metrics_summary.csv",
        index=False,
    )

    print("\nClassification summary")
    print(classification.groupby("Split")[["Accuracy", "MacroF1"]].agg(["mean", "std"]))
    print("\nHorizon summary")
    print(
        horizons.groupby(["Split", "Horizon"])[["AveragePrecision", "ROCAUC"]]
        .agg(["mean", "std"])
    )
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
