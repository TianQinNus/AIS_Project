# ============================================================
# FleetCare AI
# Test 09c
# Rebase c-track calibration/ranking protocol onto teammate's
# latest base model:
#
#   Vehicle-Aware Sampling
#   + Multi-Readout Historical Features
#   + Multiclass XGBoost
#   + NO Class Weighting
#
# Why this test exists
# --------------------
# Tests 05c-08c were run on an older base:
#
#   One-step Temporal Features
#   + Vehicle-Aware Sampling
#   + W1 Light Class Weighting
#
# Meanwhile, teammate experiments found a stronger balanced
# candidate:
#
#   Multi-Readout Historical Features
#   + Vehicle-Aware Sampling
#   + NO Class Weighting
#
# Test 09c therefore changes ONLY the BASE MODEL representation
# and weighting strategy, while keeping our robust downstream
# methodology:
#
#   70% Base Train
#   15% Calibration
#   15% Selection / Policy Dev
#   5 random seeds
#
# For each seed:
#
#   1) Train the teammate-style latest base model.
#   2) Compare Raw / PriorCorrected / LogitCalibrated probability.
#   3) Select calibration method on Selection Dev by LogLoss.
#   4) Build Risk48 / Risk24 / Risk12 / Risk6.
#   5) Evaluate fleet ranking at Top 1/2/5/10/20%.
#   6) Use official validation ONLY as external diagnostic.
#
# Main question
# -------------
# Does the teammate's stronger multi-history + no-weight base
# improve the EXTERNAL ranking generalization problem exposed
# by Test 08c?
#
# This test does NOT:
#   - tune the official validation set
#   - automatically choose a final Top-K budget
#   - automatically choose a final risk horizon
#
# ============================================================

from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIG
# ============================================================

SEEDS = [42, 52, 62, 72, 82]

BASE_TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.15
SELECTION_POLICY_FRACTION = 0.15

INSPECTION_BUDGETS = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
]

NUM_CLASSES = 5
CLASS_IDS = np.arange(NUM_CLASSES)

EPS = 1e-12
ECE_BINS = 15

pd.set_option("display.max_columns", 350)
pd.set_option("display.width", 360)


# ============================================================
# 2. OLD c-TRACK BENCHMARKS
# ============================================================
#
# These constants are only for reporting deltas.
# They come from the previous one-step + W1 c-track.
#
# Test 07c:
#   selected calibration = LogitCalibrated
#
# Test 08c official-validation ranking, LogitCalibrated:
# ============================================================

OLD_07C_SELECTION_LOGLOSS = {
    "LogitCalibrated": 0.13780679,
}

OLD_08C_EXTERNAL_RANKING = {
    "Risk48": {
        "AveragePrecision": 0.0489,
        "ROCAUC": 0.6578,
        "Top5_C4Recall": 0.1237,
    },
    "Risk24": {
        "AveragePrecision": 0.0479,
        "ROCAUC": 0.6621,
        "Top5_C4Recall": 0.1316,
    },
    "Risk12": {
        "AveragePrecision": 0.0461,
        "ROCAUC": 0.6623,
        "Top5_C4Recall": 0.1421,
    },
    "Risk6": {
        "AveragePrecision": 0.0408,
        "ROCAUC": 0.6871,
        "Top5_C4Recall": 0.1316,
    },
}


# ============================================================
# 3. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_data_dir():
    candidates = [
        SCRIPT_DIR / "raw",
        SCRIPT_DIR.parent / "data" / "raw",
        Path("../data/raw").resolve(),
        Path("data/raw").resolve(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return SCRIPT_DIR / "raw"


DATA_DIR = resolve_data_dir()
OUTPUT_DIR = DATA_DIR.parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OP_PATH = DATA_DIR / "train_operational_readouts.csv"
TRAIN_SPEC_PATH = DATA_DIR / "train_specifications.csv"
TRAIN_TTE_PATH = DATA_DIR / "train_tte.csv"

VAL_OP_PATH = DATA_DIR / "validation_operational_readouts.csv"
VAL_SPEC_PATH = DATA_DIR / "validation_specifications.csv"
VAL_LABEL_PATH = DATA_DIR / "validation_labels.csv"


# ============================================================
# 4. CHECK FILES
# ============================================================

required_files = [
    TRAIN_OP_PATH,
    TRAIN_SPEC_PATH,
    TRAIN_TTE_PATH,
    VAL_OP_PATH,
    VAL_SPEC_PATH,
    VAL_LABEL_PATH,
]

print("=" * 120)
print("TEST 09c — REBASED MULTI-HISTORY CALIBRATION + RANKING")
print("=" * 120)

print("\nResolved DATA_DIR:")
print(DATA_DIR)

for path in required_files:
    print(f"{path.name}: {path.exists()}")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Resolved DATA_DIR: {DATA_DIR}"
        )


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n" + "=" * 120)
print("LOADING DATA")
print("=" * 120)

train_op = pd.read_csv(TRAIN_OP_PATH)
train_spec = pd.read_csv(TRAIN_SPEC_PATH)
train_tte = pd.read_csv(TRAIN_TTE_PATH)

val_op = pd.read_csv(VAL_OP_PATH)
val_spec = pd.read_csv(VAL_SPEC_PATH)
val_labels = pd.read_csv(VAL_LABEL_PATH)

print("Train operational:", train_op.shape)
print("Train specifications:", train_spec.shape)
print("Train TTE:", train_tte.shape)

print("Validation operational:", val_op.shape)
print("Validation specifications:", val_spec.shape)
print("Validation labels:", val_labels.shape)

train_vehicle_count = train_op["vehicle_id"].nunique()
validation_vehicle_count = val_op["vehicle_id"].nunique()

assert train_vehicle_count == train_spec["vehicle_id"].nunique()
assert train_vehicle_count == train_tte["vehicle_id"].nunique()

print("\nTraining vehicles:", train_vehicle_count)
print("Validation vehicles:", validation_vehicle_count)


# ============================================================
# 6. RAW FEATURES
# ============================================================

RAW_FEATURE_COLS = [
    col
    for col in train_op.columns
    if col not in [
        "vehicle_id",
        "time_step",
    ]
]

print("\nRaw operational features:", len(RAW_FEATURE_COLS))


# ============================================================
# 7. MULTI-READOUT HISTORICAL FEATURES
# ============================================================
#
# Match teammate's latest candidate:
#
# Current
# Previous
# Delta
# Rate
# Change from 3 readouts ago
# Change from 5 readouts ago
# Rolling mean last 3
# Rolling std last 3
# Rolling mean last 5
# Rolling std last 5
#
# Windows are READOUT windows, not fixed-time windows.
# ============================================================

def create_history_features(
    df,
    feature_cols,
):
    df = (
        df
        .sort_values(
            ["vehicle_id", "time_step"]
        )
        .reset_index(drop=True)
        .copy()
    )

    grouped = df.groupby(
        "vehicle_id",
        sort=False,
    )

    df["time_gap"] = (
        grouped["time_step"]
        .diff()
        .astype(np.float32)
    )

    safe_gap = (
        df["time_gap"]
        .replace(0, np.nan)
    )

    previous = (
        grouped[feature_cols]
        .shift(1)
        .astype(np.float32)
    )
    previous.columns = [
        f"{col}__prev"
        for col in feature_cols
    ]

    current_array = (
        df[feature_cols]
        .to_numpy(dtype=np.float32)
    )

    previous_array = (
        previous
        .to_numpy(dtype=np.float32)
    )

    delta_array = (
        current_array
        -
        previous_array
    )

    delta = pd.DataFrame(
        delta_array,
        index=df.index,
        columns=[
            f"{col}__delta"
            for col in feature_cols
        ],
    )

    gap_array = (
        safe_gap
        .to_numpy(dtype=np.float32)
        [:, None]
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        rate_array = (
            delta_array
            /
            gap_array
        )

    rate = pd.DataFrame(
        rate_array,
        index=df.index,
        columns=[
            f"{col}__rate"
            for col in feature_cols
        ],
    )

    rate.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    lag3 = (
        grouped[feature_cols]
        .shift(3)
        .astype(np.float32)
    )

    change3 = pd.DataFrame(
        (
            current_array
            -
            lag3.to_numpy(dtype=np.float32)
        ),
        index=df.index,
        columns=[
            f"{col}__change3"
            for col in feature_cols
        ],
    )

    lag5 = (
        grouped[feature_cols]
        .shift(5)
        .astype(np.float32)
    )

    change5 = pd.DataFrame(
        (
            current_array
            -
            lag5.to_numpy(dtype=np.float32)
        ),
        index=df.index,
        columns=[
            f"{col}__change5"
            for col in feature_cols
        ],
    )

    rolling_mean3 = (
        grouped[feature_cols]
        .rolling(
            window=3,
            min_periods=1,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
        .reindex(df.index)
        .astype(np.float32)
    )
    rolling_mean3.columns = [
        f"{col}__mean3"
        for col in feature_cols
    ]

    rolling_std3 = (
        grouped[feature_cols]
        .rolling(
            window=3,
            min_periods=2,
        )
        .std()
        .reset_index(
            level=0,
            drop=True,
        )
        .reindex(df.index)
        .astype(np.float32)
    )
    rolling_std3.columns = [
        f"{col}__std3"
        for col in feature_cols
    ]

    rolling_mean5 = (
        grouped[feature_cols]
        .rolling(
            window=5,
            min_periods=1,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
        .reindex(df.index)
        .astype(np.float32)
    )
    rolling_mean5.columns = [
        f"{col}__mean5"
        for col in feature_cols
    ]

    rolling_std5 = (
        grouped[feature_cols]
        .rolling(
            window=5,
            min_periods=2,
        )
        .std()
        .reset_index(
            level=0,
            drop=True,
        )
        .reindex(df.index)
        .astype(np.float32)
    )
    rolling_std5.columns = [
        f"{col}__std5"
        for col in feature_cols
    ]

    result = pd.concat(
        [
            df,
            previous,
            delta,
            rate,
            change3,
            change5,
            rolling_mean3,
            rolling_std3,
            rolling_mean5,
            rolling_std5,
        ],
        axis=1,
    )

    result.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    del previous_array
    del delta_array
    del rate_array
    del gap_array
    del lag3
    del lag5

    gc.collect()

    return result


print("\n" + "=" * 120)
print("CREATING MULTI-READOUT HISTORY FEATURES")
print("=" * 120)

train_history = create_history_features(
    train_op,
    RAW_FEATURE_COLS,
)

val_history = create_history_features(
    val_op,
    RAW_FEATURE_COLS,
)

print("Train history:", train_history.shape)
print("Validation history:", val_history.shape)

del train_op
del val_op
gc.collect()


# ============================================================
# 8. CENSORING-AWARE FIVE-CLASS LABELS
# ============================================================

train_full = (
    train_history
    .merge(
        train_tte,
        on="vehicle_id",
        how="left",
        validate="many_to_one",
    )
)

train_full["time_to_endpoint"] = (
    train_full["length_of_study_time_step"]
    -
    train_full["time_step"]
)


def tte_to_class(tte):
    if pd.isna(tte):
        return np.nan

    if tte < 0:
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


train_full["class_label"] = np.nan

repair_mask = (
    train_full["in_study_repair"]
    ==
    1
)

train_full.loc[
    repair_mask,
    "class_label"
] = (
    train_full.loc[
        repair_mask,
        "time_to_endpoint"
    ]
    .apply(tte_to_class)
)

safe_non_repair_mask = (
    (
        train_full["in_study_repair"]
        ==
        0
    )
    &
    (
        train_full["time_to_endpoint"]
        >
        48
    )
)

train_full.loc[
    safe_non_repair_mask,
    "class_label"
] = 0

candidate_all = (
    train_full
    .dropna(
        subset=["class_label"]
    )
    .copy()
)

candidate_all["class_label"] = (
    candidate_all["class_label"]
    .astype(int)
)

del train_full
del train_history
gc.collect()


# ============================================================
# 9. OFFICIAL VALIDATION SNAPSHOT
# ============================================================

val_last = (
    val_history
    .groupby(
        "vehicle_id",
        sort=False,
    )
    .tail(1)
    .copy()
)

val_last = (
    val_last
    .merge(
        val_spec,
        on="vehicle_id",
        how="left",
        validate="one_to_one",
    )
)

val_last = (
    val_last
    .merge(
        val_labels,
        on="vehicle_id",
        how="left",
        validate="one_to_one",
    )
)

del val_history
gc.collect()

print("\nOfficial validation distribution:")
print(
    val_last["class_label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 10. VEHICLE TABLE / SPLIT
# ============================================================

vehicle_table = (
    train_tte[
        [
            "vehicle_id",
            "in_study_repair",
        ]
    ]
    .drop_duplicates("vehicle_id")
    .copy()
)


def three_way_vehicle_split(
    vehicle_df,
    random_state,
):
    base_ids, temp_ids = (
        train_test_split(
            vehicle_df["vehicle_id"].to_numpy(),

            test_size=(
                CALIBRATION_FRACTION
                +
                SELECTION_POLICY_FRACTION
            ),

            random_state=random_state,

            stratify=vehicle_df[
                "in_study_repair"
            ].to_numpy(),
        )
    )

    temp_table = (
        vehicle_df[
            vehicle_df["vehicle_id"]
            .isin(temp_ids)
        ]
        .copy()
    )

    calibration_relative_fraction = (
        CALIBRATION_FRACTION
        /
        (
            CALIBRATION_FRACTION
            +
            SELECTION_POLICY_FRACTION
        )
    )

    calibration_ids, selection_ids = (
        train_test_split(
            temp_table[
                "vehicle_id"
            ].to_numpy(),

            train_size=calibration_relative_fraction,

            random_state=random_state,

            stratify=temp_table[
                "in_study_repair"
            ].to_numpy(),
        )
    )

    base_ids = set(base_ids)
    calibration_ids = set(calibration_ids)
    selection_ids = set(selection_ids)

    assert base_ids.isdisjoint(calibration_ids)
    assert base_ids.isdisjoint(selection_ids)
    assert calibration_ids.isdisjoint(selection_ids)

    assert (
        len(base_ids)
        +
        len(calibration_ids)
        +
        len(selection_ids)
        ==
        len(vehicle_df)
    )

    return (
        base_ids,
        calibration_ids,
        selection_ids,
    )


# ============================================================
# 11. VEHICLE-AWARE BASE TRAINING SAMPLE
# ============================================================

def vehicle_aware_sample(
    candidate_df,
    random_state,
):
    repair_candidates = (
        candidate_df[
            candidate_df["in_study_repair"]
            ==
            1
        ]
        .copy()
    )

    non_repair_candidates = (
        candidate_df[
            candidate_df["in_study_repair"]
            ==
            0
        ]
        .copy()
    )

    repair_sampled = (
        repair_candidates
        .groupby(
            [
                "vehicle_id",
                "class_label",
            ],
            group_keys=False,
        )
        .sample(
            n=1,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )

    non_repair_sampled = (
        non_repair_candidates
        .groupby(
            "vehicle_id",
            group_keys=False,
        )
        .sample(
            n=1,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )

    sampled = pd.concat(
        [
            repair_sampled,
            non_repair_sampled,
        ],
        ignore_index=True,
    )

    return (
        sampled
        .sample(
            frac=1,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )


# ============================================================
# 12. VEHICLE-BALANCED WEIGHTS FOR CALIBRATION / SELECTION
# ============================================================

def add_vehicle_balanced_weights(
    df,
    weight_col,
):
    df = df.copy()

    rows_per_vehicle = (
        df
        .groupby("vehicle_id")[
            "vehicle_id"
        ]
        .transform("size")
    )

    df[weight_col] = (
        1.0
        /
        rows_per_vehicle
    )

    # Normalize mean row weight to 1.
    df[weight_col] = (
        df[weight_col]
        *
        (
            len(df)
            /
            df[weight_col].sum()
        )
    )

    return df


# ============================================================
# 13. ONE-CUTOFF POLICY SNAPSHOT
# ============================================================

def sample_one_cutoff_per_vehicle(
    df,
    random_state,
):
    snapshot = (
        df
        .groupby(
            "vehicle_id",
            group_keys=False,
        )
        .sample(
            n=1,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )

    assert (
        snapshot["vehicle_id"].nunique()
        ==
        len(snapshot)
    )

    return snapshot


# ============================================================
# 14. PROBABILITY HELPERS
# ============================================================

def normalize_probabilities(
    proba,
):
    proba = np.asarray(
        proba,
        dtype=np.float64,
    )

    proba = np.clip(
        proba,
        EPS,
        None,
    )

    return (
        proba
        /
        proba.sum(
            axis=1,
            keepdims=True,
        )
    )


def weighted_class_prior(
    y,
    sample_weight=None,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    if sample_weight is None:
        sample_weight = np.ones(
            len(y),
            dtype=np.float64,
        )

    else:
        sample_weight = np.asarray(
            sample_weight,
            dtype=np.float64,
        )

    counts = np.bincount(
        y,
        weights=sample_weight,
        minlength=NUM_CLASSES,
    )

    return (
        counts
        /
        counts.sum()
    )


def multiclass_brier(
    y_true,
    proba,
    sample_weight=None,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    proba = normalize_probabilities(
        proba
    )

    one_hot = np.eye(
        NUM_CLASSES
    )[y_true]

    row_score = (
        (
            proba
            -
            one_hot
        )
        **
        2
    ).sum(axis=1)

    if sample_weight is None:
        return float(
            row_score.mean()
        )

    return float(
        np.average(
            row_score,
            weights=sample_weight,
        )
    )


def multiclass_ece(
    y_true,
    proba,
    sample_weight=None,
    n_bins=15,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    proba = normalize_probabilities(
        proba
    )

    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)

    correct = (
        predicted
        ==
        y_true
    ).astype(float)

    if sample_weight is None:
        sample_weight = np.ones(
            len(y_true),
            dtype=np.float64,
        )

    else:
        sample_weight = np.asarray(
            sample_weight,
            dtype=np.float64,
        )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    total_weight = (
        sample_weight.sum()
    )

    ece = 0.0

    for bin_idx in range(n_bins):
        lower = edges[bin_idx]
        upper = edges[bin_idx + 1]

        if bin_idx == n_bins - 1:
            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        if not np.any(mask):
            continue

        bin_weight = (
            sample_weight[mask]
            .sum()
        )

        bin_accuracy = (
            np.average(
                correct[mask],
                weights=sample_weight[mask],
            )
        )

        bin_confidence = (
            np.average(
                confidence[mask],
                weights=sample_weight[mask],
            )
        )

        ece += (
            bin_weight
            /
            total_weight
        ) * abs(
            bin_accuracy
            -
            bin_confidence
        )

    return float(ece)


def calibration_metrics(
    seed,
    split_name,
    method_name,
    y_true,
    proba,
    sample_weight=None,
):
    proba = normalize_probabilities(
        proba
    )

    return {
        "Seed":
            seed,

        "Split":
            split_name,

        "ProbabilityMethod":
            method_name,

        "LogLoss":
            log_loss(
                y_true,
                proba,
                labels=CLASS_IDS,
                sample_weight=sample_weight,
            ),

        "Brier":
            multiclass_brier(
                y_true,
                proba,
                sample_weight=sample_weight,
            ),

        "ECE":
            multiclass_ece(
                y_true,
                proba,
                sample_weight=sample_weight,
                n_bins=ECE_BINS,
            ),

        "ArgmaxAccuracy":
            accuracy_score(
                y_true,
                np.argmax(
                    proba,
                    axis=1,
                ),
                sample_weight=sample_weight,
            ),
    }


# ============================================================
# 15. PRIOR CORRECTION
# ============================================================

def prior_correction_ratio(
    effective_train_prior,
    target_prior,
):
    return (
        target_prior
        /
        np.clip(
            effective_train_prior,
            EPS,
            None,
        )
    )


def apply_prior_ratio(
    raw_proba,
    ratio,
):
    return normalize_probabilities(
        raw_proba
        *
        ratio[None, :]
    )


# ============================================================
# 16. LOGIT CALIBRATION
# ============================================================

def probability_to_log_features(
    proba,
):
    proba = normalize_probabilities(
        proba
    )

    return np.log(
        np.clip(
            proba,
            EPS,
            1.0,
        )
    )


def fit_logit_calibrator(
    raw_proba,
    y,
    sample_weight,
    random_state,
):
    calibrator = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        random_state=random_state,
    )

    calibrator.fit(
        probability_to_log_features(
            raw_proba
        ),
        y,
        sample_weight=sample_weight,
    )

    if set(
        calibrator.classes_
    ) != set(
        CLASS_IDS
    ):
        raise ValueError(
            "Calibration split does not contain all 5 classes. "
            f"classes_ = {calibrator.classes_}"
        )

    return calibrator


def aligned_calibrator_proba(
    calibrator,
    raw_proba,
):
    partial = (
        calibrator
        .predict_proba(
            probability_to_log_features(
                raw_proba
            )
        )
    )

    aligned = np.zeros(
        (
            len(raw_proba),
            NUM_CLASSES,
        ),
        dtype=np.float64,
    )

    for idx, cls in enumerate(
        calibrator.classes_
    ):
        aligned[:, int(cls)] = (
            partial[:, idx]
        )

    return normalize_probabilities(
        aligned
    )


# ============================================================
# 17. LATEST TEAMMATE-STYLE BASE MODEL
# ============================================================
#
# IMPORTANT:
# NO class weights.
# ============================================================

def build_history_no_weight_pipeline(
    numeric_cols,
    categorical_cols,
    random_state,
):
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_cols,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_cols,
            ),
        ]
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=5,

        n_estimators=150,
        learning_rate=0.08,

        max_depth=5,
        min_child_weight=3,

        subsample=0.8,
        colsample_bytree=0.8,

        reg_alpha=0.0,
        reg_lambda=1.0,

        eval_metric="mlogloss",
        tree_method="hist",

        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                model,
            ),
        ]
    )


# ============================================================
# 18. CUMULATIVE RISKS
# ============================================================

def cumulative_risk_scores(
    proba,
):
    proba = normalize_probabilities(
        proba
    )

    return {
        "Risk48":
            (
                proba[:, 1]
                +
                proba[:, 2]
                +
                proba[:, 3]
                +
                proba[:, 4]
            ),

        "Risk24":
            (
                proba[:, 2]
                +
                proba[:, 3]
                +
                proba[:, 4]
            ),

        "Risk12":
            (
                proba[:, 3]
                +
                proba[:, 4]
            ),

        "Risk6":
            proba[:, 4],
    }


def horizon_target(
    y_class,
    horizon,
):
    y_class = np.asarray(
        y_class,
        dtype=int,
    )

    if horizon == "Risk48":
        return (
            y_class >= 1
        ).astype(int)

    if horizon == "Risk24":
        return (
            y_class >= 2
        ).astype(int)

    if horizon == "Risk12":
        return (
            y_class >= 3
        ).astype(int)

    if horizon == "Risk6":
        return (
            y_class >= 4
        ).astype(int)

    raise ValueError(
        f"Unknown horizon: {horizon}"
    )


# ============================================================
# 19. RANKING METRICS
# ============================================================

def safe_ap(
    y_true,
    score,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    if y_true.sum() == 0:
        return np.nan

    return float(
        average_precision_score(
            y_true,
            score,
        )
    )


def safe_auc(
    y_true,
    score,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    if len(
        np.unique(y_true)
    ) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_true,
            score,
        )
    )


def evaluate_horizon(
    seed,
    split_name,
    probability_method,
    horizon,
    actual_class,
    score,
):
    target = horizon_target(
        actual_class,
        horizon,
    )

    return {
        "Seed":
            seed,

        "Split":
            split_name,

        "ProbabilityMethod":
            probability_method,

        "Horizon":
            horizon,

        "N":
            len(target),

        "PositiveCount":
            int(target.sum()),

        "Prevalence":
            float(target.mean()),

        "AveragePrecision":
            safe_ap(
                target,
                score,
            ),

        "ROCAUC":
            safe_auc(
                target,
                score,
            ),
    }


def evaluate_capacity(
    seed,
    split_name,
    probability_method,
    horizon,
    actual_class,
    score,
    budget_fraction,
):
    actual_class = np.asarray(
        actual_class,
        dtype=int,
    )

    score = np.asarray(
        score,
        dtype=np.float64,
    )

    target = horizon_target(
        actual_class,
        horizon,
    )

    n = len(actual_class)

    k = max(
        1,
        int(
            np.ceil(
                n
                *
                budget_fraction
            )
        )
    )

    order = np.argsort(
        -score,
        kind="mergesort",
    )

    selected = order[:k]

    tp = int(
        target[selected].sum()
    )

    fp = int(
        k - tp
    )

    total_positive = int(
        target.sum()
    )

    precision_at_k = (
        tp / k
        if k > 0
        else np.nan
    )

    recall_at_k = (
        tp / total_positive
        if total_positive > 0
        else np.nan
    )

    prevalence = (
        total_positive / n
        if n > 0
        else np.nan
    )

    lift_at_k = (
        precision_at_k
        /
        prevalence
        if prevalence > 0
        else np.nan
    )

    nni = (
        k / tp
        if tp > 0
        else np.inf
    )

    c4_mask = (
        actual_class == 4
    )

    total_c4 = int(
        c4_mask.sum()
    )

    c4_captured = int(
        c4_mask[selected].sum()
    )

    c4_recall = (
        c4_captured
        /
        total_c4
        if total_c4 > 0
        else np.nan
    )

    c3c4_mask = (
        actual_class >= 3
    )

    total_c3c4 = int(
        c3c4_mask.sum()
    )

    c3c4_captured = int(
        c3c4_mask[selected].sum()
    )

    c3c4_recall = (
        c3c4_captured
        /
        total_c3c4
        if total_c3c4 > 0
        else np.nan
    )

    return {
        "Seed":
            seed,

        "Split":
            split_name,

        "ProbabilityMethod":
            probability_method,

        "Horizon":
            horizon,

        "BudgetFraction":
            budget_fraction,

        "BudgetPercent":
            budget_fraction * 100.0,

        "FleetSize":
            n,

        "AlertCount":
            k,

        "HorizonPositiveCount":
            total_positive,

        "TruePositivesAtK":
            tp,

        "FalseAlertsAtK":
            fp,

        "PrecisionAtK":
            precision_at_k,

        "RecallAtK":
            recall_at_k,

        "LiftAtK":
            lift_at_k,

        "NumberNeededToInspect":
            nni,

        "C4Count":
            total_c4,

        "C4Captured":
            c4_captured,

        "C4RecallAtK":
            c4_recall,

        "C3C4Count":
            total_c3c4,

        "C3C4Captured":
            c3c4_captured,

        "C3C4RecallAtK":
            c3c4_recall,

        "RiskThresholdAtK":
            float(
                score[
                    selected[-1]
                ]
            ),
    }


# ============================================================
# 20. CLASSIFICATION DIAGNOSTIC
# ============================================================

def classification_diagnostic(
    seed,
    y_true,
    proba,
):
    pred = (
        np.argmax(
            proba,
            axis=1,
        )
        .astype(int)
    )

    c4_actual = (
        np.asarray(
            y_true
        )
        ==
        4
    )

    c4_pred = (
        pred == 4
    )

    c4_tp = int(
        np.sum(
            c4_actual
            &
            c4_pred
        )
    )

    c4_pred_count = int(
        c4_pred.sum()
    )

    c4_actual_count = int(
        c4_actual.sum()
    )

    c4_precision = (
        c4_tp
        /
        c4_pred_count
        if c4_pred_count > 0
        else 0.0
    )

    c4_recall = (
        c4_tp
        /
        c4_actual_count
        if c4_actual_count > 0
        else 0.0
    )

    c4_f1 = (
        2
        *
        c4_precision
        *
        c4_recall
        /
        (
            c4_precision
            +
            c4_recall
        )
        if (
            c4_precision
            +
            c4_recall
        ) > 0
        else 0.0
    )

    return {
        "Seed":
            seed,

        "Accuracy":
            accuracy_score(
                y_true,
                pred,
            ),

        "MacroF1":
            f1_score(
                y_true,
                pred,
                average="macro",
                zero_division=0,
            ),

        "C4Precision":
            c4_precision,

        "C4Recall":
            c4_recall,

        "C4F1":
            c4_f1,

        "PredC4":
            c4_pred_count,
    }


# ============================================================
# 21. FEATURE COLUMNS
# ============================================================

NON_FEATURE_COLS = [
    "vehicle_id",
    "class_label",
    "length_of_study_time_step",
    "in_study_repair",
    "time_to_endpoint",
    "_calibration_weight",
    "_selection_weight",
]


X_val_template = (
    val_last
    .drop(
        columns=[
            "vehicle_id",
            "class_label",
        ],
        errors="ignore",
    )
)

y_val = (
    val_last["class_label"]
    .astype(int)
    .to_numpy()
)


# ============================================================
# 22. RESULT STORES
# ============================================================

split_rows = []

classification_rows = []

selection_calibration_rows = []
validation_calibration_rows = []
seed_winner_rows = []

horizon_rows = []
capacity_rows = []

selected_method_ranking_rows = []


# ============================================================
# 23. MULTI-SEED LOOP
# ============================================================

print("\n" + "=" * 120)
print("STARTING 5-SEED REBASED EXPERIMENT")
print("=" * 120)

for seed in SEEDS:

    print("\n" + "#" * 120)
    print(f"SEED = {seed}")
    print("#" * 120)

    (
        base_ids,
        calibration_ids,
        selection_ids,
    ) = three_way_vehicle_split(
        vehicle_table,
        random_state=seed,
    )

    base_candidates = (
        candidate_all[
            candidate_all["vehicle_id"]
            .isin(base_ids)
        ]
        .copy()
    )

    calibration_candidates = (
        candidate_all[
            candidate_all["vehicle_id"]
            .isin(calibration_ids)
        ]
        .copy()
    )

    selection_candidates = (
        candidate_all[
            candidate_all["vehicle_id"]
            .isin(selection_ids)
        ]
        .copy()
    )

    base_sampled = (
        vehicle_aware_sample(
            base_candidates,
            random_state=seed,
        )
    )

    calibration_candidates = (
        add_vehicle_balanced_weights(
            calibration_candidates,
            "_calibration_weight",
        )
    )

    selection_candidates = (
        add_vehicle_balanced_weights(
            selection_candidates,
            "_selection_weight",
        )
    )

    policy_snapshot = (
        sample_one_cutoff_per_vehicle(
            selection_candidates,
            random_state=seed,
        )
    )

    split_rows.append(
        {
            "Seed":
                seed,

            "BaseTrainVehicleIDs":
                len(base_ids),

            "CalibrationVehicleIDs":
                len(calibration_ids),

            "SelectionVehicleIDs":
                len(selection_ids),

            "PolicySnapshotVehicles":
                policy_snapshot[
                    "vehicle_id"
                ].nunique(),

            "PolicyCoverage":
                (
                    policy_snapshot[
                        "vehicle_id"
                    ].nunique()
                    /
                    len(selection_ids)
                ),
        }
    )

    # --------------------------------------------------------
    # Join specifications
    # --------------------------------------------------------

    base_sampled = (
        base_sampled
        .merge(
            train_spec,
            on="vehicle_id",
            how="left",
            validate="many_to_one",
        )
    )

    calibration_candidates = (
        calibration_candidates
        .merge(
            train_spec,
            on="vehicle_id",
            how="left",
            validate="many_to_one",
        )
    )

    selection_candidates = (
        selection_candidates
        .merge(
            train_spec,
            on="vehicle_id",
            how="left",
            validate="many_to_one",
        )
    )

    # Snapshot comes from selection_candidates BEFORE spec merge.
    # Rebuild to avoid duplicate Spec_* columns.
    policy_snapshot = (
        policy_snapshot
        .merge(
            train_spec,
            on="vehicle_id",
            how="left",
            validate="one_to_one",
        )
    )

    # --------------------------------------------------------
    # Build X/Y
    # --------------------------------------------------------

    X_train = (
        base_sampled
        .drop(
            columns=NON_FEATURE_COLS,
            errors="ignore",
        )
    )

    y_train = (
        base_sampled[
            "class_label"
        ]
        .astype(int)
        .copy()
    )

    X_cal = (
        calibration_candidates
        .drop(
            columns=NON_FEATURE_COLS,
            errors="ignore",
        )
    )

    y_cal = (
        calibration_candidates[
            "class_label"
        ]
        .astype(int)
        .copy()
    )

    cal_weights = (
        calibration_candidates[
            "_calibration_weight"
        ]
        .to_numpy(dtype=np.float64)
    )

    X_sel = (
        selection_candidates
        .drop(
            columns=NON_FEATURE_COLS,
            errors="ignore",
        )
    )

    y_sel = (
        selection_candidates[
            "class_label"
        ]
        .astype(int)
        .copy()
    )

    sel_weights = (
        selection_candidates[
            "_selection_weight"
        ]
        .to_numpy(dtype=np.float64)
    )

    X_policy = (
        policy_snapshot
        .drop(
            columns=NON_FEATURE_COLS,
            errors="ignore",
        )
    )

    y_policy = (
        policy_snapshot[
            "class_label"
        ]
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Align schema
    # --------------------------------------------------------

    assert set(
        X_train.columns
    ) == set(
        X_cal.columns
    )

    assert set(
        X_train.columns
    ) == set(
        X_sel.columns
    )

    assert set(
        X_train.columns
    ) == set(
        X_policy.columns
    )

    assert set(
        X_train.columns
    ) == set(
        X_val_template.columns
    )

    X_cal = X_cal[
        X_train.columns
    ]

    X_sel = X_sel[
        X_train.columns
    ]

    X_policy = X_policy[
        X_train.columns
    ]

    X_val = X_val_template[
        X_train.columns
    ].copy()

    # --------------------------------------------------------
    # Feature types
    # --------------------------------------------------------

    categorical_cols = [
        col
        for col in X_train.columns
        if col.startswith(
            "Spec_"
        )
    ]

    numeric_cols = [
        col
        for col in X_train.columns
        if col not in categorical_cols
    ]

    # --------------------------------------------------------
    # Train latest no-weight base
    # --------------------------------------------------------

    pipeline = (
        build_history_no_weight_pipeline(
            numeric_cols,
            categorical_cols,
            random_state=seed,
        )
    )

    print("Base train rows:", len(X_train))
    print("Calibration rows:", len(X_cal))
    print("Selection rows:", len(X_sel))
    print("Policy fleet:", len(X_policy))
    print("Official validation fleet:", len(X_val))

    pipeline.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Raw probability
    # --------------------------------------------------------

    cal_raw = (
        pipeline
        .predict_proba(X_cal)
    )

    sel_raw = (
        pipeline
        .predict_proba(X_sel)
    )

    policy_raw = (
        pipeline
        .predict_proba(X_policy)
    )

    val_raw = (
        pipeline
        .predict_proba(X_val)
    )

    classification_rows.append(
        classification_diagnostic(
            seed,
            y_val,
            val_raw,
        )
    )

    # --------------------------------------------------------
    # Prior correction
    # --------------------------------------------------------
    #
    # No explicit class weights now, but vehicle-aware sampling
    # still changes the effective class distribution.
    # --------------------------------------------------------

    effective_train_prior = (
        weighted_class_prior(
            y_train
        )
    )

    calibration_target_prior = (
        weighted_class_prior(
            y_cal,
            sample_weight=cal_weights,
        )
    )

    ratio = prior_correction_ratio(
        effective_train_prior,
        calibration_target_prior,
    )

    sel_prior = (
        apply_prior_ratio(
            sel_raw,
            ratio,
        )
    )

    policy_prior = (
        apply_prior_ratio(
            policy_raw,
            ratio,
        )
    )

    val_prior = (
        apply_prior_ratio(
            val_raw,
            ratio,
        )
    )

    # --------------------------------------------------------
    # Logit calibration
    # --------------------------------------------------------

    calibrator = (
        fit_logit_calibrator(
            cal_raw,
            y_cal,
            sample_weight=cal_weights,
            random_state=seed,
        )
    )

    sel_logit = (
        aligned_calibrator_proba(
            calibrator,
            sel_raw,
        )
    )

    policy_logit = (
        aligned_calibrator_proba(
            calibrator,
            policy_raw,
        )
    )

    val_logit = (
        aligned_calibrator_proba(
            calibrator,
            val_raw,
        )
    )

    selection_probability_sets = {
        "Raw":
            sel_raw,

        "PriorCorrected":
            sel_prior,

        "LogitCalibrated":
            sel_logit,
    }

    policy_probability_sets = {
        "Raw":
            policy_raw,

        "PriorCorrected":
            policy_prior,

        "LogitCalibrated":
            policy_logit,
    }

    validation_probability_sets = {
        "Raw":
            val_raw,

        "PriorCorrected":
            val_prior,

        "LogitCalibrated":
            val_logit,
    }

    # --------------------------------------------------------
    # Calibration selection on Selection Dev only
    # --------------------------------------------------------

    current_seed_rows = []

    for method, proba in (
        selection_probability_sets.items()
    ):
        row = calibration_metrics(
            seed=seed,
            split_name="SelectionDev",
            method_name=method,
            y_true=y_sel,
            proba=proba,
            sample_weight=sel_weights,
        )

        selection_calibration_rows.append(
            row
        )

        current_seed_rows.append(
            row
        )

    seed_calibration_df = pd.DataFrame(
        current_seed_rows
    )

    seed_winner = (
        seed_calibration_df
        .sort_values(
            "LogLoss",
            ascending=True,
        )
        .iloc[0]
    )

    seed_selected_method = (
        seed_winner[
            "ProbabilityMethod"
        ]
    )

    seed_winner_rows.append(
        {
            "Seed":
                seed,

            "SelectedProbabilityMethod":
                seed_selected_method,

            "SelectionLogLoss":
                seed_winner[
                    "LogLoss"
                ],

            "SelectionBrier":
                seed_winner[
                    "Brier"
                ],

            "SelectionECE":
                seed_winner[
                    "ECE"
                ],
        }
    )

    print(
        "Seed calibration winner:",
        seed_selected_method,
        "| LogLoss =",
        round(
            float(
                seed_winner[
                    "LogLoss"
                ]
            ),
            6,
        ),
    )

    # --------------------------------------------------------
    # External validation calibration diagnostics
    # --------------------------------------------------------

    for method, proba in (
        validation_probability_sets.items()
    ):
        validation_calibration_rows.append(
            calibration_metrics(
                seed=seed,
                split_name="OfficialValidation",
                method_name=method,
                y_true=y_val,
                proba=proba,
                sample_weight=None,
            )
        )

    # --------------------------------------------------------
    # Ranking metrics for ALL probability methods
    # --------------------------------------------------------

    for method in [
        "Raw",
        "PriorCorrected",
        "LogitCalibrated",
    ]:

        policy_scores = cumulative_risk_scores(
            policy_probability_sets[
                method
            ]
        )

        validation_scores = cumulative_risk_scores(
            validation_probability_sets[
                method
            ]
        )

        for horizon, score in (
            policy_scores.items()
        ):
            horizon_rows.append(
                evaluate_horizon(
                    seed=seed,
                    split_name="PolicyDev",
                    probability_method=method,
                    horizon=horizon,
                    actual_class=y_policy,
                    score=score,
                )
            )

            for budget in INSPECTION_BUDGETS:
                capacity_rows.append(
                    evaluate_capacity(
                        seed=seed,
                        split_name="PolicyDev",
                        probability_method=method,
                        horizon=horizon,
                        actual_class=y_policy,
                        score=score,
                        budget_fraction=budget,
                    )
                )

        for horizon, score in (
            validation_scores.items()
        ):
            horizon_rows.append(
                evaluate_horizon(
                    seed=seed,
                    split_name="OfficialValidation",
                    probability_method=method,
                    horizon=horizon,
                    actual_class=y_val,
                    score=score,
                )
            )

            for budget in INSPECTION_BUDGETS:
                capacity_rows.append(
                    evaluate_capacity(
                        seed=seed,
                        split_name="OfficialValidation",
                        probability_method=method,
                        horizon=horizon,
                        actual_class=y_val,
                        score=score,
                        budget_fraction=budget,
                    )
                )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del base_candidates
    del calibration_candidates
    del selection_candidates
    del policy_snapshot
    del base_sampled

    del X_train
    del X_cal
    del X_sel
    del X_policy
    del X_val

    del pipeline
    del calibrator

    del cal_raw
    del sel_raw
    del policy_raw
    del val_raw

    del sel_prior
    del policy_prior
    del val_prior

    del sel_logit
    del policy_logit
    del val_logit

    gc.collect()


# ============================================================
# 24. RESULT TABLES
# ============================================================

split_df = pd.DataFrame(
    split_rows
)

classification_df = pd.DataFrame(
    classification_rows
)

selection_calibration_df = pd.DataFrame(
    selection_calibration_rows
)

validation_calibration_df = pd.DataFrame(
    validation_calibration_rows
)

seed_winner_df = pd.DataFrame(
    seed_winner_rows
)

horizon_df = pd.DataFrame(
    horizon_rows
)

capacity_df = pd.DataFrame(
    capacity_rows
)


# ============================================================
# 25. CALIBRATION SUMMARY
# ============================================================

selection_calibration_summary_df = (
    selection_calibration_df
    .groupby(
        "ProbabilityMethod"
    )
    .agg(
        LogLoss_Mean=(
            "LogLoss",
            "mean",
        ),
        LogLoss_Std=(
            "LogLoss",
            "std",
        ),
        Brier_Mean=(
            "Brier",
            "mean",
        ),
        Brier_Std=(
            "Brier",
            "std",
        ),
        ECE_Mean=(
            "ECE",
            "mean",
        ),
        ECE_Std=(
            "ECE",
            "std",
        ),
    )
    .reset_index()
    .sort_values(
        "LogLoss_Mean",
        ascending=True,
    )
)

validation_calibration_summary_df = (
    validation_calibration_df
    .groupby(
        "ProbabilityMethod"
    )
    .agg(
        LogLoss_Mean=(
            "LogLoss",
            "mean",
        ),
        LogLoss_Std=(
            "LogLoss",
            "std",
        ),
        Brier_Mean=(
            "Brier",
            "mean",
        ),
        Brier_Std=(
            "Brier",
            "std",
        ),
        ECE_Mean=(
            "ECE",
            "mean",
        ),
        ECE_Std=(
            "ECE",
            "std",
        ),
    )
    .reset_index()
    .sort_values(
        "LogLoss_Mean",
        ascending=True,
    )
)


robust_selected_row = (
    selection_calibration_summary_df
    .iloc[0]
)

ROBUST_SELECTED_METHOD = (
    robust_selected_row[
        "ProbabilityMethod"
    ]
)

winner_count_df = (
    seed_winner_df[
        "SelectedProbabilityMethod"
    ]
    .value_counts()
    .rename_axis(
        "ProbabilityMethod"
    )
    .reset_index(
        name="WinCount"
    )
)

winner_count_df["WinRate"] = (
    winner_count_df[
        "WinCount"
    ]
    /
    len(SEEDS)
)


# ============================================================
# 26. CLASSIFICATION SUMMARY
# ============================================================

classification_summary_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "MacroF1",
            "C4Precision",
            "C4Recall",
            "C4F1",
            "PredC4",
        ],

        "Mean": [
            classification_df[
                "Accuracy"
            ].mean(),

            classification_df[
                "MacroF1"
            ].mean(),

            classification_df[
                "C4Precision"
            ].mean(),

            classification_df[
                "C4Recall"
            ].mean(),

            classification_df[
                "C4F1"
            ].mean(),

            classification_df[
                "PredC4"
            ].mean(),
        ],

        "Std": [
            classification_df[
                "Accuracy"
            ].std(),

            classification_df[
                "MacroF1"
            ].std(),

            classification_df[
                "C4Precision"
            ].std(),

            classification_df[
                "C4Recall"
            ].std(),

            classification_df[
                "C4F1"
            ].std(),

            classification_df[
                "PredC4"
            ].std(),
        ],
    }
)


# ============================================================
# 27. HORIZON / CAPACITY SUMMARY
# ============================================================

horizon_summary_df = (
    horizon_df
    .groupby(
        [
            "Split",
            "ProbabilityMethod",
            "Horizon",
        ]
    )
    .agg(
        Prevalence_Mean=(
            "Prevalence",
            "mean",
        ),

        AveragePrecision_Mean=(
            "AveragePrecision",
            "mean",
        ),

        AveragePrecision_Std=(
            "AveragePrecision",
            "std",
        ),

        ROCAUC_Mean=(
            "ROCAUC",
            "mean",
        ),

        ROCAUC_Std=(
            "ROCAUC",
            "std",
        ),
    )
    .reset_index()
)


capacity_summary_df = (
    capacity_df
    .groupby(
        [
            "Split",
            "ProbabilityMethod",
            "Horizon",
            "BudgetFraction",
            "BudgetPercent",
        ]
    )
    .agg(
        AlertCount_Mean=(
            "AlertCount",
            "mean",
        ),

        PrecisionAtK_Mean=(
            "PrecisionAtK",
            "mean",
        ),

        PrecisionAtK_Std=(
            "PrecisionAtK",
            "std",
        ),

        RecallAtK_Mean=(
            "RecallAtK",
            "mean",
        ),

        RecallAtK_Std=(
            "RecallAtK",
            "std",
        ),

        LiftAtK_Mean=(
            "LiftAtK",
            "mean",
        ),

        LiftAtK_Std=(
            "LiftAtK",
            "std",
        ),

        NumberNeededToInspect_Mean=(
            "NumberNeededToInspect",
            "mean",
        ),

        FalseAlertsAtK_Mean=(
            "FalseAlertsAtK",
            "mean",
        ),

        C4RecallAtK_Mean=(
            "C4RecallAtK",
            "mean",
        ),

        C4RecallAtK_Std=(
            "C4RecallAtK",
            "std",
        ),

        C3C4RecallAtK_Mean=(
            "C3C4RecallAtK",
            "mean",
        ),

        C3C4RecallAtK_Std=(
            "C3C4RecallAtK",
            "std",
        ),

        RiskThresholdAtK_Mean=(
            "RiskThresholdAtK",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# 28. SELECTED-METHOD RANKING VIEW
# ============================================================

selected_horizon_summary_df = (
    horizon_summary_df[
        horizon_summary_df[
            "ProbabilityMethod"
        ]
        ==
        ROBUST_SELECTED_METHOD
    ]
    .copy()
)

selected_capacity_summary_df = (
    capacity_summary_df[
        capacity_summary_df[
            "ProbabilityMethod"
        ]
        ==
        ROBUST_SELECTED_METHOD
    ]
    .copy()
)


# ============================================================
# 29. OLD 08c VS NEW 09c EXTERNAL BENCHMARK
# ============================================================

comparison_rows = []

new_external_horizon = (
    selected_horizon_summary_df[
        selected_horizon_summary_df[
            "Split"
        ]
        ==
        "OfficialValidation"
    ]
)

new_external_top5 = (
    selected_capacity_summary_df[
        (
            selected_capacity_summary_df[
                "Split"
            ]
            ==
            "OfficialValidation"
        )
        &
        (
            np.isclose(
                selected_capacity_summary_df[
                    "BudgetFraction"
                ],
                0.05,
            )
        )
    ]
)

for horizon in [
    "Risk48",
    "Risk24",
    "Risk12",
    "Risk6",
]:

    old = OLD_08C_EXTERNAL_RANKING[
        horizon
    ]

    new_horizon_row = (
        new_external_horizon[
            new_external_horizon[
                "Horizon"
            ]
            ==
            horizon
        ]
        .iloc[0]
    )

    new_top5_row = (
        new_external_top5[
            new_external_top5[
                "Horizon"
            ]
            ==
            horizon
        ]
        .iloc[0]
    )

    new_ap = float(
        new_horizon_row[
            "AveragePrecision_Mean"
        ]
    )

    new_auc = float(
        new_horizon_row[
            "ROCAUC_Mean"
        ]
    )

    new_c4_recall = float(
        new_top5_row[
            "C4RecallAtK_Mean"
        ]
    )

    comparison_rows.append(
        {
            "Horizon":
                horizon,

            "Old08c_AP":
                old[
                    "AveragePrecision"
                ],

            "New09c_AP":
                new_ap,

            "Delta_AP":
                (
                    new_ap
                    -
                    old[
                        "AveragePrecision"
                    ]
                ),

            "Old08c_ROCAUC":
                old[
                    "ROCAUC"
                ],

            "New09c_ROCAUC":
                new_auc,

            "Delta_ROCAUC":
                (
                    new_auc
                    -
                    old[
                        "ROCAUC"
                    ]
                ),

            "Old08c_Top5_C4Recall":
                old[
                    "Top5_C4Recall"
                ],

            "New09c_Top5_C4Recall":
                new_c4_recall,

            "Delta_Top5_C4Recall":
                (
                    new_c4_recall
                    -
                    old[
                        "Top5_C4Recall"
                    ]
                ),
        }
    )


old_vs_new_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# 30. PRINT KEY RESULTS
# ============================================================

print("\n" + "=" * 120)
print("09c CALIBRATION SELECTION — INTERNAL")
print("=" * 120)

print(
    selection_calibration_summary_df
    .round(6)
    .to_string(index=False)
)

print("\nPer-seed winners:")
print(
    seed_winner_df
    .round(6)
    .to_string(index=False)
)

print("\nRobust selected method:")
print(
    ROBUST_SELECTED_METHOD
)

print("\n" + "=" * 120)
print("09c OFFICIAL VALIDATION CALIBRATION — DIAGNOSTIC ONLY")
print("=" * 120)

print(
    validation_calibration_summary_df
    .round(6)
    .to_string(index=False)
)


print("\n" + "=" * 120)
print("09c RAW BASE CLASSIFICATION — 70% TRAIN PROTOCOL")
print("=" * 120)

print(
    classification_summary_df
    .round(6)
    .to_string(index=False)
)


print("\n" + "=" * 120)
print("09c SELECTED METHOD — EXTERNAL RANKING")
print("=" * 120)

print(
    new_external_horizon[
        [
            "Horizon",
            "Prevalence_Mean",
            "AveragePrecision_Mean",
            "AveragePrecision_Std",
            "ROCAUC_Mean",
            "ROCAUC_Std",
        ]
    ]
    .round(6)
    .to_string(index=False)
)


print("\n" + "=" * 120)
print("OLD 08c VS NEW 09c — EXTERNAL RANKING DELTAS")
print("=" * 120)

print(
    old_vs_new_df
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 31. SAVE OUTPUTS
# ============================================================

paths = {
    "split":
        OUTPUT_DIR
        /
        "model_09c_split_summary.csv",

    "classification_by_seed":
        OUTPUT_DIR
        /
        "model_09c_base_classification_by_seed.csv",

    "classification_summary":
        OUTPUT_DIR
        /
        "model_09c_base_classification_summary.csv",

    "selection_calibration_by_seed":
        OUTPUT_DIR
        /
        "model_09c_selection_calibration_by_seed.csv",

    "selection_calibration_summary":
        OUTPUT_DIR
        /
        "model_09c_selection_calibration_summary.csv",

    "validation_calibration_by_seed":
        OUTPUT_DIR
        /
        "model_09c_validation_calibration_by_seed.csv",

    "validation_calibration_summary":
        OUTPUT_DIR
        /
        "model_09c_validation_calibration_summary.csv",

    "seed_winners":
        OUTPUT_DIR
        /
        "model_09c_calibration_winners_by_seed.csv",

    "winner_counts":
        OUTPUT_DIR
        /
        "model_09c_calibration_win_counts.csv",

    "horizon_by_seed":
        OUTPUT_DIR
        /
        "model_09c_horizon_metrics_by_seed.csv",

    "horizon_summary":
        OUTPUT_DIR
        /
        "model_09c_horizon_metrics_summary.csv",

    "capacity_by_seed":
        OUTPUT_DIR
        /
        "model_09c_capacity_metrics_by_seed.csv",

    "capacity_summary":
        OUTPUT_DIR
        /
        "model_09c_capacity_metrics_summary.csv",

    "selected_horizon_summary":
        OUTPUT_DIR
        /
        "model_09c_selected_method_horizon_summary.csv",

    "selected_capacity_summary":
        OUTPUT_DIR
        /
        "model_09c_selected_method_capacity_summary.csv",

    "old_vs_new":
        OUTPUT_DIR
        /
        "model_09c_old08c_vs_new09c_external_ranking.csv",

    "selected_method":
        OUTPUT_DIR
        /
        "model_09c_selected_probability_method.txt",

    "interpretation":
        OUTPUT_DIR
        /
        "model_09c_interpretation_notes.txt",
}


split_df.to_csv(
    paths["split"],
    index=False,
)

classification_df.to_csv(
    paths["classification_by_seed"],
    index=False,
)

classification_summary_df.to_csv(
    paths["classification_summary"],
    index=False,
)

selection_calibration_df.to_csv(
    paths["selection_calibration_by_seed"],
    index=False,
)

selection_calibration_summary_df.to_csv(
    paths["selection_calibration_summary"],
    index=False,
)

validation_calibration_df.to_csv(
    paths["validation_calibration_by_seed"],
    index=False,
)

validation_calibration_summary_df.to_csv(
    paths["validation_calibration_summary"],
    index=False,
)

seed_winner_df.to_csv(
    paths["seed_winners"],
    index=False,
)

winner_count_df.to_csv(
    paths["winner_counts"],
    index=False,
)

horizon_df.to_csv(
    paths["horizon_by_seed"],
    index=False,
)

horizon_summary_df.to_csv(
    paths["horizon_summary"],
    index=False,
)

capacity_df.to_csv(
    paths["capacity_by_seed"],
    index=False,
)

capacity_summary_df.to_csv(
    paths["capacity_summary"],
    index=False,
)

selected_horizon_summary_df.to_csv(
    paths["selected_horizon_summary"],
    index=False,
)

selected_capacity_summary_df.to_csv(
    paths["selected_capacity_summary"],
    index=False,
)

old_vs_new_df.to_csv(
    paths["old_vs_new"],
    index=False,
)

paths["selected_method"].write_text(
    (
        "FleetCare AI — Test 09c\n\n"
        "Base model:\n"
        "Vehicle-Aware + Multi-Readout History "
        "+ Multiclass XGBoost + No Class Weighting\n\n"
        "Calibration selection rule:\n"
        "Lowest mean Selection-Dev LogLoss across 5 seeds.\n\n"
        f"Selected method: {ROBUST_SELECTED_METHOD}\n"
        f"Mean Selection LogLoss: "
        f"{float(robust_selected_row['LogLoss_Mean']):.8f}\n"
        f"Std Selection LogLoss: "
        f"{float(robust_selected_row['LogLoss_Std']):.8f}\n\n"
        "Official validation was NOT used to choose the "
        "calibration method.\n"
    ),
    encoding="utf-8",
)

paths["interpretation"].write_text(
    (
        "FleetCare AI — Test 09c Interpretation Guide\n\n"
        "Purpose:\n"
        "Re-run the c-track robust calibration and ranking "
        "protocol using teammate's latest multi-history "
        "no-weight base model.\n\n"
        "Primary questions:\n"
        "1. Does the selected calibrator remain LogitCalibrated "
        "after removing class weights and adding longer history?\n"
        "2. Does external calibration improve?\n"
        "3. Does external Risk48/24/12/6 ranking improve versus "
        "Test 08c?\n"
        "4. Does Top-5% C4 Recall improve consistently?\n"
        "5. Is the internal-to-external ranking gap reduced?\n\n"
        "Do not select a final horizon or Top-K budget from "
        "official validation alone.\n\n"
        "If external ranking improves materially and consistently, "
        "freeze the latest base + calibration and proceed to "
        "operational policy design / robustness.\n\n"
        "If external ranking remains weak, next test should be "
        "direct horizon-specific binary modeling with "
        "horizon-specific censoring (Risk48/Risk24/Risk12/Risk6).\n"
    ),
    encoding="utf-8",
)


# ============================================================
# 32. FINISH
# ============================================================

print("\n" + "=" * 120)
print("TEST 09c COMPLETE")
print("=" * 120)

print("\nRobust selected calibration:")
print(ROBUST_SELECTED_METHOD)

print("\nPrimary files to inspect:")
for key in [
    "selection_calibration_summary",
    "validation_calibration_summary",
    "selected_horizon_summary",
    "selected_capacity_summary",
    "old_vs_new",
]:
    print(" ", paths[key])

print(
    "\nImportant: the raw base classification in this script "
    "uses only 70% of official training vehicles, so it should "
    "NOT be expected to exactly reproduce teammate's full-data "
    "78.93% / 0.2031 result."
)
