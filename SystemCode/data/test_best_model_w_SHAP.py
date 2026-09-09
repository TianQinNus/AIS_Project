# ============================================================
# FleetCare AI - FINAL MODEL
#
# Vehicle-Aware Sampling
# + Multi-Readout Historical Features
# + Multiclass XGBoost
# + No Additional Class Weighting
# + Vehicle-Level OOF Logit Calibration
# + Official Validation Evaluation
# + Final Model Saving
#
# Target:
# C0 = >48 operational time units before repair
# C1 = 24-48
# C2 = 12-24
# C3 = 6-12
# C4 = 0-6
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import gc
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score,
    roc_auc_score,
    log_loss,
)

from xgboost import XGBClassifier


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_CLASSES = 5
N_CALIBRATION_FOLDS = 5

EPSILON = 1e-6

CLASS_IDS = [0, 1, 2, 3, 4]

CLASS_NAMES = [
    "C0",
    "C1",
    "C2",
    "C3",
    "C4"
]

HORIZONS = {

    "Risk48": [1, 2, 3, 4],

    "Risk24": [2, 3, 4],

    "Risk12": [3, 4],

    "Risk6": [4],
}


# ============================================================
# 3. PATHS
# ============================================================

DATA_DIR = Path("../data/raw")

OUTPUT_DIR = Path("../data/processed")

MODEL_DIR = (
    OUTPUT_DIR
    / "final_model"
)

EVALUATION_DIR = (
    OUTPUT_DIR
    / "final_evaluation"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_OP_PATH = (
    DATA_DIR
    / "train_operational_readouts.csv"
)

TRAIN_SPEC_PATH = (
    DATA_DIR
    / "train_specifications.csv"
)

TRAIN_TTE_PATH = (
    DATA_DIR
    / "train_tte.csv"
)

VAL_OP_PATH = (
    DATA_DIR
    / "validation_operational_readouts.csv"
)

VAL_SPEC_PATH = (
    DATA_DIR
    / "validation_specifications.csv"
)

VAL_LABEL_PATH = (
    DATA_DIR
    / "validation_labels.csv"
)


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


print("=" * 80)
print("FLEETCARE AI - FINAL CALIBRATED MODEL")
print("=" * 80)


for path in required_files:

    print(
        f"{path.name}: {path.exists()}"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: {path}"
        )


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)


train_op = pd.read_csv(
    TRAIN_OP_PATH
)

train_spec = pd.read_csv(
    TRAIN_SPEC_PATH
)

train_tte = pd.read_csv(
    TRAIN_TTE_PATH
)

val_op = pd.read_csv(
    VAL_OP_PATH
)

val_spec = pd.read_csv(
    VAL_SPEC_PATH
)

val_labels = pd.read_csv(
    VAL_LABEL_PATH
)


print(
    "Train operational:",
    train_op.shape
)

print(
    "Train specifications:",
    train_spec.shape
)

print(
    "Train TTE:",
    train_tte.shape
)

print(
    "Validation operational:",
    val_op.shape
)

print(
    "Validation specifications:",
    val_spec.shape
)

print(
    "Validation labels:",
    val_labels.shape
)


# ============================================================
# 6. IDENTIFY RAW OPERATIONAL FEATURES
# ============================================================

RAW_FEATURE_COLS = [

    col

    for col in train_op.columns

    if col not in [

        "vehicle_id",

        "time_step"
    ]
]


print(
    "\nRaw operational features:",
    len(RAW_FEATURE_COLS)
)


# ============================================================
# 7. MULTI-READOUT HISTORICAL FEATURE ENGINEERING
# ============================================================

def create_history_features(
    df,
    feature_cols
):

    print(
        "  Sorting vehicle history..."
    )

    df = (

        df

        .sort_values(
            [
                "vehicle_id",
                "time_step"
            ]
        )

        .reset_index(
            drop=True
        )

        .copy()
    )


    grouped = df.groupby(
        "vehicle_id",
        sort=False
    )


    # --------------------------------------------------------
    # Time gap
    # --------------------------------------------------------

    print(
        "  Creating time gap..."
    )


    df["time_gap"] = (

        grouped[
            "time_step"
        ]

        .diff()

        .astype(
            np.float32
        )
    )


    safe_gap = (

        df[
            "time_gap"
        ]

        .replace(
            0,
            np.nan
        )
    )


    # --------------------------------------------------------
    # Previous values
    # --------------------------------------------------------

    print(
        "  Creating previous values..."
    )


    previous = (

        grouped[
            feature_cols
        ]

        .shift(1)

        .astype(
            np.float32
        )
    )


    previous.columns = [

        f"{col}__prev"

        for col in feature_cols
    ]


    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    current_array = (

        df[
            feature_cols
        ]

        .to_numpy(
            dtype=np.float32
        )
    )


    previous_array = (

        previous

        .to_numpy(
            dtype=np.float32
        )
    )


    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    print(
        "  Creating delta..."
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
        ]
    )


    # --------------------------------------------------------
    # Rate
    # --------------------------------------------------------

    print(
        "  Creating rate..."
    )


    gap_array = (

        safe_gap

        .to_numpy(
            dtype=np.float32
        )

        [:, None]
    )


    with np.errstate(
        divide="ignore",
        invalid="ignore"
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
        ]
    )


    rate.replace(

        [
            np.inf,
            -np.inf
        ],

        np.nan,

        inplace=True
    )


    # --------------------------------------------------------
    # Change from 3 readouts ago
    # --------------------------------------------------------

    print(
        "  Creating change3..."
    )


    lag3 = (

        grouped[
            feature_cols
        ]

        .shift(3)

        .astype(
            np.float32
        )
    )


    change3_array = (

        current_array

        -

        lag3.to_numpy(
            dtype=np.float32
        )
    )


    change3 = pd.DataFrame(

        change3_array,

        index=df.index,

        columns=[

            f"{col}__change3"

            for col in feature_cols
        ]
    )


    del lag3
    del change3_array

    gc.collect()


    # --------------------------------------------------------
    # Change from 5 readouts ago
    # --------------------------------------------------------

    print(
        "  Creating change5..."
    )


    lag5 = (

        grouped[
            feature_cols
        ]

        .shift(5)

        .astype(
            np.float32
        )
    )


    change5_array = (

        current_array

        -

        lag5.to_numpy(
            dtype=np.float32
        )
    )


    change5 = pd.DataFrame(

        change5_array,

        index=df.index,

        columns=[

            f"{col}__change5"

            for col in feature_cols
        ]
    )


    del lag5
    del change5_array

    gc.collect()


    # --------------------------------------------------------
    # Rolling mean 3
    # --------------------------------------------------------

    print(
        "  Creating rolling mean 3..."
    )


    rolling_mean3 = (

        grouped[
            feature_cols
        ]

        .rolling(
            window=3,
            min_periods=1
        )

        .mean()

        .reset_index(
            level=0,
            drop=True
        )

        .reindex(
            df.index
        )

        .astype(
            np.float32
        )
    )


    rolling_mean3.columns = [

        f"{col}__mean3"

        for col in feature_cols
    ]


    # --------------------------------------------------------
    # Rolling std 3
    # --------------------------------------------------------

    print(
        "  Creating rolling std 3..."
    )


    rolling_std3 = (

        grouped[
            feature_cols
        ]

        .rolling(
            window=3,
            min_periods=2
        )

        .std()

        .reset_index(
            level=0,
            drop=True
        )

        .reindex(
            df.index
        )

        .astype(
            np.float32
        )
    )


    rolling_std3.columns = [

        f"{col}__std3"

        for col in feature_cols
    ]


    # --------------------------------------------------------
    # Rolling mean 5
    # --------------------------------------------------------

    print(
        "  Creating rolling mean 5..."
    )


    rolling_mean5 = (

        grouped[
            feature_cols
        ]

        .rolling(
            window=5,
            min_periods=1
        )

        .mean()

        .reset_index(
            level=0,
            drop=True
        )

        .reindex(
            df.index
        )

        .astype(
            np.float32
        )
    )


    rolling_mean5.columns = [

        f"{col}__mean5"

        for col in feature_cols
    ]


    # --------------------------------------------------------
    # Rolling std 5
    # --------------------------------------------------------

    print(
        "  Creating rolling std 5..."
    )


    rolling_std5 = (

        grouped[
            feature_cols
        ]

        .rolling(
            window=5,
            min_periods=2
        )

        .std()

        .reset_index(
            level=0,
            drop=True
        )

        .reindex(
            df.index
        )

        .astype(
            np.float32
        )
    )


    rolling_std5.columns = [

        f"{col}__std5"

        for col in feature_cols
    ]


    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    print(
        "  Combining historical features..."
    )


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

            rolling_std5
        ],

        axis=1
    )


    result.replace(

        [
            np.inf,
            -np.inf
        ],

        np.nan,

        inplace=True
    )


    print(
        "  Historical feature shape:",
        result.shape
    )


    return result


# ============================================================
# 8. CREATE TRAIN / VALIDATION HISTORY
# ============================================================

print("\n" + "=" * 80)
print("CREATING TRAIN HISTORY FEATURES")
print("=" * 80)


train_history = create_history_features(
    train_op,
    RAW_FEATURE_COLS
)


print("\n" + "=" * 80)
print("CREATING VALIDATION HISTORY FEATURES")
print("=" * 80)


val_history = create_history_features(
    val_op,
    RAW_FEATURE_COLS
)


del train_op
del val_op

gc.collect()


# ============================================================
# 9. CREATE CENSORING-AWARE TRAINING LABELS
# ============================================================

train_full = (

    train_history

    .merge(
        train_tte,
        on="vehicle_id",
        how="left",
        validate="many_to_one"
    )
)


train_full["time_to_endpoint"] = (

    train_full[
        "length_of_study_time_step"
    ]

    -

    train_full[
        "time_step"
    ]
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


# Repair vehicles

repair_mask = (

    train_full[
        "in_study_repair"
    ]

    == 1
)


train_full.loc[

    repair_mask,

    "class_label"

] = (

    train_full.loc[
        repair_mask,
        "time_to_endpoint"
    ]

    .apply(
        tte_to_class
    )
)


# Non-repair vehicles:
# Only safely label C0 when >48 repair-free units remain

safe_non_repair_mask = (

    (
        train_full[
            "in_study_repair"
        ]

        == 0
    )

    &

    (
        train_full[
            "time_to_endpoint"
        ]

        > 48
    )
)


train_full.loc[

    safe_non_repair_mask,

    "class_label"

] = 0


candidate_train = (

    train_full

    .dropna(
        subset=[
            "class_label"
        ]
    )

    .copy()
)


candidate_train["class_label"] = (

    candidate_train[
        "class_label"
    ]

    .astype(int)
)


print("\nEligible training observations:")

print(

    candidate_train[
        "class_label"
    ]

    .value_counts()

    .sort_index()
)


del train_full
del train_history

gc.collect()


# ============================================================
# 10. VEHICLE-AWARE SAMPLING
# ============================================================

print("\n" + "=" * 80)
print("VEHICLE-AWARE SAMPLING")
print("=" * 80)


repair_candidates = (

    candidate_train[

        candidate_train[
            "in_study_repair"
        ]

        == 1
    ]

    .copy()
)


non_repair_candidates = (

    candidate_train[

        candidate_train[
            "in_study_repair"
        ]

        == 0
    ]

    .copy()
)


# One observation per repaired vehicle per available class

repair_sampled = (

    repair_candidates

    .groupby(
        [
            "vehicle_id",
            "class_label"
        ],
        group_keys=False
    )

    .sample(
        n=1,
        random_state=RANDOM_STATE
    )

    .reset_index(
        drop=True
    )
)


# One safe C0 observation per non-repair vehicle

non_repair_sampled = (

    non_repair_candidates

    .groupby(
        "vehicle_id",
        group_keys=False
    )

    .sample(
        n=1,
        random_state=RANDOM_STATE
    )

    .reset_index(
        drop=True
    )
)


train_sampled = pd.concat(

    [
        repair_sampled,
        non_repair_sampled
    ],

    ignore_index=True
)


train_sampled = (

    train_sampled

    .sample(
        frac=1,
        random_state=RANDOM_STATE
    )

    .reset_index(
        drop=True
    )
)


print(
    "Training rows:",
    len(train_sampled)
)


print(
    "Unique training vehicles:",
    train_sampled[
        "vehicle_id"
    ].nunique()
)


print(
    "\nVehicle-aware class distribution:"
)


print(

    train_sampled[
        "class_label"
    ]

    .value_counts()

    .sort_index()
)


del candidate_train
del repair_candidates
del non_repair_candidates
del repair_sampled
del non_repair_sampled

gc.collect()


# ============================================================
# 11. ADD VEHICLE SPECIFICATIONS
# ============================================================

train_sampled = (

    train_sampled

    .merge(
        train_spec,
        on="vehicle_id",
        how="left",
        validate="many_to_one"
    )
)


# ============================================================
# 12. BUILD OFFICIAL VALIDATION TABLE
# ============================================================

val_last = (

    val_history

    .groupby(
        "vehicle_id",
        sort=False
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
        validate="one_to_one"
    )
)


val_last = (

    val_last

    .merge(
        val_labels,
        on="vehicle_id",
        how="left",
        validate="one_to_one"
    )
)


del val_history

gc.collect()


print(
    "\nOfficial Validation distribution:"
)


print(

    val_last[
        "class_label"
    ]

    .value_counts()

    .sort_index()
)


# ============================================================
# 13. BUILD X / Y
# ============================================================

NON_FEATURE_COLS = [

    "vehicle_id",

    "class_label",

    "length_of_study_time_step",

    "in_study_repair",

    "time_to_endpoint"
]


X_train = (

    train_sampled

    .drop(
        columns=NON_FEATURE_COLS,
        errors="ignore"
    )
)


y_train = (

    train_sampled[
        "class_label"
    ]

    .astype(int)

    .copy()
)


X_val = (

    val_last

    .drop(
        columns=[
            "vehicle_id",
            "class_label"
        ],
        errors="ignore"
    )
)


y_val = (

    val_last[
        "class_label"
    ]

    .astype(int)

    .copy()
)


# Make validation columns exactly match training

X_val = X_val.reindex(
    columns=X_train.columns
)


assert (

    list(
        X_train.columns
    )

    ==

    list(
        X_val.columns
    )
)


# ============================================================
# 14. FEATURE TYPES / PREPROCESSING
# ============================================================

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


numeric_transformer = Pipeline([

    (

        "imputer",

        SimpleImputer(
            strategy="median"
        )
    )
])


categorical_transformer = Pipeline([

    (

        "imputer",

        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (

        "onehot",

        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


preprocessor = ColumnTransformer([

    (

        "numeric",

        numeric_transformer,

        numeric_cols
    ),

    (

        "categorical",

        categorical_transformer,

        categorical_cols
    )
])


# ============================================================
# 15. XGBOOST
# ============================================================

history_xgb = XGBClassifier(

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

    random_state=RANDOM_STATE,

    n_jobs=-1
)


history_pipeline = Pipeline([

    (

        "preprocessor",

        preprocessor
    ),

    (

        "classifier",

        history_xgb
    )
])


# ============================================================
# 16. CREATE VEHICLE-LEVEL OOF PROBABILITIES
# ============================================================
#
# Calibration is fitted using OOF predictions.
#
# IMPORTANT:
# GroupKFold uses vehicle_id.
# Therefore observations from one vehicle never appear
# simultaneously in fold training and fold holdout.
#
# Official Validation is NEVER used here.
# ============================================================

print("\n" + "=" * 80)
print("CREATING OOF PROBABILITIES FOR CALIBRATION")
print("=" * 80)


groups = (

    train_sampled[
        "vehicle_id"
    ]

    .to_numpy()
)


y_train_array = (

    y_train

    .to_numpy()
)


oof_raw_proba = np.zeros(

    (
        len(X_train),
        N_CLASSES
    ),

    dtype=np.float64
)


oof_filled = np.zeros(

    len(X_train),

    dtype=bool
)


group_kfold = GroupKFold(
    n_splits=N_CALIBRATION_FOLDS
)


for fold, (
    fit_idx,
    holdout_idx
) in enumerate(

    group_kfold.split(

        X_train,

        y_train_array,

        groups=groups
    ),

    start=1
):

    print(
        f"OOF fold "
        f"{fold}/{N_CALIBRATION_FOLDS}"
    )


    fold_pipeline = clone(
        history_pipeline
    )


    X_fold_train = (

        X_train

        .iloc[
            fit_idx
        ]
    )


    y_fold_train = (

        y_train

        .iloc[
            fit_idx
        ]
    )


    X_fold_holdout = (

        X_train

        .iloc[
            holdout_idx
        ]
    )


    fold_pipeline.fit(

        X_fold_train,

        y_fold_train
    )


    fold_proba = (

        fold_pipeline

        .predict_proba(
            X_fold_holdout
        )
    )


    aligned_proba = np.zeros(

        (
            len(holdout_idx),
            N_CLASSES
        ),

        dtype=np.float64
    )


    fold_classes = (

        fold_pipeline

        .named_steps[
            "classifier"
        ]

        .classes_

        .astype(int)
    )


    aligned_proba[
        :,
        fold_classes
    ] = fold_proba


    oof_raw_proba[
        holdout_idx
    ] = aligned_proba


    oof_filled[
        holdout_idx
    ] = True


    del fold_pipeline

    gc.collect()


assert oof_filled.all()


print(
    "✓ OOF probabilities created:",
    oof_raw_proba.shape
)


# ============================================================
# 17. CALIBRATION FUNCTIONS
# ============================================================

def normalize_probabilities(
    probabilities
):

    probabilities = np.asarray(

        probabilities,

        dtype=np.float64
    )


    probabilities = np.clip(

        probabilities,

        EPSILON,

        1.0
    )


    probabilities = (

        probabilities

        /

        probabilities.sum(
            axis=1,
            keepdims=True
        )
    )


    return probabilities


def logit_features(
    probabilities
):

    probabilities = (

        normalize_probabilities(
            probabilities
        )
    )


    return np.log(

        np.clip(

            probabilities,

            EPSILON,

            1.0
        )
    )


def calibrated_proba(

    calibrator,

    raw_proba
):

    predicted = (

        calibrator

        .predict_proba(

            logit_features(
                raw_proba
            )
        )
    )


    aligned = np.zeros(

        (
            len(predicted),
            N_CLASSES
        ),

        dtype=np.float64
    )


    aligned[
        :,
        calibrator.classes_.astype(int)
    ] = predicted


    return normalize_probabilities(
        aligned
    )


# ============================================================
# 18. FIT LOGIT CALIBRATOR
# ============================================================

print("\n" + "=" * 80)
print("FITTING LOGIT CALIBRATOR")
print("=" * 80)


oof_raw_proba = (

    normalize_probabilities(
        oof_raw_proba
    )
)


calibrator = LogisticRegression(

    solver="lbfgs",

    max_iter=2000,

    random_state=RANDOM_STATE
)


calibrator.fit(

    logit_features(
        oof_raw_proba
    ),

    y_train_array
)


print(
    "✓ Calibrator fitted"
)


print(
    "Calibrator classes:",
    calibrator.classes_
)


# ============================================================
# 19. TRAIN FINAL MODEL ON ALL VEHICLE-AWARE TRAINING DATA
# ============================================================

print("\n" + "=" * 80)
print("TRAINING FINAL HISTORY XGBOOST")
print("=" * 80)


history_pipeline.fit(

    X_train,

    y_train
)


print(
    "✓ Final model training complete"
)


# ============================================================
# 20. OFFICIAL VALIDATION - RAW + CALIBRATED
# ============================================================

print("\n" + "=" * 80)
print("OFFICIAL VALIDATION")
print("=" * 80)


raw_y_proba = (

    history_pipeline

    .predict_proba(
        X_val
    )
)


raw_y_proba = (

    normalize_probabilities(
        raw_y_proba
    )
)


calibrated_y_proba = (

    calibrated_proba(

        calibrator,

        raw_y_proba
    )
)


raw_y_pred = np.argmax(

    raw_y_proba,

    axis=1
)


calibrated_y_pred = np.argmax(

    calibrated_y_proba,

    axis=1
)


# ============================================================
# 21. EVALUATION FUNCTION
# ============================================================

def evaluate_classification(

    name,

    y_true,

    y_pred,

    probabilities
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    macro_precision = precision_score(

        y_true,

        y_pred,

        labels=CLASS_IDS,

        average="macro",

        zero_division=0
    )


    macro_recall = recall_score(

        y_true,

        y_pred,

        labels=CLASS_IDS,

        average="macro",

        zero_division=0
    )


    macro_f1 = f1_score(

        y_true,

        y_pred,

        labels=CLASS_IDS,

        average="macro",

        zero_division=0
    )


    weighted_f1 = f1_score(

        y_true,

        y_pred,

        labels=CLASS_IDS,

        average="weighted",

        zero_division=0
    )


    loss = log_loss(

        y_true,

        probabilities,

        labels=CLASS_IDS
    )


    report = classification_report(

        y_true,

        y_pred,

        labels=CLASS_IDS,

        target_names=CLASS_NAMES,

        digits=4,

        zero_division=0
    )


    print(
        "\n" + "=" * 80
    )

    print(
        name
    )

    print(
        "=" * 80
    )

    print(report)


    print(
        f"Accuracy:        "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall:    "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1:        "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1:     "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Log Loss:        "
        f"{loss:.4f}"
    )


    return {

        "accuracy":
            float(accuracy),

        "macro_precision":
            float(macro_precision),

        "macro_recall":
            float(macro_recall),

        "macro_f1":
            float(macro_f1),

        "weighted_f1":
            float(weighted_f1),

        "log_loss":
            float(loss)
    }


# ============================================================
# 22. RAW EVALUATION
# ============================================================

raw_metrics = evaluate_classification(

    "RAW MODEL",

    y_val,

    raw_y_pred,

    raw_y_proba
)


# ============================================================
# 23. CALIBRATED EVALUATION
# ============================================================

calibrated_metrics = evaluate_classification(

    "CALIBRATED MODEL",

    y_val,

    calibrated_y_pred,

    calibrated_y_proba
)


# ============================================================
# 24. RISK EVALUATION
# ============================================================

def evaluate_risk_horizons(

    labels,

    probabilities
):

    labels = np.asarray(
        labels
    )


    rows = []


    for (
        horizon,
        positive_classes
    ) in HORIZONS.items():


        binary = np.isin(

            labels,

            positive_classes
        ).astype(int)


        score = (

            probabilities[
                :,
                positive_classes
            ]

            .sum(
                axis=1
            )
        )


        ap = average_precision_score(

            binary,

            score
        )


        auc = roc_auc_score(

            binary,

            score
        )


        # Top 5%

        k5 = max(

            1,

            int(
                np.ceil(
                    len(labels)
                    * 0.05
                )
            )
        )


        top5_idx = np.argsort(

            -score,

            kind="stable"

        )[:k5]


        top5_recall = (

            binary[
                top5_idx
            ].sum()

            /

            binary.sum()
        )


        top5_precision = (

            binary[
                top5_idx
            ].mean()
        )


        # Top 10%

        k10 = max(

            1,

            int(
                np.ceil(
                    len(labels)
                    * 0.10
                )
            )
        )


        top10_idx = np.argsort(

            -score,

            kind="stable"

        )[:k10]


        top10_recall = (

            binary[
                top10_idx
            ].sum()

            /

            binary.sum()
        )


        top10_precision = (

            binary[
                top10_idx
            ].mean()
        )


        rows.append({

            "Horizon":
                horizon,

            "AveragePrecision":
                float(ap),

            "ROCAUC":
                float(auc),

            "Top5Recall":
                float(top5_recall),

            "Top5Precision":
                float(top5_precision),

            "Top10Recall":
                float(top10_recall),

            "Top10Precision":
                float(top10_precision)
        })


    return pd.DataFrame(
        rows
    )


raw_risk_metrics = (

    evaluate_risk_horizons(

        y_val,

        raw_y_proba
    )
)


calibrated_risk_metrics = (

    evaluate_risk_horizons(

        y_val,

        calibrated_y_proba
    )
)


print("\n" + "=" * 80)
print("RAW RISK METRICS")
print("=" * 80)

print(

    raw_risk_metrics

    .to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("CALIBRATED RISK METRICS")
print("=" * 80)

print(

    calibrated_risk_metrics

    .to_string(
        index=False
    )
)


raw_risk_metrics.to_csv(

    EVALUATION_DIR
    / "raw_risk_metrics.csv",

    index=False
)


calibrated_risk_metrics.to_csv(

    EVALUATION_DIR
    / "calibrated_risk_metrics.csv",

    index=False
)


# ============================================================
# 25. CONFUSION MATRICES
# ============================================================

for (
    name,
    predictions
) in [

    (
        "raw",
        raw_y_pred
    ),

    (
        "calibrated",
        calibrated_y_pred
    )
]:

    cm = confusion_matrix(

        y_val,

        predictions,

        labels=CLASS_IDS
    )


    print(
        f"\n{name.upper()} CONFUSION MATRIX"
    )

    print(cm)


    fig, ax = plt.subplots(
        figsize=(8, 7)
    )


    display = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=CLASS_NAMES
    )


    display.plot(

        ax=ax,

        values_format="d",

        colorbar=False
    )


    ax.set_title(

        f"FleetCare AI - "
        f"{name.capitalize()} Confusion Matrix"
    )


    plt.tight_layout()


    plt.savefig(

        EVALUATION_DIR
        / f"{name}_confusion_matrix.png",

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 26. SAVE VALIDATION PREDICTIONS
# ============================================================

validation_predictions = pd.DataFrame({

    "vehicle_id":
        val_last[
            "vehicle_id"
        ].to_numpy(),

    "actual_class":
        y_val.to_numpy(),

    "raw_predicted_class":
        raw_y_pred,

    "calibrated_predicted_class":
        calibrated_y_pred,

    "raw_P_C0":
        raw_y_proba[:, 0],

    "raw_P_C1":
        raw_y_proba[:, 1],

    "raw_P_C2":
        raw_y_proba[:, 2],

    "raw_P_C3":
        raw_y_proba[:, 3],

    "raw_P_C4":
        raw_y_proba[:, 4],

    "P_C0":
        calibrated_y_proba[:, 0],

    "P_C1":
        calibrated_y_proba[:, 1],

    "P_C2":
        calibrated_y_proba[:, 2],

    "P_C3":
        calibrated_y_proba[:, 3],

    "P_C4":
        calibrated_y_proba[:, 4],

    "Risk48":
        calibrated_y_proba[
            :,
            1:
        ].sum(
            axis=1
        ),

    "Risk24":
        calibrated_y_proba[
            :,
            2:
        ].sum(
            axis=1
        ),

    "Risk12":
        calibrated_y_proba[
            :,
            3:
        ].sum(
            axis=1
        ),

    "Risk6":
        calibrated_y_proba[
            :,
            4
        ]
})


validation_predictions.to_csv(

    EVALUATION_DIR
    / "final_validation_predictions.csv",

    index=False
)


# ============================================================
# 27. SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 80)
print("SAVING FINAL MODEL")
print("=" * 80)


MODEL_PATH = (

    MODEL_DIR

    / "fleetcare_vehicle_aware_history_xgb.joblib"
)


CALIBRATOR_PATH = (

    MODEL_DIR

    / "fleetcare_logit_calibrator.joblib"
)


FEATURE_PATH = (

    MODEL_DIR

    / "fleetcare_history_feature_columns.joblib"
)


RAW_FEATURE_PATH = (

    MODEL_DIR

    / "fleetcare_raw_feature_columns.joblib"
)


METADATA_PATH = (

    MODEL_DIR

    / "fleetcare_history_model_metadata.json"
)


joblib.dump(

    history_pipeline,

    MODEL_PATH
)


joblib.dump(

    calibrator,

    CALIBRATOR_PATH
)


joblib.dump(

    list(
        X_train.columns
    ),

    FEATURE_PATH
)


joblib.dump(

    RAW_FEATURE_COLS,

    RAW_FEATURE_PATH
)


# ============================================================
# 28. METADATA
# ============================================================

metadata = {

    "model_name":
        "FleetCare Vehicle-Aware Multi-History XGBoost",

    "model_type":
        "XGBClassifier",

    "feature_representation":
        "Multi-readout historical features",

    "sampling":
        "Vehicle-aware",

    "class_weighting":
        "None",

    "calibration":
        "5-fold vehicle-level OOF multinomial logit calibration",

    "random_state":
        RANDOM_STATE,

    "num_classes":
        N_CLASSES,

    "xgboost_parameters": {

        "n_estimators":
            150,

        "learning_rate":
            0.08,

        "max_depth":
            5,

        "min_child_weight":
            3,

        "subsample":
            0.8,

        "colsample_bytree":
            0.8,

        "reg_alpha":
            0.0,

        "reg_lambda":
            1.0,

        "objective":
            "multi:softprob"
    },

    "raw_validation_metrics":
        raw_metrics,

    "calibrated_validation_metrics":
        calibrated_metrics,

    "class_definition": {

        "C0":
            ">48",

        "C1":
            "24-48",

        "C2":
            "12-24",

        "C3":
            "6-12",

        "C4":
            "0-6"
    },

    "risk_definition": {

        "Risk48":
            "P(C1)+P(C2)+P(C3)+P(C4)",

        "Risk24":
            "P(C2)+P(C3)+P(C4)",

        "Risk12":
            "P(C3)+P(C4)",

        "Risk6":
            "P(C4)"
    }
}


with open(

    METADATA_PATH,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        metadata,

        f,

        indent=2
    )


# ============================================================
# 29. VERIFY SAVED MODEL + CALIBRATOR
# ============================================================

print("\n" + "=" * 80)
print("VERIFYING SAVED MODEL")
print("=" * 80)


loaded_model = joblib.load(
    MODEL_PATH
)


loaded_calibrator = joblib.load(
    CALIBRATOR_PATH
)


loaded_features = joblib.load(
    FEATURE_PATH
)


X_verify = (

    X_val

    .iloc[:20]

    .reindex(
        columns=loaded_features
    )
)


verify_raw = (

    loaded_model

    .predict_proba(
        X_verify
    )
)


verify_calibrated = (

    calibrated_proba(

        loaded_calibrator,

        verify_raw
    )
)


original_calibrated = (

    calibrated_y_proba[
        :20
    ]
)


assert np.allclose(

    verify_calibrated,

    original_calibrated,

    atol=1e-10
)


assert np.allclose(

    verify_calibrated.sum(
        axis=1
    ),

    1.0
)


print(
    "✓ Saved model + calibrator verification PASSED."
)


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL SAVED")
print("=" * 80)


print(
    "\nModel:",
    MODEL_PATH
)


print(
    "Calibrator:",
    CALIBRATOR_PATH
)


print(
    "Feature columns:",
    FEATURE_PATH
)


print(
    "Raw feature columns:",
    RAW_FEATURE_PATH
)


print(
    "Metadata:",
    METADATA_PATH
)


print(
    "\nRaw Macro-F1:",
    round(
        raw_metrics[
            "macro_f1"
        ],
        4
    )
)


print(
    "Calibrated Macro-F1:",
    round(
        calibrated_metrics[
            "macro_f1"
        ],
        4
    )
)


print(
    "Raw Log Loss:",
    round(
        raw_metrics[
            "log_loss"
        ],
        4
    )
)


print(
    "Calibrated Log Loss:",
    round(
        calibrated_metrics[
            "log_loss"
        ],
        4
    )
)


print("\nRaw Risk Metrics:")

print(

    raw_risk_metrics

    .round(4)

    .to_string(
        index=False
    )
)


print("\nCalibrated Risk Metrics:")

print(

    calibrated_risk_metrics

    .round(4)

    .to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("FLEETCARE FINAL PIPELINE COMPLETE")
print("=" * 80)