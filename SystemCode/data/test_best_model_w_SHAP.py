# ============================================================
# FleetCare AI
# Final Candidate Model
#
# Vehicle-Aware Sampling
# + Multi-Readout Historical Features
# + Multiclass XGBoost
# + No Class Weighting
# + SHAP Explainability
#
# Target:
# C0 = >48 time units before repair
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

SHAP_SAMPLE_SIZE = 300

pd.set_option(
    "display.max_columns",
    200
)

pd.set_option(
    "display.width",
    250
)


# ============================================================
# 3. PATHS
# ============================================================
#
# Adjust DATA_DIR if necessary.
#
# This assumes the script is executed from:
#
# AIS_Project/SystemCode/data/
#
# and the dataset is:
#
# AIS_Project/SystemCode/data/raw/
#
# If your existing ../data/raw path works, keep this.
# ============================================================

DATA_DIR = Path("../data/raw")

OUTPUT_DIR = Path(
    "../data/processed"
)

SHAP_OUTPUT_DIR = (
    OUTPUT_DIR
    /
    "shap"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SHAP_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_OP_PATH = (
    DATA_DIR
    /
    "train_operational_readouts.csv"
)

TRAIN_SPEC_PATH = (
    DATA_DIR
    /
    "train_specifications.csv"
)

TRAIN_TTE_PATH = (
    DATA_DIR
    /
    "train_tte.csv"
)

VAL_OP_PATH = (
    DATA_DIR
    /
    "validation_operational_readouts.csv"
)

VAL_SPEC_PATH = (
    DATA_DIR
    /
    "validation_specifications.csv"
)

VAL_LABEL_PATH = (
    DATA_DIR
    /
    "validation_labels.csv"
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
    VAL_LABEL_PATH
]


print("=" * 80)
print("CHECKING DATA FILES")
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
# 6. BASIC CHECKS
# ============================================================

train_vehicle_count = (
    train_op[
        "vehicle_id"
    ]
    .nunique()
)

validation_vehicle_count = (
    val_op[
        "vehicle_id"
    ]
    .nunique()
)


print(
    "\nTraining vehicles:",
    train_vehicle_count
)

print(
    "Validation vehicles:",
    validation_vehicle_count
)


assert (
    train_vehicle_count
    ==
    train_spec[
        "vehicle_id"
    ].nunique()
)

assert (
    train_vehicle_count
    ==
    train_tte[
        "vehicle_id"
    ].nunique()
)


# ============================================================
# 7. IDENTIFY OPERATIONAL FEATURES
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
    len(
        RAW_FEATURE_COLS
    )
)


# ============================================================
# 8. SORT TIME SERIES
# ============================================================

train_op = (
    train_op
    .sort_values(
        [
            "vehicle_id",
            "time_step"
        ]
    )
    .reset_index(
        drop=True
    )
)


val_op = (
    val_op
    .sort_values(
        [
            "vehicle_id",
            "time_step"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 9. MULTI-READOUT HISTORICAL FEATURE ENGINEERING
# ============================================================
#
# For each operational feature:
#
# Current:
#   x_t
#
# Previous:
#   x_(t-1)
#
# Delta:
#   x_t - x_(t-1)
#
# Rate:
#   delta / time gap
#
# Longer history:
#   change from 3 readouts ago
#   change from 5 readouts ago
#
# Rolling history:
#   mean over last 3 readouts
#   std over last 3 readouts
#   mean over last 5 readouts
#   std over last 5 readouts
#
# NOTE:
# These are readout windows, not fixed time windows.
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


    # ========================================================
    # TIME GAP
    # ========================================================

    print(
        "  Creating time gap..."
    )


    df[
        "time_gap"
    ] = (
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


    # ========================================================
    # PREVIOUS VALUE
    # ========================================================

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


    # ========================================================
    # CURRENT / PREVIOUS ARRAYS
    # ========================================================

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


    # ========================================================
    # DELTA
    # ========================================================

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


    # ========================================================
    # RATE OF CHANGE
    # ========================================================

    print(
        "  Creating rate of change..."
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


    # ========================================================
    # CHANGE FROM 3 READOUTS AGO
    # ========================================================

    print(
        "  Creating change from 3 readouts ago..."
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


    # ========================================================
    # CHANGE FROM 5 READOUTS AGO
    # ========================================================

    print(
        "  Creating change from 5 readouts ago..."
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


    # ========================================================
    # ROLLING MEAN — LAST 3
    # ========================================================

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


    # ========================================================
    # ROLLING STD — LAST 3
    # ========================================================

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


    # ========================================================
    # ROLLING MEAN — LAST 5
    # ========================================================

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


    # ========================================================
    # ROLLING STD — LAST 5
    # ========================================================

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


    # ========================================================
    # COMBINE
    # ========================================================

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
# 10. CREATE TRAINING HISTORY FEATURES
# ============================================================

print("\n" + "=" * 80)
print("CREATING TRAIN HISTORY FEATURES")
print("=" * 80)


train_history = (
    create_history_features(
        train_op,
        RAW_FEATURE_COLS
    )
)


# ============================================================
# 11. CREATE VALIDATION HISTORY FEATURES
# ============================================================

print("\n" + "=" * 80)
print("CREATING VALIDATION HISTORY FEATURES")
print("=" * 80)


val_history = (
    create_history_features(
        val_op,
        RAW_FEATURE_COLS
    )
)


del train_op
del val_op

gc.collect()


# ============================================================
# 12. MERGE TTE
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


# ============================================================
# 13. TIME TO ENDPOINT
# ============================================================

train_full[
    "time_to_endpoint"
] = (
    train_full[
        "length_of_study_time_step"
    ]
    -
    train_full[
        "time_step"
    ]
)


# ============================================================
# 14. C0-C4 LABEL FUNCTION
# ============================================================

def tte_to_class(
    tte
):

    if pd.isna(
        tte
    ):

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


# ============================================================
# 15. CREATE TRAINING LABELS
# ============================================================

train_full[
    "class_label"
] = np.nan


# ------------------------------------------------------------
# Repair vehicles
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Non-repair vehicles
#
# Only use C0 when there are >48 observed repair-free
# time units remaining.
# ------------------------------------------------------------

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


# ============================================================
# 16. REMOVE UNKNOWN LABELS
# ============================================================

candidate_train = (
    train_full
    .dropna(
        subset=[
            "class_label"
        ]
    )
    .copy()
)


candidate_train[
    "class_label"
] = (
    candidate_train[
        "class_label"
    ]
    .astype(int)
)


print("\n" + "=" * 80)
print("ELIGIBLE TRAINING OBSERVATIONS")
print("=" * 80)


print(
    candidate_train[
        "class_label"
    ]
    .value_counts()
    .sort_index()
)


del train_full

gc.collect()


# ============================================================
# 17. VEHICLE-AWARE SAMPLING
# ============================================================
#
# Repair vehicle:
#
# Maximum one observation per available C0-C4 class.
#
# Non-repair vehicle:
#
# One safely-labelled C0 observation.
#
# ============================================================


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


# ------------------------------------------------------------
# Repair vehicles
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Non-repair vehicles
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

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


print("\n" + "=" * 80)
print("VEHICLE-AWARE TRAINING DISTRIBUTION")
print("=" * 80)


print(
    train_sampled[
        "class_label"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nTraining rows:",
    len(
        train_sampled
    )
)


print(
    "Unique vehicles:",
    train_sampled[
        "vehicle_id"
    ]
    .nunique()
)


del candidate_train
del repair_candidates
del non_repair_candidates
del repair_sampled
del non_repair_sampled

gc.collect()


# ============================================================
# 18. JOIN VEHICLE SPECIFICATIONS
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
# 19. BUILD VALIDATION TABLE
# ============================================================
#
# Validation data are already truncated by the challenge.
# Therefore use the last available readout.
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


print("\n" + "=" * 80)
print("VALIDATION DISTRIBUTION")
print("=" * 80)


print(
    val_last[
        "class_label"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 20. BUILD X / Y
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


# ============================================================
# 21. ALIGN FEATURES
# ============================================================

train_only = (
    set(
        X_train.columns
    )
    -
    set(
        X_val.columns
    )
)


val_only = (
    set(
        X_val.columns
    )
    -
    set(
        X_train.columns
    )
)


print(
    "\nTrain-only columns:",
    train_only
)

print(
    "Validation-only columns:",
    val_only
)


assert len(
    train_only
) == 0


assert len(
    val_only
) == 0


X_val = X_val[
    X_train.columns
]


assert (
    list(
        X_train.columns
    )
    ==
    list(
        X_val.columns
    )
)


print(
    "✓ Train and validation features aligned."
)


# ============================================================
# 22. FEATURE TYPES
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


print(
    "\nNumeric features:",
    len(
        numeric_cols
    )
)

print(
    "Categorical features:",
    len(
        categorical_cols
    )
)

print(
    "Total raw model features:",
    len(
        X_train.columns
    )
)


# ============================================================
# 23. PREPROCESSING
# ============================================================

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
# 24. XGBOOST MODEL
# ============================================================
#
# Current selected configuration:
#
# Vehicle-aware
# Multi-readout history
# No additional class weighting
#
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
# 25. TRAIN MODEL
# ============================================================

print("\n" + "=" * 80)
print("TRAINING HISTORY XGBOOST")
print("=" * 80)


history_pipeline.fit(
    X_train,
    y_train
)


print(
    "✓ Training complete"
)


# ============================================================
# 26. PREDICT
# ============================================================

y_pred = (
    history_pipeline
    .predict(
        X_val
    )
    .astype(int)
)


y_proba = (
    history_pipeline
    .predict_proba(
        X_val
    )
)


assert (
    y_proba.shape
    ==
    (
        len(
            X_val
        ),
        5
    )
)


# ============================================================
# 27. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)


print(
    classification_report(
        y_val,
        y_pred,

        labels=[
            0,
            1,
            2,
            3,
            4
        ],

        target_names=[
            "C0",
            "C1",
            "C2",
            "C3",
            "C4"
        ],

        digits=4,

        zero_division=0
    )
)


accuracy = (
    accuracy_score(
        y_val,
        y_pred
    )
)


macro_f1 = (
    f1_score(
        y_val,
        y_pred,
        average="macro"
    )
)


weighted_f1 = (
    f1_score(
        y_val,
        y_pred,
        average="weighted"
    )
)


report = (
    classification_report(
        y_val,
        y_pred,

        labels=[
            0,
            1,
            2,
            3,
            4
        ],

        target_names=[
            "C0",
            "C1",
            "C2",
            "C3",
            "C4"
        ],

        output_dict=True,

        zero_division=0
    )
)


c4_precision = (
    report[
        "C4"
    ][
        "precision"
    ]
)


c4_recall = (
    report[
        "C4"
    ][
        "recall"
    ]
)


c4_f1 = (
    report[
        "C4"
    ][
        "f1-score"
    ]
)


print(
    "Accuracy:",
    round(
        accuracy,
        4
    )
)

print(
    "Macro F1:",
    round(
        macro_f1,
        4
    )
)

print(
    "Weighted F1:",
    round(
        weighted_f1,
        4
    )
)

print(
    "C4 Precision:",
    round(
        c4_precision,
        4
    )
)

print(
    "C4 Recall:",
    round(
        c4_recall,
        4
    )
)

print(
    "C4 F1:",
    round(
        c4_f1,
        4
    )
)


# ============================================================
# 28. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    y_pred,

    labels=[
        0,
        1,
        2,
        3,
        4
    ]
)


print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)


print(
    cm
)


# ============================================================
# 29. SAVE VALIDATION PREDICTIONS
# ============================================================

validation_predictions = pd.DataFrame({

    "vehicle_id":
        val_last[
            "vehicle_id"
        ]
        .to_numpy(),

    "actual_class":
        y_val
        .to_numpy(),

    "predicted_class":
        y_pred,

    "P_C0":
        y_proba[
            :,
            0
        ],

    "P_C1":
        y_proba[
            :,
            1
        ],

    "P_C2":
        y_proba[
            :,
            2
        ],

    "P_C3":
        y_proba[
            :,
            3
        ],

    "P_C4":
        y_proba[
            :,
            4
        ]
})


validation_predictions.to_csv(

    OUTPUT_DIR
    /
    "history_xgboost_validation_predictions.csv",

    index=False
)


# ============================================================
# 30. SHAP
# ============================================================

print("\n" + "=" * 80)
print("SHAP EXPLAINABILITY")
print("=" * 80)


# ------------------------------------------------------------
# Get fitted model components
# ------------------------------------------------------------

fitted_preprocessor = (
    history_pipeline
    .named_steps[
        "preprocessor"
    ]
)


fitted_model = (
    history_pipeline
    .named_steps[
        "classifier"
    ]
)


# ------------------------------------------------------------
# Transform validation data
# ------------------------------------------------------------

print(
    "Transforming validation data..."
)


X_val_transformed = (
    fitted_preprocessor
    .transform(
        X_val
    )
)


# ------------------------------------------------------------
# Feature names
# ------------------------------------------------------------

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


feature_names = np.array([
    name
    .replace(
        "numeric__",
        ""
    )
    .replace(
        "categorical__",
        ""
    )
    for name in feature_names
])


print(
    "Transformed validation shape:",
    X_val_transformed.shape
)

print(
    "Number of transformed features:",
    len(
        feature_names
    )
)


# ============================================================
# 31. SHAP SAMPLE
# ============================================================

sample_size = min(
    SHAP_SAMPLE_SIZE,
    X_val_transformed.shape[0]
)


rng = np.random.default_rng(
    RANDOM_STATE
)


sample_indices = rng.choice(
    X_val_transformed.shape[0],
    size=sample_size,
    replace=False
)


X_shap = (
    X_val_transformed[
        sample_indices
    ]
)


if hasattr(
    X_shap,
    "toarray"
):

    X_shap = (
        X_shap
        .toarray()
    )


print(
    "SHAP sample shape:",
    X_shap.shape
)


# ============================================================
# 32. SHAP TREE EXPLAINER
# ============================================================

print(
    "Creating TreeExplainer..."
)


explainer = shap.TreeExplainer(
    fitted_model
)


print(
    "Calculating SHAP values..."
)


shap_values = (
    explainer(
        X_shap
    )
)


print(
    "SHAP values shape:",
    shap_values.values.shape
)


# ============================================================
# 33. CHECK MULTICLASS SHAP
# ============================================================

if shap_values.values.ndim != 3:

    raise ValueError(
        "Expected SHAP output with shape "
        "(samples, features, classes). "
        f"Received {shap_values.values.shape}"
    )


if shap_values.values.shape[2] != 5:

    raise ValueError(
        "Expected five model classes. "
        f"Received {shap_values.values.shape}"
    )


# ============================================================
# 34. C4 GLOBAL SHAP
# ============================================================

C4_INDEX = 4


c4_shap_values = (
    shap_values
    .values[
        :,
        :,
        C4_INDEX
    ]
)


mean_abs_c4_shap = (
    np.abs(
        c4_shap_values
    )
    .mean(
        axis=0
    )
)


c4_global_importance = pd.DataFrame({

    "feature":
        feature_names,

    "mean_abs_shap_c4":
        mean_abs_c4_shap
})


c4_global_importance = (
    c4_global_importance
    .sort_values(
        "mean_abs_shap_c4",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 80)
print("TOP 20 FEATURES FOR C4")
print("=" * 80)


print(
    c4_global_importance
    .head(20)
    .to_string(
        index=False
    )
)


c4_global_importance.to_csv(

    SHAP_OUTPUT_DIR
    /
    "c4_global_feature_importance.csv",

    index=False
)


# ============================================================
# 35. C4 SHAP BAR PLOT
# ============================================================

print(
    "\nCreating C4 SHAP bar plot..."
)


shap.summary_plot(
    c4_shap_values,
    X_shap,

    feature_names=
        feature_names,

    plot_type=
        "bar",

    max_display=
        20,

    show=
        False
)


plt.title(
    "Top Features Influencing C4 Prediction"
)


plt.tight_layout()


plt.savefig(

    SHAP_OUTPUT_DIR
    /
    "c4_shap_bar.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 36. C4 SHAP BEESWARM
# ============================================================

print(
    "Creating C4 SHAP beeswarm..."
)


shap.summary_plot(
    c4_shap_values,
    X_shap,

    feature_names=
        feature_names,

    max_display=
        20,

    show=
        False
)


plt.title(
    "SHAP Distribution for C4 Prediction"
)


plt.tight_layout()


plt.savefig(

    SHAP_OUTPUT_DIR
    /
    "c4_shap_beeswarm.png",

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 37. FIND EXPLANATION CASES
# ============================================================

actual_classes = (
    y_val
    .to_numpy()
)


predicted_classes = (
    y_pred
)


true_positive_c4 = np.where(
    (
        actual_classes
        ==
        4
    )
    &
    (
        predicted_classes
        ==
        4
    )
)[0]


false_positive_c4 = np.where(
    (
        actual_classes
        !=
        4
    )
    &
    (
        predicted_classes
        ==
        4
    )
)[0]


false_negative_c4 = np.where(
    (
        actual_classes
        ==
        4
    )
    &
    (
        predicted_classes
        !=
        4
    )
)[0]


print("\n" + "=" * 80)
print("C4 EXPLANATION CASES")
print("=" * 80)


print(
    "True-positive C4:",
    len(
        true_positive_c4
    )
)


print(
    "False-positive C4:",
    len(
        false_positive_c4
    )
)


print(
    "False-negative C4:",
    len(
        false_negative_c4
    )
)


# ============================================================
# 38. LOCAL VEHICLE EXPLANATION FUNCTION
# ============================================================

def explain_vehicle(
    vehicle_index,
    case_name
):

    vehicle_id = (
        val_last
        .iloc[
            vehicle_index
        ][
            "vehicle_id"
        ]
    )


    actual_class = int(
        y_val
        .iloc[
            vehicle_index
        ]
    )


    predicted_class = int(
        y_pred[
            vehicle_index
        ]
    )


    probabilities = (
        y_proba[
            vehicle_index
        ]
    )


    print("\n" + "=" * 80)
    print(case_name)
    print("=" * 80)


    print(
        "Vehicle ID:",
        vehicle_id
    )


    print(
        "Actual class:",
        f"C{actual_class}"
    )


    print(
        "Predicted class:",
        f"C{predicted_class}"
    )


    print(
        "Probabilities:"
    )


    for class_index in range(
        5
    ):

        print(
            f"P(C{class_index}) = "
            f"{probabilities[class_index]:.4f}"
        )


    # --------------------------------------------------------
    # Transform one vehicle
    # --------------------------------------------------------

    vehicle_X = (
        fitted_preprocessor
        .transform(
            X_val
            .iloc[
                [
                    vehicle_index
                ]
            ]
        )
    )


    if hasattr(
        vehicle_X,
        "toarray"
    ):

        vehicle_X = (
            vehicle_X
            .toarray()
        )


    # --------------------------------------------------------
    # Calculate local SHAP
    # --------------------------------------------------------

    vehicle_shap = (
        explainer(
            vehicle_X
        )
    )


    vehicle_c4_shap = (
        vehicle_shap
        .values[
            0,
            :,
            C4_INDEX
        ]
    )


    vehicle_feature_values = (
        vehicle_X[
            0
        ]
    )


    # --------------------------------------------------------
    # Build explanation table
    # --------------------------------------------------------

    local_df = pd.DataFrame({

        "feature":
            feature_names,

        "feature_value":
            vehicle_feature_values,

        "shap_c4":
            vehicle_c4_shap,

        "abs_shap":
            np.abs(
                vehicle_c4_shap
            )
    })


    local_df = (
        local_df
        .sort_values(
            "abs_shap",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "\nTop 10 factors affecting C4:"
    )


    print(
        local_df[
            [
                "feature",
                "feature_value",
                "shap_c4"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


    safe_name = (
        case_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )


    local_df.to_csv(

        SHAP_OUTPUT_DIR
        /
        (
            f"{safe_name}"
            f"_vehicle_{vehicle_id}.csv"
        ),

        index=False
    )


    return {

        "case":
            case_name,

        "vehicle_id":
            vehicle_id,

        "actual_class":
            actual_class,

        "predicted_class":
            predicted_class,

        "P_C0":
            probabilities[0],

        "P_C1":
            probabilities[1],

        "P_C2":
            probabilities[2],

        "P_C3":
            probabilities[3],

        "P_C4":
            probabilities[4],

        "top_feature_1":
            local_df.iloc[
                0
            ][
                "feature"
            ],

        "top_feature_1_shap":
            local_df.iloc[
                0
            ][
                "shap_c4"
            ],

        "top_feature_2":
            local_df.iloc[
                1
            ][
                "feature"
            ],

        "top_feature_2_shap":
            local_df.iloc[
                1
            ][
                "shap_c4"
            ],

        "top_feature_3":
            local_df.iloc[
                2
            ][
                "feature"
            ],

        "top_feature_3_shap":
            local_df.iloc[
                2
            ][
                "shap_c4"
            ]
    }


# ============================================================
# 39. EXPLAIN REPRESENTATIVE CASES
# ============================================================

case_summaries = []


# ------------------------------------------------------------
# True Positive
# ------------------------------------------------------------

if len(
    true_positive_c4
) > 0:

    tp_index = (
        true_positive_c4[
            np.argmax(
                y_proba[
                    true_positive_c4,
                    4
                ]
            )
        ]
    )


    case_summaries.append(

        explain_vehicle(
            tp_index,
            "True Positive C4"
        )

    )


# ------------------------------------------------------------
# False Positive
# ------------------------------------------------------------

if len(
    false_positive_c4
) > 0:

    fp_index = (
        false_positive_c4[
            np.argmax(
                y_proba[
                    false_positive_c4,
                    4
                ]
            )
        ]
    )


    case_summaries.append(

        explain_vehicle(
            fp_index,
            "False Positive C4"
        )

    )


# ------------------------------------------------------------
# False Negative
# ------------------------------------------------------------

if len(
    false_negative_c4
) > 0:

    fn_index = (
        false_negative_c4[
            np.argmax(
                y_proba[
                    false_negative_c4,
                    4
                ]
            )
        ]
    )


    case_summaries.append(

        explain_vehicle(
            fn_index,
            "False Negative C4"
        )

    )


# ============================================================
# 40. SAVE CASE SUMMARY
# ============================================================

case_summary_df = pd.DataFrame(
    case_summaries
)


case_summary_df.to_csv(

    SHAP_OUTPUT_DIR
    /
    "c4_explanation_cases.csv",

    index=False
)


print("\n" + "=" * 80)
print("C4 SHAP CASE SUMMARY")
print("=" * 80)


print(
    case_summary_df
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 41. SIGNED GLOBAL SHAP
# ============================================================

mean_signed_c4_shap = (
    c4_shap_values
    .mean(
        axis=0
    )
)


signed_importance = pd.DataFrame({

    "feature":
        feature_names,

    "mean_shap_c4":
        mean_signed_c4_shap,

    "mean_abs_shap_c4":
        mean_abs_c4_shap
})


signed_importance.to_csv(

    SHAP_OUTPUT_DIR
    /
    "c4_signed_shap_importance.csv",

    index=False
)


top_positive = (
    signed_importance
    .sort_values(
        "mean_shap_c4",
        ascending=False
    )
    .head(15)
)


top_negative = (
    signed_importance
    .sort_values(
        "mean_shap_c4",
        ascending=True
    )
    .head(15)
)


print("\n" + "=" * 80)
print("TOP FEATURES PUSHING TOWARD C4")
print("=" * 80)


print(
    top_positive
    .round(6)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("TOP FEATURES PUSHING AWAY FROM C4")
print("=" * 80)


print(
    top_negative
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 42. FINISH
# ============================================================

print("\n" + "=" * 80)
print("FLEETCARE MODEL + SHAP COMPLETE")
print("=" * 80)


print(
    "\nModel:"
)

print(
    "Vehicle-aware + Multi-readout History "
    "+ XGBoost + No Weight"
)


print(
    "\nValidation:"
)

print(
    "Accuracy:",
    round(
        accuracy,
        4
    )
)

print(
    "Macro F1:",
    round(
        macro_f1,
        4
    )
)

print(
    "C4 Precision:",
    round(
        c4_precision,
        4
    )
)

print(
    "C4 Recall:",
    round(
        c4_recall,
        4
    )
)

print(
    "C4 F1:",
    round(
        c4_f1,
        4
    )
)


print(
    "\nSHAP outputs:"
)

print(
    SHAP_OUTPUT_DIR
)


print(
    "\nGenerated:"
)

print(
    "- c4_global_feature_importance.csv"
)

print(
    "- c4_signed_shap_importance.csv"
)

print(
    "- c4_explanation_cases.csv"
)

print(
    "- c4_shap_bar.png"
)

print(
    "- c4_shap_beeswarm.png"
)

print(
    "- local explanation CSVs"
)