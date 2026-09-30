#Additional oversampling did not improve generalization. Although random oversampling increased raw C4 recall, it substantially increased false positives and reduced Top-K critical-vehicle recall. SMOTE further degraded ranking performance. Therefore, the final model retained vehicle-aware sampling without additional synthetic oversampling.
# | Strategy | Macro-F1 | Risk6 AP | C4 Recall@5% |
# |---|---:|---:|---:|
# | **Vehicle-aware only** | **0.203** | **0.044** | **19.7%** |
# | + Random Oversampling | 0.194 | 0.040 | 14.5% |
# | + SMOTE | 0.189 | 0.031 | 13.2% |
# ============================================================
# FleetCare AI
# TEST 13 — IMBALANCE HANDLING ABLATION
#
# Compare:
#   A. Current Final Base Model
#   B. Random Oversampling
#   C. SMOTE
#
# IMPORTANT:
# - Same feature engineering
# - Same vehicle-aware sampling
# - Same XGBoost parameters
# - Same official validation set
# - No calibration in this experiment
# - ONLY resampling strategy changes
# ============================================================

from pathlib import Path
import gc

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
)


# ============================================================
# 1. CONFIG
# ============================================================

RANDOM_STATE = 42

CLASS_IDS = [0, 1, 2, 3, 4]

CLASS_NAMES = [
    "C0",
    "C1",
    "C2",
    "C3",
    "C4",
]

pd.set_option(
    "display.max_columns",
    200
)

pd.set_option(
    "display.width",
    260
)


# ============================================================
# 2. PATHS
#
# test_13.py sits in:
# AIS_Project/SystemCode/data/
#
# raw/
# processed/
# are in the same folder.
# ============================================================

SCRIPT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

DATA_DIR = (
    SCRIPT_DIR
    / "raw"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "processed"
)

OUTPUT_DIR.mkdir(
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
# 3. CHECK FILES
# ============================================================

required_files = [
    TRAIN_OP_PATH,
    TRAIN_SPEC_PATH,
    TRAIN_TTE_PATH,
    VAL_OP_PATH,
    VAL_SPEC_PATH,
    VAL_LABEL_PATH,
]


print("=" * 90)
print("FLEETCARE AI — TEST 13")
print("IMBALANCE / OVERSAMPLING ABLATION")
print("=" * 90)

print(
    "\nDATA_DIR:",
    DATA_DIR
)

for path in required_files:

    print(
        path.name,
        "->",
        path.exists()
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: {path}"
        )


# ============================================================
# 4. LOAD DATA
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "LOADING DATA"
)

print(
    "=" * 90
)


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
# 5. RAW OPERATIONAL FEATURES
# ============================================================

RAW_FEATURE_COLS = [

    col

    for col in train_op.columns

    if col not in [
        "vehicle_id",
        "time_step",
    ]
]


print(
    "\nRaw operational features:",
    len(RAW_FEATURE_COLS)
)


# ============================================================
# 6. MULTI-READOUT HISTORICAL FEATURES
#
# SAME representation as final model:
#
# current
# previous
# delta
# rate
# change3
# change5
# rolling mean 3
# rolling std 3
# rolling mean 5
# rolling std 5
# ============================================================

def create_history_features(
    df,
    feature_cols,
):

    print(
        "  Sorting vehicle history..."
    )

    df = (
        df
        .sort_values(
            [
                "vehicle_id",
                "time_step",
            ]
        )
        .reset_index(
            drop=True
        )
        .copy()
    )


    grouped = df.groupby(
        "vehicle_id",
        sort=False,
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
        df["time_gap"]
        .replace(
            0,
            np.nan
        )
    )


    # --------------------------------------------------------
    # Previous
    # --------------------------------------------------------

    print(
        "  Creating previous..."
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
        ]
    )

    rate.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
        inplace=True,
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
            min_periods=1,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
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
            min_periods=2,
        )
        .std()
        .reset_index(
            level=0,
            drop=True,
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
            min_periods=1,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
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
            min_periods=2,
        )
        .std()
        .reset_index(
            level=0,
            drop=True,
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
        "  Combining features..."
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
            rolling_std5,
        ],

        axis=1,
    )


    result.replace(

        [
            np.inf,
            -np.inf,
        ],

        np.nan,

        inplace=True,
    )


    print(
        "  Historical feature shape:",
        result.shape
    )


    return result


# ============================================================
# 7. BUILD HISTORY FEATURES
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "CREATING TRAIN HISTORY"
)

print(
    "=" * 90
)


train_history = (
    create_history_features(
        train_op,
        RAW_FEATURE_COLS,
    )
)


print(
    "\n"
    + "=" * 90
)

print(
    "CREATING VALIDATION HISTORY"
)

print(
    "=" * 90
)


val_history = (
    create_history_features(
        val_op,
        RAW_FEATURE_COLS,
    )
)


del train_op
del val_op

gc.collect()


# ============================================================
# 8. CENSORING-AWARE LABELS
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


train_full[
    "class_label"
] = np.nan


# Repair vehicles
repair_mask = (
    train_full[
        "in_study_repair"
    ]
    ==
    1
)


train_full.loc[
    repair_mask,
    "class_label",
] = (

    train_full.loc[
        repair_mask,
        "time_to_endpoint",
    ]

    .apply(
        tte_to_class
    )
)


# Non-repair vehicles:
# only safely label C0 if >48 units remain
safe_non_repair_mask = (

    (
        train_full[
            "in_study_repair"
        ]
        ==
        0
    )

    &

    (
        train_full[
            "time_to_endpoint"
        ]
        >
        48
    )
)


train_full.loc[
    safe_non_repair_mask,
    "class_label",
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


candidate_train[
    "class_label"
] = (

    candidate_train[
        "class_label"
    ]

    .astype(int)
)


print(
    "\nEligible training distribution:"
)

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
# 9. VEHICLE-AWARE SAMPLING
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "VEHICLE-AWARE SAMPLING"
)

print(
    "=" * 90
)


repair_candidates = (

    candidate_train[

        candidate_train[
            "in_study_repair"
        ]
        ==
        1

    ]

    .copy()
)


non_repair_candidates = (

    candidate_train[

        candidate_train[
            "in_study_repair"
        ]
        ==
        0

    ]

    .copy()
)


# One observation per repair vehicle
# per available class
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
        random_state=RANDOM_STATE,
    )

    .reset_index(
        drop=True
    )
)


# One safe C0 per non-repair vehicle
non_repair_sampled = (

    non_repair_candidates

    .groupby(
        "vehicle_id",
        group_keys=False,
    )

    .sample(
        n=1,
        random_state=RANDOM_STATE,
    )

    .reset_index(
        drop=True
    )
)


train_sampled = pd.concat(

    [
        repair_sampled,
        non_repair_sampled,
    ],

    ignore_index=True,
)


train_sampled = (

    train_sampled

    .sample(
        frac=1,
        random_state=RANDOM_STATE,
    )

    .reset_index(
        drop=True
    )
)


print(
    "\nVehicle-aware training distribution:"
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
# 10. ADD VEHICLE SPECIFICATIONS
# ============================================================

train_sampled = (

    train_sampled

    .merge(
        train_spec,
        on="vehicle_id",
        how="left",
        validate="many_to_one",
    )
)


# ============================================================
# 11. BUILD OFFICIAL VALIDATION
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


print(
    "\nOfficial validation distribution:"
)

print(

    val_last[
        "class_label"
    ]

    .value_counts()

    .sort_index()
)


# ============================================================
# 12. X / Y
# ============================================================

NON_FEATURE_COLS = [

    "vehicle_id",

    "class_label",

    "length_of_study_time_step",

    "in_study_repair",

    "time_to_endpoint",
]


X_train = (

    train_sampled

    .drop(
        columns=NON_FEATURE_COLS,
        errors="ignore",
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
            "class_label",
        ],
        errors="ignore",
    )
)


y_val = (

    val_last[
        "class_label"
    ]

    .astype(int)

    .copy()
)


X_val = X_val.reindex(
    columns=X_train.columns
)


assert (
    list(X_train.columns)
    ==
    list(X_val.columns)
)


# ============================================================
# 13. PREPROCESSING
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
        ),
    )

])


categorical_transformer = Pipeline([

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

])


preprocessor = ColumnTransformer([

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

])


print(
    "\n"
    + "=" * 90
)

print(
    "PREPROCESSING"
)

print(
    "=" * 90
)


X_train_transformed = (
    preprocessor
    .fit_transform(
        X_train
    )
)


X_val_transformed = (
    preprocessor
    .transform(
        X_val
    )
)


y_train_array = (
    y_train
    .to_numpy()
)


y_val_array = (
    y_val
    .to_numpy()
)


print(
    "Transformed train:",
    X_train_transformed.shape
)

print(
    "Transformed validation:",
    X_val_transformed.shape
)


# ============================================================
# 14. MODEL FACTORY
#
# SAME XGBoost as generate_model.py
# ============================================================

def create_xgb():

    return XGBClassifier(

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

        n_jobs=-1,
    )


# ============================================================
# 15. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    model,
):

    probabilities = (
        model.predict_proba(
            X_val_transformed
        )
    )


    predictions = (
        np.argmax(
            probabilities,
            axis=1,
        )
    )


    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val_array,
        predictions,
    )


    macro_precision = precision_score(
        y_val_array,
        predictions,
        labels=CLASS_IDS,
        average="macro",
        zero_division=0,
    )


    macro_recall = recall_score(
        y_val_array,
        predictions,
        labels=CLASS_IDS,
        average="macro",
        zero_division=0,
    )


    macro_f1 = f1_score(
        y_val_array,
        predictions,
        labels=CLASS_IDS,
        average="macro",
        zero_division=0,
    )


    per_class_precision = (
        precision_score(
            y_val_array,
            predictions,
            labels=CLASS_IDS,
            average=None,
            zero_division=0,
        )
    )


    per_class_recall = (
        recall_score(
            y_val_array,
            predictions,
            labels=CLASS_IDS,
            average=None,
            zero_division=0,
        )
    )


    per_class_f1 = (
        f1_score(
            y_val_array,
            predictions,
            labels=CLASS_IDS,
            average=None,
            zero_division=0,
        )
    )


    # --------------------------------------------------------
    # Risk6
    # --------------------------------------------------------

    actual_c4 = (
        y_val_array
        ==
        4
    ).astype(int)


    risk6 = (
        probabilities[
            :,
            4
        ]
    )


    risk6_ap = (
        average_precision_score(
            actual_c4,
            risk6,
        )
    )


    risk6_auc = (
        roc_auc_score(
            actual_c4,
            risk6,
        )
    )


    # --------------------------------------------------------
    # Top 2.5%, 5%, 10%, 20%
    # --------------------------------------------------------

    ranking = np.argsort(
        -risk6,
        kind="stable",
    )


    total_c4 = int(
        actual_c4.sum()
    )


    capacity_results = {}


    for capacity in [
        0.025,
        0.05,
        0.10,
        0.20,
    ]:

        k = max(
            1,
            int(
                np.ceil(
                    len(y_val_array)
                    *
                    capacity
                )
            )
        )


        selected = (
            ranking[
                :k
            ]
        )


        c4_found = int(
            actual_c4[
                selected
            ]
            .sum()
        )


        recall_at_k = (
            c4_found
            /
            total_c4
            if total_c4
            else np.nan
        )


        precision_at_k = (
            actual_c4[
                selected
            ]
            .mean()
        )


        pct = int(
            capacity
            *
            1000
        )


        capacity_results[
            f"C4Recall_{pct}"
        ] = recall_at_k


        capacity_results[
            f"C4Precision_{pct}"
        ] = precision_at_k


        capacity_results[
            f"C4Found_{pct}"
        ] = c4_found


    # --------------------------------------------------------
    # Ordinal metrics
    # --------------------------------------------------------

    absolute_class_error = np.abs(
        predictions
        -
        y_val_array
    )


    ordinal_mae = float(
        absolute_class_error.mean()
    )


    within_one_class = float(
        (
            absolute_class_error
            <=
            1
        )
        .mean()
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_val_array,
        predictions,
        labels=CLASS_IDS,
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        name
    )

    print(
        "=" * 90
    )


    print(

        classification_report(

            y_val_array,

            predictions,

            labels=CLASS_IDS,

            target_names=CLASS_NAMES,

            digits=4,

            zero_division=0,
        )
    )


    print(
        "Confusion matrix:"
    )

    print(
        cm
    )


    print(
        "\nRisk6 AP:",
        round(
            risk6_ap,
            4
        )
    )


    print(
        "Risk6 AUC:",
        round(
            risk6_auc,
            4
        )
    )


    print(
        "Ordinal MAE:",
        round(
            ordinal_mae,
            4
        )
    )


    print(
        "Within ±1 class:",
        round(
            within_one_class,
            4
        )
    )


    result = {

        "Model":
            name,

        "Accuracy":
            accuracy,

        "MacroPrecision":
            macro_precision,

        "MacroRecall":
            macro_recall,

        "MacroF1":
            macro_f1,

        "C0Precision":
            per_class_precision[0],

        "C1Precision":
            per_class_precision[1],

        "C2Precision":
            per_class_precision[2],

        "C3Precision":
            per_class_precision[3],

        "C4Precision":
            per_class_precision[4],

        "C0Recall":
            per_class_recall[0],

        "C1Recall":
            per_class_recall[1],

        "C2Recall":
            per_class_recall[2],

        "C3Recall":
            per_class_recall[3],

        "C4Recall":
            per_class_recall[4],

        "C0F1":
            per_class_f1[0],

        "C1F1":
            per_class_f1[1],

        "C2F1":
            per_class_f1[2],

        "C3F1":
            per_class_f1[3],

        "C4F1":
            per_class_f1[4],

        "Risk6_AP":
            risk6_ap,

        "Risk6_AUC":
            risk6_auc,

        "OrdinalMAE":
            ordinal_mae,

        "Within1Class":
            within_one_class,
    }


    result.update(
        capacity_results
    )


    return result


# ============================================================
# 16. ORIGINAL DISTRIBUTION
# ============================================================

original_counts = (

    pd.Series(
        y_train_array
    )

    .value_counts()

    .sort_index()
)


print(
    "\n"
    + "=" * 90
)

print(
    "ORIGINAL TRAINING DISTRIBUTION"
)

print(
    "=" * 90
)

print(
    original_counts
)


c0_count = int(
    original_counts.loc[0]
)


# ============================================================
# 17. MODEL A — CURRENT FINAL BASE MODEL
# ============================================================

print(
    "\nTraining CURRENT model..."
)


current_model = create_xgb()


current_model.fit(
    X_train_transformed,
    y_train_array,
)


results = []


results.append(

    evaluate_model(
        "A_Current"
        ,
        current_model,
    )
)


# ============================================================
# 18. MODEL B — RANDOM OVERSAMPLING
#
# Moderate oversampling:
#
# C1 >= 20% of C0
# C2 >= 20% of C0
# C3 >= 25% of C0
# C4 >= 30% of C0
#
# If a class already exceeds target, leave it unchanged.
# ============================================================

ros_ratios = {

    1: 0.20,

    2: 0.20,

    3: 0.25,

    4: 0.30,
}


ros_strategy = {}


for class_id, ratio in (
    ros_ratios.items()
):

    current_count = int(
        original_counts.get(
            class_id,
            0
        )
    )


    target_count = max(

        current_count,

        int(
            c0_count
            *
            ratio
        ),
    )


    if target_count > current_count:

        ros_strategy[
            class_id
        ] = target_count


print(
    "\nRandom oversampling strategy:"
)

print(
    ros_strategy
)


ros = RandomOverSampler(

    sampling_strategy=
        ros_strategy,

    random_state=
        RANDOM_STATE,
)


X_ros, y_ros = (
    ros.fit_resample(
        X_train_transformed,
        y_train_array,
    )
)


print(
    "\nROS distribution:"
)

print(

    pd.Series(
        y_ros
    )

    .value_counts()

    .sort_index()
)


print(
    "\nTraining RANDOM OVERSAMPLING model..."
)


ros_model = create_xgb()


ros_model.fit(
    X_ros,
    y_ros,
)


results.append(

    evaluate_model(
        "B_RandomOversampling",
        ros_model,
    )
)


# ============================================================
# 19. MODEL C — SMOTE
# ============================================================

smote_ratios = {

    1: 0.20,

    2: 0.20,

    3: 0.25,

    4: 0.30,
}


smote_strategy = {}


for class_id, ratio in (
    smote_ratios.items()
):

    current_count = int(
        original_counts.get(
            class_id,
            0
        )
    )


    target_count = max(

        current_count,

        int(
            c0_count
            *
            ratio
        ),
    )


    if target_count > current_count:

        smote_strategy[
            class_id
        ] = target_count


print(
    "\nSMOTE strategy:"
)

print(
    smote_strategy
)


# Check smallest minority class
minority_counts = [

    int(
        original_counts.get(
            class_id,
            0
        )
    )

    for class_id in smote_strategy
]


if len(
    minority_counts
) == 0:

    raise RuntimeError(
        "No minority classes require SMOTE."
    )


smallest_class = min(
    minority_counts
)


# k_neighbors must be smaller than class size
smote_k = min(
    3,
    smallest_class - 1,
)


if smote_k < 1:

    raise RuntimeError(
        "A minority class has fewer than 2 "
        "training observations. "
        "SMOTE cannot be applied safely."
    )


print(
    "SMOTE k_neighbors:",
    smote_k
)


smote = SMOTE(

    sampling_strategy=
        smote_strategy,

    random_state=
        RANDOM_STATE,

    k_neighbors=
        smote_k,
)


X_smote, y_smote = (
    smote.fit_resample(
        X_train_transformed,
        y_train_array,
    )
)


print(
    "\nSMOTE distribution:"
)

print(

    pd.Series(
        y_smote
    )

    .value_counts()

    .sort_index()
)


print(
    "\nTraining SMOTE model..."
)


smote_model = create_xgb()


smote_model.fit(
    X_smote,
    y_smote,
)


results.append(

    evaluate_model(
        "C_SMOTE",
        smote_model,
    )
)


# ============================================================
# 20. FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


comparison_columns = [

    "Model",

    "Accuracy",

    "MacroF1",

    "C1Precision",

    "C1Recall",

    "C1F1",

    "C2Recall",

    "C3Recall",

    "C4Precision",

    "C4Recall",

    "C4F1",

    "Risk6_AP",

    "Risk6_AUC",

    "C4Recall_25",

    "C4Recall_50",

    "C4Recall_100",

    "C4Recall_200",

    "OrdinalMAE",

    "Within1Class",
]


print(
    "\n"
    + "=" * 120
)

print(
    "FINAL IMBALANCE ABLATION"
)

print(
    "=" * 120
)


print(

    results_df[
        comparison_columns
    ]

    .round(4)

    .to_string(
        index=False
    )
)


# ============================================================
# 21. CHANGE VS CURRENT
# ============================================================

baseline = (
    results_df
    .iloc[0]
)


delta_rows = []


for index in range(
    1,
    len(results_df)
):

    row = (
        results_df
        .iloc[index]
    )


    delta_rows.append({

        "Model":
            row[
                "Model"
            ],

        "Delta_Accuracy":
            row[
                "Accuracy"
            ]
            -
            baseline[
                "Accuracy"
            ],

        "Delta_MacroF1":
            row[
                "MacroF1"
            ]
            -
            baseline[
                "MacroF1"
            ],

        "Delta_C1Recall":
            row[
                "C1Recall"
            ]
            -
            baseline[
                "C1Recall"
            ],

        "Delta_C2Recall":
            row[
                "C2Recall"
            ]
            -
            baseline[
                "C2Recall"
            ],

        "Delta_C3Recall":
            row[
                "C3Recall"
            ]
            -
            baseline[
                "C3Recall"
            ],

        "Delta_C4Recall":
            row[
                "C4Recall"
            ]
            -
            baseline[
                "C4Recall"
            ],

        "Delta_Risk6_AP":
            row[
                "Risk6_AP"
            ]
            -
            baseline[
                "Risk6_AP"
            ],

        "Delta_Risk6_AUC":
            row[
                "Risk6_AUC"
            ]
            -
            baseline[
                "Risk6_AUC"
            ],

        "Delta_Top2.5_C4Recall":
            row[
                "C4Recall_25"
            ]
            -
            baseline[
                "C4Recall_25"
            ],

        "Delta_Top5_C4Recall":
            row[
                "C4Recall_50"
            ]
            -
            baseline[
                "C4Recall_50"
            ],

        "Delta_Top10_C4Recall":
            row[
                "C4Recall_100"
            ]
            -
            baseline[
                "C4Recall_100"
            ],

        "Delta_OrdinalMAE":
            row[
                "OrdinalMAE"
            ]
            -
            baseline[
                "OrdinalMAE"
            ],
    })


delta_df = pd.DataFrame(
    delta_rows
)


print(
    "\n"
    + "=" * 120
)

print(
    "CHANGE VS CURRENT MODEL"
)

print(
    "=" * 120
)


print(

    delta_df

    .round(4)

    .to_string(
        index=False
    )
)


# ============================================================
# 22. SAVE RESULTS
# ============================================================

RESULT_PATH = (

    OUTPUT_DIR
    /
    "test_13_imbalance_ablation.csv"
)


DELTA_PATH = (

    OUTPUT_DIR
    /
    "test_13_imbalance_delta.csv"
)


results_df.to_csv(

    RESULT_PATH,

    index=False,
)


delta_df.to_csv(

    DELTA_PATH,

    index=False,
)


print(
    "\n"
    + "=" * 90
)

print(
    "TEST 13 COMPLETE"
)

print(
    "=" * 90
)


print(
    "\nResults:",
    RESULT_PATH
)


print(
    "Delta:",
    DELTA_PATH
)