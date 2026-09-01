# ============================================================
# FleetCare AI
# Model 6: Multi-Readout Historical XGBoost
#
# Purpose:
# Compare richer historical features against the previous
# current + previous + delta + rate model.
#
# Target:
# C0 / C1 / C2 / C3 / C4
#
# Training:
# Vehicle-aware sampling
# No class weighting
# ============================================================

from pathlib import Path

import gc
import numpy as np
import pandas as pd

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
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIG
# ============================================================

RANDOM_STATE = 42

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 250)


# ============================================================
# 2. PATHS
# ============================================================

DATA_DIR = Path("../data/raw")
OUTPUT_DIR = Path("../data/processed")

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
    VAL_LABEL_PATH
]

print("=" * 80)
print("CHECKING FILES")
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
# 4. LOAD DATA
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
# 5. BASIC CHECKS
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
# 6. RAW FEATURES
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
# 7. SORT
# ============================================================

train_op = (
    train_op
    .sort_values(
        [
            "vehicle_id",
            "time_step"
        ]
    )
    .reset_index(drop=True)
)

val_op = (
    val_op
    .sort_values(
        [
            "vehicle_id",
            "time_step"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 8. MULTI-READOUT HISTORICAL FEATURES
# ============================================================

def create_history_features(
    df,
    feature_cols
):

    print(
        "  Sorting..."
    )

    df = (
        df
        .sort_values(
            [
                "vehicle_id",
                "time_step"
            ]
        )
        .reset_index(drop=True)
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
        "  Creating time gaps..."
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
    # PREVIOUS
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
    # CURRENT ARRAY
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
    # RATE
    # ========================================================

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


    # ========================================================
    # CHANGE FROM 3 READOUTS AGO
    # ========================================================

    print(
        "  Creating change-from-3..."
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
        "  Creating change-from-5..."
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
    # ROLLING MEAN 3
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
    # ROLLING STD 3
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
    # ROLLING MEAN 5
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
    # ROLLING STD 5
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
# 9. CREATE HISTORY FEATURES
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


print("\n" + "=" * 80)
print("CREATING VALIDATION HISTORY FEATURES")
print("=" * 80)

val_history = (
    create_history_features(
        val_op,
        RAW_FEATURE_COLS
    )
)


# Raw data no longer needed

del train_op
del val_op

gc.collect()


# ============================================================
# 10. MERGE TTE
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
# 11. TIME TO ENDPOINT
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
# 12. C0-C4 FUNCTION
# ============================================================

def tte_to_class(
    tte
):

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


# ============================================================
# 13. LABEL TRAINING OBSERVATIONS
# ============================================================

train_full[
    "class_label"
] = np.nan


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


# Safely known C0 for censored vehicles

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
# 14. ELIGIBLE OBSERVATIONS
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
print("ALL ELIGIBLE OBSERVATIONS")
print("=" * 80)

print(
    candidate_train[
        "class_label"
    ]
    .value_counts()
    .sort_index()
)


# We no longer need train_full

del train_full

gc.collect()


# ============================================================
# 15. VEHICLE-AWARE SAMPLING
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


# Repair vehicles:
# max one sample per vehicle per class

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
    .reset_index(drop=True)
)


# Non-repair:
# one safe C0 per vehicle

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
    .reset_index(drop=True)
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
    .reset_index(drop=True)
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


# Free huge candidate table

del candidate_train
del repair_candidates
del non_repair_candidates
del repair_sampled
del non_repair_sampled

gc.collect()


# ============================================================
# 16. JOIN TRAIN SPECIFICATIONS
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
# 17. VALIDATION — LAST AVAILABLE READOUT
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
# 18. BUILD MULTICLASS X/Y
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
# 19. ALIGN
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
    "\nTrain-only:",
    train_only
)

print(
    "Validation-only:",
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


print(
    "✓ Feature schemas aligned."
)


# ============================================================
# 20. FEATURE TYPES
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
    "Total input features:",
    len(
        X_train.columns
    )
)


# ============================================================
# 21. PREPROCESSING
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
# 22. MULTICLASS HISTORY XGBOOST
# ============================================================
#
# IMPORTANT:
# Same idea as Model C:
# NO class weights.
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
# 23. TRAIN MULTICLASS
# ============================================================

print("\n" + "=" * 80)
print("TRAINING MULTI-READOUT HISTORY XGBOOST")
print("=" * 80)


history_pipeline.fit(
    X_train,
    y_train
)




print(
    "✓ Multiclass training complete"
)


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Get fitted preprocessor and XGBoost model
fitted_preprocessor = (
    history_pipeline.named_steps["preprocessor"]
)

fitted_model = (
    history_pipeline.named_steps["classifier"]
)


# 2. Transform validation data exactly as during training
X_val_transformed = (
    fitted_preprocessor.transform(X_val)
)


# 3. Get transformed feature names
feature_names = (
    fitted_preprocessor.get_feature_names_out()
)

# Clean sklearn prefixes to make plots easier to read
feature_names = [
    name
    .replace("numeric__", "")
    .replace("categorical__", "")
    for name in feature_names
]


print("Transformed validation shape:")
print(X_val_transformed.shape)

print("Number of feature names:")
print(len(feature_names))


# ============================================================
# 4. SAMPLE VALIDATION DATA
#
# Don't explain all 5046 vehicles initially.
# Start with 300 to keep SHAP fast.
# ============================================================

SHAP_SAMPLE_SIZE = min(
    300,
    X_val_transformed.shape[0]
)

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    X_val_transformed.shape[0],
    size=SHAP_SAMPLE_SIZE,
    replace=False
)

X_shap = X_val_transformed[
    sample_indices
]


# Convert sparse matrix if necessary
if hasattr(X_shap, "toarray"):
    X_shap = X_shap.toarray()


# ============================================================
# 5. CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    fitted_model
)

shap_values = explainer(
    X_shap
)

print("SHAP values shape:")
print(shap_values.values.shape)

# ============================================================
# 24. MULTICLASS PREDICTION
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


# ============================================================
# 25. MULTICLASS REPORT
# ============================================================

print("\n" + "=" * 80)
print("MULTI-READOUT MULTICLASS REPORT")
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


# ============================================================
# 26. MULTICLASS METRICS
# ============================================================

multiclass_accuracy = (
    accuracy_score(
        y_val,
        y_pred
    )
)


multiclass_macro_f1 = (
    f1_score(
        y_val,
        y_pred,
        average="macro"
    )
)


multiclass_report = (
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
    multiclass_report[
        "C4"
    ][
        "precision"
    ]
)


c4_recall = (
    multiclass_report[
        "C4"
    ][
        "recall"
    ]
)


c4_f1 = (
    multiclass_report[
        "C4"
    ][
        "f1-score"
    ]
)


print(
    "Accuracy:",
    round(
        multiclass_accuracy,
        4
    )
)

print(
    "Macro F1:",
    round(
        multiclass_macro_f1,
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
# 27. COMPARE WITH PREVIOUS MODEL C
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Previous Temporal Model",
        "Multi-Readout History Model"
    ],

    "Accuracy": [
        0.7527,
        multiclass_accuracy
    ],

    "Macro F1": [
        0.1883,
        multiclass_macro_f1
    ],

    "C4 Precision": [
        0.0384,
        c4_precision
    ],

    "C4 Recall": [
        0.2632,
        c4_recall
    ],

    "C4 F1": [
        0.0670,
        c4_f1
    ]
})


print("\n" + "=" * 80)
print("HISTORY FEATURE COMPARISON")
print("=" * 80)


print(
    comparison
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 28. BINARY DIAGNOSTIC
#
# C0     -> 0
# C1-C4  -> 1
#
# Use the SAME historical features.
# ============================================================

print("\n" + "=" * 80)
print("BINARY HISTORY MODEL")
print("=" * 80)


y_train_binary = (
    y_train
    > 0
).astype(int)


y_val_binary = (
    y_val
    > 0
).astype(int)


binary_history_xgb = XGBClassifier(
    objective="binary:logistic",

    n_estimators=150,
    learning_rate=0.08,

    max_depth=5,
    min_child_weight=3,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=0.0,
    reg_lambda=1.0,

    eval_metric="logloss",
    tree_method="hist",

    random_state=RANDOM_STATE,
    n_jobs=-1
)


binary_history_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        binary_history_xgb
    )
])


binary_history_pipeline.fit(
    X_train,
    y_train_binary
)


risk_probability = (
    binary_history_pipeline
    .predict_proba(
        X_val
    )[:, 1]
)


binary_roc_auc = (
    roc_auc_score(
        y_val_binary,
        risk_probability
    )
)


binary_pr_auc = (
    average_precision_score(
        y_val_binary,
        risk_probability
    )
)


print(
    "Binary ROC-AUC:",
    round(
        binary_roc_auc,
        4
    )
)

print(
    "Binary PR-AUC:",
    round(
        binary_pr_auc,
        4
    )
)


# ============================================================
# 29. BINARY THRESHOLDS
# ============================================================

thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


threshold_results = []


for threshold in thresholds:

    pred_binary = (
        risk_probability
        >=
        threshold
    ).astype(int)


    precision = (
        precision_score(
            y_val_binary,
            pred_binary,
            zero_division=0
        )
    )


    recall = (
        recall_score(
            y_val_binary,
            pred_binary,
            zero_division=0
        )
    )


    binary_f1 = (
        f1_score(
            y_val_binary,
            pred_binary,
            zero_division=0
        )
    )


    binary_accuracy = (
        accuracy_score(
            y_val_binary,
            pred_binary
        )
    )


    cm = confusion_matrix(
        y_val_binary,
        pred_binary
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    threshold_results.append({

        "Threshold":
            threshold,

        "Accuracy":
            binary_accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            binary_f1,

        "TP":
            tp,

        "FP":
            fp,

        "FN":
            fn,

        "TN":
            tn
    })


threshold_df = pd.DataFrame(
    threshold_results
)


print("\n" + "=" * 80)
print("BINARY THRESHOLD COMPARISON")
print("=" * 80)


print(
    threshold_df
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 30. SAVE RESULTS
# ============================================================

comparison.to_csv(
    OUTPUT_DIR
    /
    "model_06_history_comparison.csv",
    index=False
)


threshold_df.to_csv(
    OUTPUT_DIR
    /
    "model_06_binary_thresholds.csv",
    index=False
)


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
        ],

    "binary_risk_probability":
        risk_probability
})


validation_predictions.to_csv(
    OUTPUT_DIR
    /
    "model_06_history_predictions.csv",
    index=False
)


# ============================================================
# 31. DONE
# ============================================================

print("\n" + "=" * 80)
print("MODEL 6 COMPLETE")
print("=" * 80)

print(
    "Previous Macro F1: 0.1883"
)

print(
    "New Macro F1:",
    round(
        multiclass_macro_f1,
        4
    )
)

print(
    "Previous C4 Precision: 0.0384"
)

print(
    "New C4 Precision:",
    round(
        c4_precision,
        4
    )
)

print(
    "Previous C4 Recall: 0.2632"
)

print(
    "New C4 Recall:",
    round(
        c4_recall,
        4
    )
)

print(
    "Binary ROC-AUC:",
    round(
        binary_roc_auc,
        4
    )
)

print(
    "Binary PR-AUC:",
    round(
        binary_pr_auc,
        4
    )
)

print(
    "\nResults saved to:",
    OUTPUT_DIR
)