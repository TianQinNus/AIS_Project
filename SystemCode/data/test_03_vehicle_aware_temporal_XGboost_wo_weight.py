# ============================================================
# FleetCare AI
# Model C: Vehicle-Aware Temporal XGBoost
# WITHOUT Class Weights
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

pd.set_option("display.max_columns", 150)
pd.set_option("display.max_rows", 100)


# ============================================================
# 2. PATHS
# ============================================================
#
# Current script:
#
# SystemCode/data/test_03.py
#
# Dataset:
#
# project_root/data/raw/
#
# Therefore:
# __file__
#   → data/
#   → SystemCode/
#   → AIS_Project/
#
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = (
    SCRIPT_DIR
    .parent
    .parent
)

# ============================================================

DATA_DIR = Path("../data/raw")

TRAIN_OP_PATH = DATA_DIR / "train_operational_readouts.csv"
TRAIN_SPEC_PATH = DATA_DIR / "train_specifications.csv"
TRAIN_TTE_PATH = DATA_DIR / "train_tte.csv"

VAL_OP_PATH = DATA_DIR / "validation_operational_readouts.csv"
VAL_SPEC_PATH = DATA_DIR / "validation_specifications.csv"
VAL_LABEL_PATH = DATA_DIR / "validation_labels.csv"

OUTPUT_DIR = Path("../data/processed")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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

print("=" * 70)
print("CHECKING DATA FILES")
print("=" * 70)

for path in required_files:

    print(
        path.name,
        "->",
        path.exists()
    )

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: {path}"
        )


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

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
# 5. BASIC VEHICLE CHECKS
# ============================================================

train_vehicle_count = (
    train_op["vehicle_id"]
    .nunique()
)

validation_vehicle_count = (
    val_op["vehicle_id"]
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
    train_spec["vehicle_id"].nunique()
)

assert (
    train_vehicle_count
    ==
    train_tte["vehicle_id"].nunique()
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
# 7. SORT TIME SERIES
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
# 8. TEMPORAL FEATURE ENGINEERING
# ============================================================
#
# For each operational variable:
#
# current:
#     x(t)
#
# previous:
#     x(t-1)
#
# delta:
#     x(t) - x(t-1)
#
# rate:
#
#     x(t) - x(t-1)
#     ----------------
#       t - t_previous
#
# The rate is useful because SCANIA readouts are irregularly
# sampled.
#
# ============================================================

def create_temporal_features(
    df,
    feature_cols
):

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

    # --------------------------------------------------------
    # Time gap
    # --------------------------------------------------------

    df["time_gap"] = (
        grouped["time_step"]
        .diff()
    )

    safe_time_gap = (
        df["time_gap"]
        .replace(
            0,
            np.nan
        )
    )

    # --------------------------------------------------------
    # Previous values
    # --------------------------------------------------------

    previous = (
        grouped[
            feature_cols
        ]
        .shift(1)
    )

    previous.columns = [

        f"{column}__prev"

        for column in feature_cols
    ]


    # --------------------------------------------------------
    # Current / previous arrays
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

    delta_array = (
        current_array
        -
        previous_array
    )


    delta = pd.DataFrame(

        delta_array,

        index=df.index,

        columns=[

            f"{column}__delta"

            for column in feature_cols
        ]
    )


    # --------------------------------------------------------
    # Rate of change
    # --------------------------------------------------------

    time_gap_array = (
        safe_time_gap
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
            time_gap_array
        )


    rate = pd.DataFrame(

        rate_array,

        index=df.index,

        columns=[

            f"{column}__rate"

            for column in feature_cols
        ]
    )


    rate = rate.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    result = pd.concat(

        [
            df,
            previous,
            delta,
            rate
        ],

        axis=1
    )


    return result


# ============================================================
# 9. CREATE TEMPORAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING TEMPORAL FEATURES")
print("=" * 70)


print(
    "Creating training temporal features..."
)

train_temporal = (
    create_temporal_features(
        train_op,
        RAW_FEATURE_COLS
    )
)

print(
    "Training temporal shape:",
    train_temporal.shape
)


print(
    "\nCreating validation temporal features..."
)

val_temporal = (
    create_temporal_features(
        val_op,
        RAW_FEATURE_COLS
    )
)

print(
    "Validation temporal shape:",
    val_temporal.shape
)


# ============================================================
# 10. MERGE TRAINING DATA WITH TTE
# ============================================================

train_full = (
    train_temporal
    .merge(
        train_tte,
        on="vehicle_id",
        how="left",
        validate="many_to_one"
    )
)


# ============================================================
# 11. CALCULATE TIME TO ENDPOINT
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
# 12. CLASS FUNCTION
# ============================================================

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


# ============================================================
# 13. CREATE TRAINING LABELS
# ============================================================

train_full[
    "class_label"
] = np.nan


# ------------------------------------------------------------
# Vehicles with observed repair
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
# Vehicles without observed repair
#
# Only use C0 observations when we KNOW at least another
# 48 time units were observed without repair.
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
# 14. REMOVE UNKNOWN / CENSORED LABELS
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


print("\n" + "=" * 70)
print("ALL ELIGIBLE TRAINING OBSERVATIONS")
print("=" * 70)


print(

    candidate_train[
        "class_label"
    ]

    .value_counts()

    .sort_index()
)


# ============================================================
# 15. VEHICLE-AWARE SAMPLING
# ============================================================
#
# Repair vehicle:
#
#     maximum one readout for each available class.
#
# Non-repair vehicle:
#
#     maximum one safe C0 readout.
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


# ============================================================
# 16. SAMPLE REPAIR VEHICLES
# ============================================================

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


# ============================================================
# 17. SAMPLE NON-REPAIR VEHICLES
# ============================================================

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


# ============================================================
# 18. COMBINE TRAINING SAMPLES
# ============================================================

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


print("\n" + "=" * 70)
print("VEHICLE-AWARE TRAINING DISTRIBUTION")
print("=" * 70)


print(

    train_sampled[
        "class_label"
    ]

    .value_counts()

    .sort_index()
)


print(
    "\nTraining rows:",
    len(train_sampled)
)


print(
    "Unique training vehicles:",
    train_sampled[
        "vehicle_id"
    ].nunique()
)


assert set(

    train_sampled[
        "class_label"
    ].unique()

) == {

    0,
    1,
    2,
    3,
    4
}


# ============================================================
# 19. JOIN SPECIFICATIONS
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
# 20. BUILD VALIDATION TABLE
# ============================================================
#
# Validation data have already been truncated by the
# challenge organisers.
#
# Therefore:
#
# use LAST AVAILABLE READOUT for each vehicle.
#
# ============================================================

val_last = (

    val_temporal

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


print("\n" + "=" * 70)
print("VALIDATION DISTRIBUTION")
print("=" * 70)


print(

    val_last[
        "class_label"
    ]

    .value_counts()

    .sort_index()
)


# ============================================================
# 21. DEFINE FEATURES
# ============================================================

NON_FEATURE_COLS = [

    "vehicle_id",

    "class_label",

    # Future / outcome information
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
# 22. ALIGN FEATURES
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


assert list(
    X_train.columns
) == list(
    X_val.columns
)


print(
    "✓ Train and validation features aligned."
)


# ============================================================
# 23. FEATURE TYPES
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
    len(numeric_cols)
)


print(
    "Categorical features:",
    len(categorical_cols)
)


print(
    "Total input features:",
    len(X_train.columns)
)


# ============================================================
# 24. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_transformer = Pipeline(

    steps=[

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
    ]
)


preprocessor = ColumnTransformer(

    transformers=[

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
    ]
)


# ============================================================
# 25. MODEL C
# ============================================================
#
# Vehicle-aware
# +
# Temporal features
# +
# XGBoost
# +
# NO CLASS WEIGHTS
#
# ============================================================

model_c = XGBClassifier(

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


model_c_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            model_c
        )
    ]
)


# ============================================================
# 26. TRAIN MODEL C
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MODEL C")
print("Vehicle-Aware Temporal XGBoost")
print("NO CLASS WEIGHTS")
print("=" * 70)


print(
    "\nTraining shape:",
    X_train.shape
)


print(
    "Validation shape:",
    X_val.shape
)


print(
    "\nTraining distribution:"
)


print(

    y_train

    .value_counts()

    .sort_index()
)


print(
    "\nTraining Model C..."
)


# IMPORTANT:
#
# NO sample_weight here.
#

model_c_pipeline.fit(

    X_train,

    y_train
)


print(
    "✓ Model C training complete"
)


# ============================================================
# 27. PREDICT
# ============================================================

y_pred = (

    model_c_pipeline

    .predict(
        X_val
    )

    .astype(int)
)


y_proba = (

    model_c_pipeline

    .predict_proba(
        X_val
    )
)


assert y_proba.shape == (

    len(
        X_val
    ),

    5
)


# ============================================================
# 28. PREDICTED CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTED CLASS DISTRIBUTION")
print("=" * 70)


predicted_distribution = (

    pd.Series(

        y_pred,

        name="predicted_class"
    )

    .value_counts()

    .sort_index()
)


print(
    predicted_distribution
)


# ============================================================
# 29. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("MODEL C CLASSIFICATION REPORT")
print("=" * 70)


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
# 30. OVERALL METRICS
# ============================================================

accuracy = accuracy_score(

    y_val,

    y_pred
)


macro_f1 = f1_score(

    y_val,

    y_pred,

    average="macro"
)


weighted_f1 = f1_score(

    y_val,

    y_pred,

    average="weighted"
)


print(
    "Model C Accuracy:",
    round(
        accuracy,
        4
    )
)


print(
    "Model C Macro F1:",
    round(
        macro_f1,
        4
    )
)


print(
    "Model C Weighted F1:",
    round(
        weighted_f1,
        4
    )
)


# ============================================================
# 31. CONFUSION MATRIX
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


disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "C0",
        "C1",
        "C2",
        "C3",
        "C4"
    ]
)


disp.plot(
    values_format="d"
)


plt.title(
    "Model C\n"
    "Vehicle-Aware Temporal XGBoost"
)


plt.tight_layout()

plt.show()


# ============================================================
# 32. PER-CLASS PERFORMANCE
# ============================================================

report = classification_report(

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


performance = pd.DataFrame({

    "Class": [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4"
    ],

    "Precision": [
        report["C0"]["precision"],
        report["C1"]["precision"],
        report["C2"]["precision"],
        report["C3"]["precision"],
        report["C4"]["precision"]
    ],

    "Recall": [
        report["C0"]["recall"],
        report["C1"]["recall"],
        report["C2"]["recall"],
        report["C3"]["recall"],
        report["C4"]["recall"]
    ],

    "F1": [
        report["C0"]["f1-score"],
        report["C1"]["f1-score"],
        report["C2"]["f1-score"],
        report["C3"]["f1-score"],
        report["C4"]["f1-score"]
    ],

    "Support": [
        report["C0"]["support"],
        report["C1"]["support"],
        report["C2"]["support"],
        report["C3"]["support"],
        report["C4"]["support"]
    ]
})


print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)


print(
    performance.round(4)
)


# ============================================================
# 33. PROBABILITY ANALYSIS
# ============================================================

proba_df = pd.DataFrame(

    y_proba,

    columns=[
        "P_C0",
        "P_C1",
        "P_C2",
        "P_C3",
        "P_C4"
    ]
)


proba_df[
    "actual_class"
] = y_val.to_numpy()


proba_df[
    "predicted_class"
] = y_pred


mean_probability = (

    proba_df

    .groupby(
        "actual_class"
    )

    [
        [
            "P_C0",
            "P_C1",
            "P_C2",
            "P_C3",
            "P_C4"
        ]
    ]

    .mean()
)


print("\n" + "=" * 70)
print("MEAN PREDICTED PROBABILITY BY ACTUAL CLASS")
print("=" * 70)


print(
    mean_probability.round(4)
)


# ============================================================
# 34. C4 PROBABILITY SEPARATION
# ============================================================

actual_array = (
    y_val.to_numpy()
)


actual_c0_mask = (
    actual_array == 0
)


actual_c4_mask = (
    actual_array == 4
)


p_c4_actual_c0 = (

    y_proba[
        actual_c0_mask,
        4
    ]
)


p_c4_actual_c4 = (

    y_proba[
        actual_c4_mask,
        4
    ]
)


mean_p_c4_c0 = (
    p_c4_actual_c0.mean()
)


mean_p_c4_c4 = (
    p_c4_actual_c4.mean()
)


print("\n" + "=" * 70)
print("C4 PROBABILITY SEPARATION")
print("=" * 70)


print(
    "Mean P(C4) | Actual C0:",
    round(
        mean_p_c4_c0,
        4
    )
)


print(
    "Mean P(C4) | Actual C4:",
    round(
        mean_p_c4_c4,
        4
    )
)


print(
    "Difference:",
    round(
        mean_p_c4_c4
        -
        mean_p_c4_c0,
        4
    )
)


# ============================================================
# 35. MODEL COMPARISON
# ============================================================
#
# A:
# All-readout Temporal XGBoost
#
# B:
# Vehicle-aware + class weights
#
# C:
# Vehicle-aware + NO class weights
#
# ============================================================

comparison = pd.DataFrame({

    "Model": [

        "A: All-readout Temporal XGBoost",

        "B: Vehicle-aware + Class Weights",

        "C: Vehicle-aware, No Class Weights"
    ],

    "Accuracy": [

        0.9556,

        0.2218,

        accuracy
    ],

    "Macro F1": [

        0.2041,

        0.0851,

        macro_f1
    ],

    "C4 Precision": [

        0.1111,

        0.0221,

        report[
            "C4"
        ][
            "precision"
        ]
    ],

    "C4 Recall": [

        0.0263,

        0.2895,

        report[
            "C4"
        ][
            "recall"
        ]
    ],

    "C4 F1": [

        0.0426,

        0.0410,

        report[
            "C4"
        ][
            "f1-score"
        ]
    ]
})


print("\n" + "=" * 70)
print("MODEL A / B / C COMPARISON")
print("=" * 70)


print(
    comparison.round(4)
)


# ============================================================
# 36. SAVE PREDICTIONS
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
        y_proba[:, 0],

    "P_C1":
        y_proba[:, 1],

    "P_C2":
        y_proba[:, 2],

    "P_C3":
        y_proba[:, 3],

    "P_C4":
        y_proba[:, 4]
})


OUTPUT_FILE = (

    OUTPUT_DIR

    /

    "model_c_validation_predictions.csv"
)


validation_predictions.to_csv(

    OUTPUT_FILE,

    index=False
)


print(
    "\nPredictions saved to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# END
# ============================================================