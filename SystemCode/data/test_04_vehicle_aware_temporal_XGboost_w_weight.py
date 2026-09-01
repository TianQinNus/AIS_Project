# ============================================================
# FleetCare AI
# Model 4: Vehicle-Aware Temporal XGBoost
# Partial Class-Weight Tuning
# ============================================================

from pathlib import Path

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
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 250)


# ============================================================
# 2. PATHS
# ============================================================

DATA_DIR = Path("../data/raw")

TRAIN_OP_PATH = DATA_DIR / "train_operational_readouts.csv"
TRAIN_SPEC_PATH = DATA_DIR / "train_specifications.csv"
TRAIN_TTE_PATH = DATA_DIR / "train_tte.csv"

VAL_OP_PATH = DATA_DIR / "validation_operational_readouts.csv"
VAL_SPEC_PATH = DATA_DIR / "validation_specifications.csv"
VAL_LABEL_PATH = DATA_DIR / "validation_labels.csv"

OUTPUT_DIR = Path("../data/processed")

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
    train_op["vehicle_id"].nunique()
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

print(
    "\nTraining vehicles:",
    train_vehicle_count
)

print(
    "Validation vehicles:",
    val_op["vehicle_id"].nunique()
)


# ============================================================
# 6. RAW OPERATIONAL FEATURES
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
# 7. SORT DATA
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
        f"{col}__prev"
        for col in feature_cols
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
            f"{col}__delta"
            for col in feature_cols
        ]
    )

    # --------------------------------------------------------
    # Rate
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
            f"{col}__rate"
            for col in feature_cols
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

print("\n" + "=" * 80)
print("CREATING TEMPORAL FEATURES")
print("=" * 80)

print(
    "Training temporal features..."
)

train_temporal = (
    create_temporal_features(
        train_op,
        RAW_FEATURE_COLS
    )
)

print(
    "Train temporal shape:",
    train_temporal.shape
)

print(
    "Validation temporal features..."
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
# 10. MERGE TTE
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
# 13. CREATE LABELS
# ============================================================

train_full[
    "class_label"
] = np.nan


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


# Non-repair vehicles
# Only safely known C0 observations

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
# 14. REMOVE UNKNOWN LABELS
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


# ------------------------------------------------------------
# Repair:
# max one observation per vehicle per class
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
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Non-repair:
# one safe C0 per vehicle
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
    .reset_index(drop=True)
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
    len(train_sampled)
)

print(
    "Unique vehicles:",
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
# 16. JOIN SPECIFICATIONS
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
# 17. BUILD VALIDATION TABLE
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
# 18. DEFINE FEATURES
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
# 19. ALIGN FEATURES
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

assert list(
    X_train.columns
) == list(
    X_val.columns
)

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
    len(numeric_cols)
)

print(
    "Categorical features:",
    len(categorical_cols)
)

print(
    "Total features:",
    len(X_train.columns)
)


# ============================================================
# 21. PREPROCESSOR
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
# 22. OFFICIAL IDA COST MATRIX
# ============================================================
#
# Row    = Actual
# Column = Predicted
#
# ============================================================

COST_MATRIX = np.array([
    [0,   7,   8,   9,  10],
    [200, 0,   7,   8,   9],
    [300, 200, 0,   7,   8],
    [400, 300, 200, 0,   7],
    [500, 400, 300, 200, 0]
])


def calculate_total_cost(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int
    )

    return int(
        COST_MATRIX[
            y_true,
            y_pred
        ]
        .sum()
    )


# ============================================================
# 23. WEIGHT STRATEGIES
# ============================================================
#
# W0 = Model C
#
# Then progressively increase minority-class importance.
#
# ============================================================

WEIGHT_STRATEGIES = {

    "W0_NoWeight": {
        0: 1.00,
        1: 1.00,
        2: 1.00,
        3: 1.00,
        4: 1.00
    },

    "W1_Light": {
        0: 1.00,
        1: 1.25,
        2: 1.50,
        3: 1.75,
        4: 2.00
    },

    "W2_Moderate": {
        0: 1.00,
        1: 1.50,
        2: 2.00,
        3: 2.50,
        4: 3.00
    },

    "W3_Stronger": {
        0: 1.00,
        1: 2.00,
        2: 2.50,
        3: 3.00,
        4: 4.00
    },

    "W4_RiskFocused": {
        0: 1.00,
        1: 2.00,
        2: 3.00,
        3: 4.00,
        4: 5.00
    }
}


# ============================================================
# 24. TRAIN ALL STRATEGIES
# ============================================================

results = []

predictions = {}

probabilities = {}

trained_models = {}


print("\n" + "=" * 80)
print("PARTIAL CLASS-WEIGHT EXPERIMENT")
print("=" * 80)

print(
    "Training rows:",
    len(X_train)
)

print(
    "Validation rows:",
    len(X_val)
)


for (
    strategy_name,
    weight_map
) in WEIGHT_STRATEGIES.items():

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "TRAINING:",
        strategy_name
    )

    print(
        "Weights:",
        weight_map
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Row-level weights
    # --------------------------------------------------------

    sample_weights = (
        y_train
        .map(
            weight_map
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # SAME model architecture for every strategy
    # --------------------------------------------------------

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

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                model
            )
        ]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train,
        classifier__sample_weight=sample_weights
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = (
        pipeline
        .predict(
            X_val
        )
        .astype(int)
    )

    y_proba = (
        pipeline
        .predict_proba(
            X_val
        )
    )

    trained_models[
        strategy_name
    ] = pipeline

    predictions[
        strategy_name
    ] = y_pred

    probabilities[
        strategy_name
    ] = y_proba

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    macro_precision = precision_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_val,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Official challenge cost
    # --------------------------------------------------------

    total_cost = (
        calculate_total_cost(
            y_val,
            y_pred
        )
    )

    average_cost = (
        total_cost
        /
        len(y_val)
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    pred_counts = (
        pd.Series(
            y_pred
        )
        .value_counts()
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.append({

        "Strategy":
            strategy_name,

        "Accuracy":
            accuracy,

        "Macro Precision":
            macro_precision,

        "Macro Recall":
            macro_recall,

        "Macro F1":
            macro_f1,

        "Weighted F1":
            weighted_f1,

        "C0 Recall":
            report[
                "C0"
            ][
                "recall"
            ],

        "C1 Recall":
            report[
                "C1"
            ][
                "recall"
            ],

        "C2 Recall":
            report[
                "C2"
            ][
                "recall"
            ],

        "C3 Recall":
            report[
                "C3"
            ][
                "recall"
            ],

        "C4 Recall":
            report[
                "C4"
            ][
                "recall"
            ],

        "C4 Precision":
            report[
                "C4"
            ][
                "precision"
            ],

        "C4 F1":
            report[
                "C4"
            ][
                "f1-score"
            ],

        "Pred C0":
            int(
                pred_counts.get(
                    0,
                    0
                )
            ),

        "Pred C1":
            int(
                pred_counts.get(
                    1,
                    0
                )
            ),

        "Pred C2":
            int(
                pred_counts.get(
                    2,
                    0
                )
            ),

        "Pred C3":
            int(
                pred_counts.get(
                    3,
                    0
                )
            ),

        "Pred C4":
            int(
                pred_counts.get(
                    4,
                    0
                )
            ),

        "Total Cost":
            total_cost,

        "Average Cost":
            average_cost
    })

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
            report[
                "C4"
            ][
                "precision"
            ],
            4
        )
    )

    print(
        "C4 Recall:",
        round(
            report[
                "C4"
            ][
                "recall"
            ],
            4
        )
    )

    print(
        "C4 F1:",
        round(
            report[
                "C4"
            ][
                "f1-score"
            ],
            4
        )
    )

    print(
        "Total Cost:",
        total_cost
    )


# ============================================================
# 25. COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

display_cols = [
    "Strategy",
    "Accuracy",
    "Macro F1",
    "C0 Recall",
    "C1 Recall",
    "C2 Recall",
    "C3 Recall",
    "C4 Recall",
    "C4 Precision",
    "C4 F1",
    "Pred C0",
    "Pred C1",
    "Pred C2",
    "Pred C3",
    "Pred C4",
    "Total Cost"
]


print("\n" + "=" * 80)
print("FINAL WEIGHT COMPARISON")
print("=" * 80)

print(
    results_df[
        display_cols
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 26. BEST BY MACRO F1
# ============================================================

best_macro_row = (
    results_df
    .sort_values(
        "Macro F1",
        ascending=False
    )
    .iloc[0]
)

best_macro_strategy = (
    best_macro_row[
        "Strategy"
    ]
)


print("\n" + "=" * 80)
print("BEST STRATEGY BY MACRO F1")
print("=" * 80)

print(
    "Strategy:",
    best_macro_strategy
)

print(
    "Accuracy:",
    round(
        best_macro_row[
            "Accuracy"
        ],
        4
    )
)

print(
    "Macro F1:",
    round(
        best_macro_row[
            "Macro F1"
        ],
        4
    )
)

print(
    "C4 Recall:",
    round(
        best_macro_row[
            "C4 Recall"
        ],
        4
    )
)

print(
    "Total Cost:",
    int(
        best_macro_row[
            "Total Cost"
        ]
    )
)


# ============================================================
# 27. BEST BY OFFICIAL COST
# ============================================================

best_cost_row = (
    results_df
    .sort_values(
        "Total Cost",
        ascending=True
    )
    .iloc[0]
)

best_cost_strategy = (
    best_cost_row[
        "Strategy"
    ]
)


print("\n" + "=" * 80)
print("BEST STRATEGY BY OFFICIAL COST")
print("=" * 80)

print(
    "Strategy:",
    best_cost_strategy
)

print(
    "Total Cost:",
    int(
        best_cost_row[
            "Total Cost"
        ]
    )
)

print(
    "Accuracy:",
    round(
        best_cost_row[
            "Accuracy"
        ],
        4
    )
)

print(
    "Macro F1:",
    round(
        best_cost_row[
            "Macro F1"
        ],
        4
    )
)

print(
    "C4 Recall:",
    round(
        best_cost_row[
            "C4 Recall"
        ],
        4
    )
)


# ============================================================
# 28. CLASSIFICATION REPORT
#     BEST MACRO-F1 MODEL
# ============================================================

best_pred = (
    predictions[
        best_macro_strategy
    ]
)


print("\n" + "=" * 80)
print(
    "CLASSIFICATION REPORT —",
    best_macro_strategy
)
print("=" * 80)

print(
    classification_report(
        y_val,
        best_pred,

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
# 29. CONFUSION MATRIX
# ============================================================

best_cm = confusion_matrix(
    y_val,
    best_pred,

    labels=[
        0,
        1,
        2,
        3,
        4
    ]
)


print("\n" + "=" * 80)
print(
    "CONFUSION MATRIX —",
    best_macro_strategy
)
print("=" * 80)

print(
    best_cm
)


# ============================================================
# 30. SAVE RESULTS
# ============================================================

RESULTS_PATH = (
    OUTPUT_DIR
    /
    "model_04_weight_tuning_results.csv"
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# 31. SAVE BEST MODEL PREDICTIONS
# ============================================================

best_proba = (
    probabilities[
        best_macro_strategy
    ]
)

best_predictions_df = pd.DataFrame({

    "vehicle_id":
        val_last[
            "vehicle_id"
        ]
        .to_numpy(),

    "actual_class":
        y_val
        .to_numpy(),

    "predicted_class":
        best_pred,

    "P_C0":
        best_proba[
            :,
            0
        ],

    "P_C1":
        best_proba[
            :,
            1
        ],

    "P_C2":
        best_proba[
            :,
            2
        ],

    "P_C3":
        best_proba[
            :,
            3
        ],

    "P_C4":
        best_proba[
            :,
            4
        ]
})


PREDICTIONS_PATH = (
    OUTPUT_DIR
    /
    "model_04_best_validation_predictions.csv"
)

best_predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# 32. DONE
# ============================================================

print("\n" + "=" * 80)
print("MODEL 4 COMPLETE")
print("=" * 80)

print(
    "Results saved to:"
)

print(
    RESULTS_PATH
)

print(
    "\nBest predictions saved to:"
)

print(
    PREDICTIONS_PATH
)