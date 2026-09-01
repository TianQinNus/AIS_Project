# ============================================================
# FleetCare AI
# Model 8: Cumulative Risk + Capacity-Constrained Ranking
# ============================================================
#
# What Test 07 already fixed:
#
#   Base classifier:
#       Vehicle-Aware Temporal XGBoost
#       + W1 Light Class Weighting
#
#   Calibration method:
#       LogitCalibrated
#
# Test 08 does NOT try to select another classifier/calibrator.
#
# Instead, Test 08 asks the deployment question:
#
#   "If a fleet manager can inspect only a limited fraction of
#    the fleet, can calibrated risk scores rank the truly urgent
#    vehicles near the top?"
#
# ------------------------------------------------------------
# Main idea
# ------------------------------------------------------------
#
# The C0-C4 labels are ordered repair-proximity classes:
#
#   C0: >48
#   C1: 24-48
#   C2: 12-24
#   C3: 6-12
#   C4: 0-6
#
# From calibrated class probabilities we construct cumulative
# horizon risks:
#
#   Risk48 = P(C1)+P(C2)+P(C3)+P(C4) = 1-P(C0)
#   Risk24 = P(C2)+P(C3)+P(C4)
#   Risk12 = P(C3)+P(C4)
#   Risk6  = P(C4)
#
# We then rank vehicles by each score and evaluate fixed
# inspection budgets:
#
#   Top 1%, 2%, 5%, 10%, 20%
#
# ------------------------------------------------------------
# Important experimental rule
# ------------------------------------------------------------
#
# We keep the Test 07 three-way vehicle split:
#
#   70% Base Train
#   15% Calibration
#   15% Selection/Policy Dev
#
# Calibration is fitted ONLY on the Calibration split.
#
# Selection/Policy Dev is converted into a deployment-like
# synthetic fleet snapshot:
#
#   ONE safely-labelled cutoff per vehicle
#
# so the same vehicle cannot appear multiple times in a Top-K
# inspection list.
#
# Official validation is used only as an external diagnostic.
#
# ------------------------------------------------------------
# Primary calibration
# ------------------------------------------------------------
#
# LogitCalibrated is PRIMARY because Test 07 selected it on
# Selection Dev across 5/5 seeds.
#
# Raw and PriorCorrected are retained only as sensitivity
# references. They are NOT allowed to replace the selected
# calibrator based on official validation.
#
# ------------------------------------------------------------
# What Test 08 does NOT do
# ------------------------------------------------------------
#
# It does NOT automatically choose "Top 5%" or any other budget,
# because the real inspection capacity has not yet been supplied
# by the business/user.
#
# Instead it produces a trade-off table:
#
#   budget
#   -> alerts
#   -> precision
#   -> recall
#   -> C4 captured
#   -> C3+C4 captured
#   -> false alerts
#   -> number-needed-to-inspect
#
# The final operating point should later be chosen from a real
# fleet inspection capacity / business rule.
# ============================================================

from pathlib import Path
import gc

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEEDS = [
    42,
    52,
    62,
    72,
    82,
]

BASE_TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.15
POLICY_DEV_FRACTION = 0.15

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

PRIMARY_PROBABILITY_METHOD = "LogitCalibrated"

pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 340)


# ============================================================
# 2. PATHS
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

print("=" * 115)
print("MODEL 8 — CAPACITY-CONSTRAINED RISK RANKING")
print("=" * 115)

print("\nResolved DATA_DIR:")
print(DATA_DIR)

print("\nChecking files:")

for path in required_files:
    print(f"  {path.name}: {path.exists()}")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Current DATA_DIR = {DATA_DIR}\n"
            "Please keep this script in the same project data folder "
            "used by test_00-test_07, or update DATA_DIR."
        )


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 115)
print("LOADING DATA")
print("=" * 115)

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


# ============================================================
# 5. BASIC CHECKS
# ============================================================

train_vehicle_count = train_op["vehicle_id"].nunique()
validation_vehicle_count = val_op["vehicle_id"].nunique()

assert train_vehicle_count == train_spec["vehicle_id"].nunique()
assert train_vehicle_count == train_tte["vehicle_id"].nunique()

print("\nTraining vehicles:", train_vehicle_count)
print("Validation vehicles:", validation_vehicle_count)


# ============================================================
# 6. RAW OPERATIONAL FEATURES
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
# 7. SORT TIME SERIES
# ============================================================

train_op = (
    train_op
    .sort_values(
        [
            "vehicle_id",
            "time_step",
        ]
    )
    .reset_index(drop=True)
)

val_op = (
    val_op
    .sort_values(
        [
            "vehicle_id",
            "time_step",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 8. TEMPORAL FEATURE ENGINEERING
# ============================================================

def create_temporal_features(
    df,
    feature_cols,
):
    df = (
        df
        .sort_values(
            [
                "vehicle_id",
                "time_step",
            ]
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
    )

    safe_time_gap = (
        df["time_gap"]
        .replace(
            0,
            np.nan,
        )
    )

    previous = (
        grouped[feature_cols]
        .shift(1)
    )

    previous.columns = [
        f"{col}__prev"
        for col in feature_cols
    ]

    current_array = (
        df[feature_cols]
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

    time_gap_array = (
        safe_time_gap
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
            time_gap_array
        )

    rate = pd.DataFrame(
        rate_array,
        index=df.index,
        columns=[
            f"{col}__rate"
            for col in feature_cols
        ],
    )

    rate = rate.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return pd.concat(
        [
            df,
            previous,
            delta,
            rate,
        ],
        axis=1,
    )


print("\n" + "=" * 115)
print("CREATING TEMPORAL FEATURES")
print("=" * 115)

train_temporal = create_temporal_features(
    train_op,
    RAW_FEATURE_COLS,
)

val_temporal = create_temporal_features(
    val_op,
    RAW_FEATURE_COLS,
)

print("Train temporal shape:", train_temporal.shape)
print("Validation temporal shape:", val_temporal.shape)


# ============================================================
# 9. MERGE TRAIN TTE
# ============================================================

train_full = (
    train_temporal
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


# ============================================================
# 10. CLASS FUNCTION
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
# 11. CENSORING-AWARE LABELS
# ============================================================

train_full[
    "class_label"
] = np.nan

repair_mask = (
    train_full[
        "in_study_repair"
    ]
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
    .apply(
        tte_to_class
    )
)

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
    "class_label"
] = 0

candidate_all = (
    train_full
    .dropna(
        subset=[
            "class_label"
        ]
    )
    .copy()
)

candidate_all[
    "class_label"
] = (
    candidate_all[
        "class_label"
    ]
    .astype(int)
)


# ============================================================
# 12. BUILD OFFICIAL VALIDATION SNAPSHOT
# ============================================================

val_last = (
    val_temporal
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

print("\nOfficial validation class distribution:")

print(
    val_last[
        "class_label"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 13. VEHICLE TABLE
# ============================================================

vehicle_table = (
    train_tte[
        [
            "vehicle_id",
            "in_study_repair",
        ]
    ]
    .drop_duplicates(
        "vehicle_id"
    )
    .copy()
)


# ============================================================
# 14. THREE-WAY VEHICLE SPLIT
# ============================================================

def three_way_vehicle_split(
    vehicle_df,
    random_state,
):
    base_ids, temp_ids = (
        train_test_split(
            vehicle_df[
                "vehicle_id"
            ].to_numpy(),

            test_size=(
                CALIBRATION_FRACTION
                +
                POLICY_DEV_FRACTION
            ),

            random_state=random_state,

            stratify=vehicle_df[
                "in_study_repair"
            ].to_numpy(),
        )
    )

    temp_table = (
        vehicle_df[
            vehicle_df[
                "vehicle_id"
            ].isin(
                temp_ids
            )
        ]
        .copy()
    )

    calibration_relative_fraction = (
        CALIBRATION_FRACTION
        /
        (
            CALIBRATION_FRACTION
            +
            POLICY_DEV_FRACTION
        )
    )

    calibration_ids, policy_ids = (
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
    policy_ids = set(policy_ids)

    assert base_ids.isdisjoint(
        calibration_ids
    )

    assert base_ids.isdisjoint(
        policy_ids
    )

    assert calibration_ids.isdisjoint(
        policy_ids
    )

    return (
        base_ids,
        calibration_ids,
        policy_ids,
    )


# ============================================================
# 15. VEHICLE-AWARE TRAINING SAMPLE
# ============================================================

def vehicle_aware_sample(
    candidate_df,
    random_state,
):
    repair_candidates = (
        candidate_df[
            candidate_df[
                "in_study_repair"
            ]
            ==
            1
        ]
        .copy()
    )

    non_repair_candidates = (
        candidate_df[
            candidate_df[
                "in_study_repair"
            ]
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
        .reset_index(
            drop=True
        )
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
        .reset_index(
            drop=True
        )
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
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 16. VEHICLE-BALANCED CALIBRATION WEIGHTS
# ============================================================

def add_vehicle_balanced_weights(
    df,
    weight_column,
):
    df = df.copy()

    rows_per_vehicle = (
        df
        .groupby(
            "vehicle_id"
        )[
            "vehicle_id"
        ]
        .transform(
            "size"
        )
    )

    df[
        weight_column
    ] = (
        1.0
        /
        rows_per_vehicle
    )

    df[
        weight_column
    ] = (
        df[
            weight_column
        ]
        *
        (
            len(df)
            /
            df[
                weight_column
            ].sum()
        )
    )

    return df


# ============================================================
# 17. POLICY DEV: ONE CUTOFF PER VEHICLE
# ============================================================
#
# Ranking is a FLEET-level task.
#
# Therefore a vehicle must appear only once in the candidate
# inspection list.
#
# We sample ONE safely-labelled eligible cutoff per vehicle to
# form a synthetic deployment-like fleet snapshot.
# ============================================================

def sample_one_policy_cutoff_per_vehicle(
    candidate_df,
    random_state,
):
    snapshot = (
        candidate_df
        .groupby(
            "vehicle_id",
            group_keys=False,
        )
        .sample(
            n=1,
            random_state=random_state,
        )
        .reset_index(
            drop=True
        )
    )

    assert (
        snapshot[
            "vehicle_id"
        ].nunique()
        ==
        len(snapshot)
    )

    return snapshot


# ============================================================
# 18. PROBABILITY UTILITIES
# ============================================================

def normalize_probabilities(
    proba
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


# ============================================================
# 19. PRIOR CORRECTION
# ============================================================

def apply_prior_correction(
    raw_proba,
    effective_train_prior,
    calibration_target_prior,
):
    ratio = (
        calibration_target_prior
        /
        np.clip(
            effective_train_prior,
            EPS,
            None,
        )
    )

    corrected = (
        raw_proba
        *
        ratio[
            None,
            :
        ]
    )

    return (
        normalize_probabilities(
            corrected
        ),
        ratio,
    )


# ============================================================
# 20. LOGIT CALIBRATION
# ============================================================

def probability_to_log_features(
    proba
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
            "Calibration split does not support all 5 classes. "
            f"classes_={calibrator.classes_}"
        )

    return calibrator


def aligned_calibrator_proba(
    fitted_calibrator,
    raw_proba,
):
    partial = (
        fitted_calibrator
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

    for (
        column_index,
        class_id
    ) in enumerate(
        fitted_calibrator.classes_
    ):
        aligned[
            :,
            int(class_id)
        ] = partial[
            :,
            column_index
        ]

    return normalize_probabilities(
        aligned
    )


# ============================================================
# 21. BUILD REFERENCE MODEL
# ============================================================

LIGHT_WEIGHT_MAP = {
    0: 1.00,
    1: 1.25,
    2: 1.50,
    3: 1.75,
    4: 2.00,
}


def build_reference_pipeline(
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

    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_CLASSES,

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
                xgb_model,
            ),
        ]
    )


# ============================================================
# 22. CUMULATIVE RISK SCORES
# ============================================================

def cumulative_risk_scores(
    proba
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


# ============================================================
# 23. HORIZON TARGETS
# ============================================================

def horizon_target(
    y_class,
    horizon_name,
):
    y_class = np.asarray(
        y_class,
        dtype=int,
    )

    if horizon_name == "Risk48":
        return (
            y_class >= 1
        ).astype(int)

    if horizon_name == "Risk24":
        return (
            y_class >= 2
        ).astype(int)

    if horizon_name == "Risk12":
        return (
            y_class >= 3
        ).astype(int)

    if horizon_name == "Risk6":
        return (
            y_class >= 4
        ).astype(int)

    raise ValueError(
        f"Unknown horizon: {horizon_name}"
    )


# ============================================================
# 24. RANKING-LEVEL METRICS
# ============================================================

def safe_average_precision(
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


def safe_roc_auc(
    y_true,
    score,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    if len(
        np.unique(
            y_true
        )
    ) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_true,
            score,
        )
    )


def evaluate_horizon_ranking(
    seed,
    split_name,
    probability_method,
    horizon_name,
    actual_class,
    risk_score,
):
    target = horizon_target(
        actual_class,
        horizon_name,
    )

    prevalence = (
        target.mean()
    )

    return {
        "Seed":
            seed,

        "Split":
            split_name,

        "ProbabilityMethod":
            probability_method,

        "Horizon":
            horizon_name,

        "N":
            len(target),

        "PositiveCount":
            int(
                target.sum()
            ),

        "Prevalence":
            float(
                prevalence
            ),

        "AveragePrecision":
            safe_average_precision(
                target,
                risk_score,
            ),

        "ROCAUC":
            safe_roc_auc(
                target,
                risk_score,
            ),
    }


# ============================================================
# 25. CAPACITY-CONSTRAINED TOP-K METRICS
# ============================================================

def evaluate_capacity_policy(
    seed,
    split_name,
    probability_method,
    horizon_name,
    actual_class,
    risk_score,
    budget_fraction,
):
    actual_class = np.asarray(
        actual_class,
        dtype=int,
    )

    risk_score = np.asarray(
        risk_score,
        dtype=np.float64,
    )

    target = horizon_target(
        actual_class,
        horizon_name,
    )

    n = len(
        actual_class
    )

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
        -risk_score,
        kind="mergesort",
    )

    selected = order[
        :k
    ]

    selected_target = target[
        selected
    ]

    tp = int(
        selected_target.sum()
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
        if (
            prevalence is not None
            and prevalence > 0
        )
        else np.nan
    )

    number_needed_to_inspect = (
        k / tp
        if tp > 0
        else np.inf
    )

    # C4 capture regardless of score horizon
    c4_mask = (
        actual_class == 4
    )

    total_c4 = int(
        c4_mask.sum()
    )

    selected_c4 = int(
        c4_mask[
            selected
        ].sum()
    )

    c4_recall_at_k = (
        selected_c4
        /
        total_c4
        if total_c4 > 0
        else np.nan
    )

    # C3 + C4 capture
    critical_mask = (
        actual_class >= 3
    )

    total_critical = int(
        critical_mask.sum()
    )

    selected_critical = int(
        critical_mask[
            selected
        ].sum()
    )

    critical_recall_at_k = (
        selected_critical
        /
        total_critical
        if total_critical > 0
        else np.nan
    )

    critical_precision_at_k = (
        selected_critical
        /
        k
        if k > 0
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
            horizon_name,

        "BudgetFraction":
            budget_fraction,

        "BudgetPercent":
            budget_fraction * 100.0,

        "FleetSize":
            n,

        "AlertCount":
            k,

        "AlertRate":
            k / n,

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
            number_needed_to_inspect,

        "C4Count":
            total_c4,

        "C4Captured":
            selected_c4,

        "C4RecallAtK":
            c4_recall_at_k,

        "C3C4Count":
            total_critical,

        "C3C4Captured":
            selected_critical,

        "C3C4RecallAtK":
            critical_recall_at_k,

        "C3C4PrecisionAtK":
            critical_precision_at_k,

        "RiskThresholdAtK":
            float(
                risk_score[
                    selected[-1]
                ]
            ),
    }


# ============================================================
# 26. NON-FEATURE COLUMNS
# ============================================================

NON_FEATURE_COLS = [
    "vehicle_id",
    "class_label",
    "length_of_study_time_step",
    "in_study_repair",
    "time_to_endpoint",
    "_calibration_weight",
]


# ============================================================
# 27. OFFICIAL VALIDATION FEATURES
# ============================================================

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
    val_last[
        "class_label"
    ]
    .astype(int)
    .to_numpy()
)


# ============================================================
# 28. RESULT STORES
# ============================================================

horizon_metric_rows = []
capacity_metric_rows = []
split_rows = []
risk_score_rows = []


# ============================================================
# 29. MULTI-SEED TEST 08
# ============================================================

print("\n" + "=" * 115)
print("STARTING TEST 08 MULTI-SEED RANKING EXPERIMENT")
print("=" * 115)

for seed in SEEDS:

    print("\n" + "#" * 115)
    print(f"SEED = {seed}")
    print("#" * 115)

    # --------------------------------------------------------
    # A. Split vehicles
    # --------------------------------------------------------

    (
        base_train_ids,
        calibration_ids,
        policy_dev_ids,
    ) = three_way_vehicle_split(
        vehicle_table,
        random_state=seed,
    )

    # --------------------------------------------------------
    # B. Candidate observations by split
    # --------------------------------------------------------

    base_candidates = (
        candidate_all[
            candidate_all[
                "vehicle_id"
            ].isin(
                base_train_ids
            )
        ]
        .copy()
    )

    calibration_candidates = (
        candidate_all[
            candidate_all[
                "vehicle_id"
            ].isin(
                calibration_ids
            )
        ]
        .copy()
    )

    policy_candidates = (
        candidate_all[
            candidate_all[
                "vehicle_id"
            ].isin(
                policy_dev_ids
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # C. Base train sampling
    # --------------------------------------------------------

    base_sampled = (
        vehicle_aware_sample(
            base_candidates,
            random_state=seed,
        )
    )

    # --------------------------------------------------------
    # D. Calibration weighting
    # --------------------------------------------------------

    calibration_candidates = (
        add_vehicle_balanced_weights(
            calibration_candidates,
            "_calibration_weight",
        )
    )

    # --------------------------------------------------------
    # E. Policy dev -> one cutoff per eligible vehicle
    # --------------------------------------------------------

    policy_snapshot = (
        sample_one_policy_cutoff_per_vehicle(
            policy_candidates,
            random_state=seed,
        )
    )

    split_rows.append(
        {
            "Seed":
                seed,

            "BaseTrainVehicleIDs":
                len(
                    base_train_ids
                ),

            "CalibrationVehicleIDs":
                len(
                    calibration_ids
                ),

            "PolicyDevVehicleIDs":
                len(
                    policy_dev_ids
                ),

            "PolicyDevEligibleVehicles":
                policy_snapshot[
                    "vehicle_id"
                ].nunique(),

            "PolicyDevCoverage":
                (
                    policy_snapshot[
                        "vehicle_id"
                    ].nunique()
                    /
                    len(
                        policy_dev_ids
                    )
                ),
        }
    )

    # --------------------------------------------------------
    # F. Check calibration classes
    # --------------------------------------------------------

    calibration_classes = set(
        calibration_candidates[
            "class_label"
        ].unique()
    )

    if calibration_classes != set(
        CLASS_IDS
    ):
        raise ValueError(
            f"Seed {seed}: calibration split missing classes: "
            f"{sorted(calibration_classes)}"
        )

    # --------------------------------------------------------
    # G. Join specifications
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
    # H. Build train/cal/policy matrices
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
        .to_numpy(
            dtype=np.float64
        )
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
    # I. Align schemas
    # --------------------------------------------------------

    assert set(
        X_train.columns
    ) == set(
        X_cal.columns
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

    X_policy = X_policy[
        X_train.columns
    ]

    X_val = X_val_template[
        X_train.columns
    ].copy()

    # --------------------------------------------------------
    # J. Feature types
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
    # K. Train base model
    # --------------------------------------------------------

    train_sample_weights = (
        y_train
        .map(
            LIGHT_WEIGHT_MAP
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    pipeline = (
        build_reference_pipeline(
            numeric_cols,
            categorical_cols,
            random_state=seed,
        )
    )

    print("Base train rows:", len(X_train))
    print("Calibration rows:", len(X_cal))
    print("Policy-dev fleet rows:", len(X_policy))
    print("Official-validation fleet rows:", len(X_val))

    pipeline.fit(
        X_train,
        y_train,
        classifier__sample_weight=train_sample_weights,
    )

    # --------------------------------------------------------
    # L. Raw probabilities
    # --------------------------------------------------------

    cal_raw = (
        pipeline
        .predict_proba(
            X_cal
        )
    )

    policy_raw = (
        pipeline
        .predict_proba(
            X_policy
        )
    )

    val_raw = (
        pipeline
        .predict_proba(
            X_val
        )
    )

    # --------------------------------------------------------
    # M. Prior correction reference
    # --------------------------------------------------------

    effective_train_prior = (
        weighted_class_prior(
            y_train,
            sample_weight=train_sample_weights,
        )
    )

    calibration_target_prior = (
        weighted_class_prior(
            y_cal,
            sample_weight=cal_weights,
        )
    )

    (
        _,
        prior_ratio,
    ) = (
        apply_prior_correction(
            cal_raw,
            effective_train_prior,
            calibration_target_prior,
        )
    )

    policy_prior = (
        normalize_probabilities(
            policy_raw
            *
            prior_ratio[
                None,
                :
            ]
        )
    )

    val_prior = (
        normalize_probabilities(
            val_raw
            *
            prior_ratio[
                None,
                :
            ]
        )
    )

    # --------------------------------------------------------
    # N. PRIMARY: Logit calibrator
    # --------------------------------------------------------

    logit_calibrator = (
        fit_logit_calibrator(
            cal_raw,
            y_cal,
            sample_weight=cal_weights,
            random_state=seed,
        )
    )

    policy_logit = (
        aligned_calibrator_proba(
            logit_calibrator,
            policy_raw,
        )
    )

    val_logit = (
        aligned_calibrator_proba(
            logit_calibrator,
            val_raw,
        )
    )

    probability_sets = {
        "Raw":
            (
                policy_raw,
                val_raw,
            ),

        "PriorCorrected":
            (
                policy_prior,
                val_prior,
            ),

        "LogitCalibrated":
            (
                policy_logit,
                val_logit,
            ),
    }

    # --------------------------------------------------------
    # O. Evaluate every probability method as sensitivity
    # --------------------------------------------------------

    for (
        probability_method,
        (
            policy_proba,
            validation_proba,
        )
    ) in probability_sets.items():

        policy_scores = (
            cumulative_risk_scores(
                policy_proba
            )
        )

        validation_scores = (
            cumulative_risk_scores(
                validation_proba
            )
        )

        # ----------------------------------------------------
        # O1. Selection / policy dev ranking metrics
        # ----------------------------------------------------

        for (
            horizon_name,
            risk_score
        ) in policy_scores.items():

            horizon_metric_rows.append(
                evaluate_horizon_ranking(
                    seed=seed,
                    split_name="PolicyDev",
                    probability_method=probability_method,
                    horizon_name=horizon_name,
                    actual_class=y_policy,
                    risk_score=risk_score,
                )
            )

            for budget in (
                INSPECTION_BUDGETS
            ):
                capacity_metric_rows.append(
                    evaluate_capacity_policy(
                        seed=seed,
                        split_name="PolicyDev",
                        probability_method=probability_method,
                        horizon_name=horizon_name,
                        actual_class=y_policy,
                        risk_score=risk_score,
                        budget_fraction=budget,
                    )
                )

        # ----------------------------------------------------
        # O2. Official validation external diagnostics
        # ----------------------------------------------------

        for (
            horizon_name,
            risk_score
        ) in validation_scores.items():

            horizon_metric_rows.append(
                evaluate_horizon_ranking(
                    seed=seed,
                    split_name="OfficialValidation",
                    probability_method=probability_method,
                    horizon_name=horizon_name,
                    actual_class=y_val,
                    risk_score=risk_score,
                )
            )

            for budget in (
                INSPECTION_BUDGETS
            ):
                capacity_metric_rows.append(
                    evaluate_capacity_policy(
                        seed=seed,
                        split_name="OfficialValidation",
                        probability_method=probability_method,
                        horizon_name=horizon_name,
                        actual_class=y_val,
                        risk_score=risk_score,
                        budget_fraction=budget,
                    )
                )

        # ----------------------------------------------------
        # O3. Save PRIMARY validation risk scores per vehicle
        # ----------------------------------------------------

        if (
            probability_method
            ==
            PRIMARY_PROBABILITY_METHOD
        ):
            for row_index in range(
                len(
                    val_last
                )
            ):
                risk_score_rows.append(
                    {
                        "Seed":
                            seed,

                        "vehicle_id":
                            val_last[
                                "vehicle_id"
                            ].iloc[
                                row_index
                            ],

                        "actual_class":
                            int(
                                y_val[
                                    row_index
                                ]
                            ),

                        "P_C0":
                            float(
                                validation_proba[
                                    row_index,
                                    0
                                ]
                            ),

                        "P_C1":
                            float(
                                validation_proba[
                                    row_index,
                                    1
                                ]
                            ),

                        "P_C2":
                            float(
                                validation_proba[
                                    row_index,
                                    2
                                ]
                            ),

                        "P_C3":
                            float(
                                validation_proba[
                                    row_index,
                                    3
                                ]
                            ),

                        "P_C4":
                            float(
                                validation_proba[
                                    row_index,
                                    4
                                ]
                            ),

                        "Risk48":
                            float(
                                validation_scores[
                                    "Risk48"
                                ][
                                    row_index
                                ]
                            ),

                        "Risk24":
                            float(
                                validation_scores[
                                    "Risk24"
                                ][
                                    row_index
                                ]
                            ),

                        "Risk12":
                            float(
                                validation_scores[
                                    "Risk12"
                                ][
                                    row_index
                                ]
                            ),

                        "Risk6":
                            float(
                                validation_scores[
                                    "Risk6"
                                ][
                                    row_index
                                ]
                            ),
                    }
                )

    # --------------------------------------------------------
    # P. Cleanup seed objects
    # --------------------------------------------------------

    del base_candidates
    del calibration_candidates
    del policy_candidates
    del base_sampled
    del policy_snapshot

    del X_train
    del X_cal
    del X_policy
    del X_val

    del pipeline
    del logit_calibrator

    del cal_raw
    del policy_raw
    del val_raw
    del policy_prior
    del val_prior
    del policy_logit
    del val_logit

    gc.collect()


# ============================================================
# 30. RESULT TABLES
# ============================================================

horizon_metrics_df = pd.DataFrame(
    horizon_metric_rows
)

capacity_metrics_df = pd.DataFrame(
    capacity_metric_rows
)

split_df = pd.DataFrame(
    split_rows
)

validation_risk_scores_df = pd.DataFrame(
    risk_score_rows
)


# ============================================================
# 31. HORIZON SUMMARY
# ============================================================

horizon_summary_df = (
    horizon_metrics_df
    .groupby(
        [
            "Split",
            "ProbabilityMethod",
            "Horizon",
        ]
    )
    .agg(
        N_Mean=(
            "N",
            "mean",
        ),

        PositiveCount_Mean=(
            "PositiveCount",
            "mean",
        ),

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


# ============================================================
# 32. CAPACITY SUMMARY
# ============================================================

capacity_summary_df = (
    capacity_metrics_df
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
        FleetSize_Mean=(
            "FleetSize",
            "mean",
        ),

        AlertCount_Mean=(
            "AlertCount",
            "mean",
        ),

        HorizonPositiveCount_Mean=(
            "HorizonPositiveCount",
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

        C3C4PrecisionAtK_Mean=(
            "C3C4PrecisionAtK",
            "mean",
        ),

        RiskThresholdAtK_Mean=(
            "RiskThresholdAtK",
            "mean",
        ),

        RiskThresholdAtK_Std=(
            "RiskThresholdAtK",
            "std",
        ),
    )
    .reset_index()
)


# ============================================================
# 33. PRIMARY LOGIT TABLES
# ============================================================

primary_horizon_summary_df = (
    horizon_summary_df[
        horizon_summary_df[
            "ProbabilityMethod"
        ]
        ==
        PRIMARY_PROBABILITY_METHOD
    ]
    .copy()
)

primary_capacity_summary_df = (
    capacity_summary_df[
        capacity_summary_df[
            "ProbabilityMethod"
        ]
        ==
        PRIMARY_PROBABILITY_METHOD
    ]
    .copy()
)


# ============================================================
# 34. PRINT IMPORTANT RESULTS
# ============================================================

print("\n" + "=" * 115)
print("PRIMARY LOGIT-CALIBRATED — HORIZON RANKING SUMMARY")
print("=" * 115)

print(
    primary_horizon_summary_df[
        [
            "Split",
            "Horizon",
            "Prevalence_Mean",
            "AveragePrecision_Mean",
            "AveragePrecision_Std",
            "ROCAUC_Mean",
            "ROCAUC_Std",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 115)
print("PRIMARY LOGIT-CALIBRATED — OFFICIAL VALIDATION CAPACITY TABLE")
print("=" * 115)

primary_validation_capacity = (
    primary_capacity_summary_df[
        primary_capacity_summary_df[
            "Split"
        ]
        ==
        "OfficialValidation"
    ]
    .copy()
)

print(
    primary_validation_capacity[
        [
            "Horizon",
            "BudgetPercent",
            "AlertCount_Mean",
            "PrecisionAtK_Mean",
            "RecallAtK_Mean",
            "LiftAtK_Mean",
            "NumberNeededToInspect_Mean",
            "C4RecallAtK_Mean",
            "C3C4RecallAtK_Mean",
            "C3C4PrecisionAtK_Mean",
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 35. SAVE OUTPUTS
# ============================================================

HORIZON_BY_SEED_PATH = (
    OUTPUT_DIR
    /
    "model_08_horizon_metrics_by_seed.csv"
)

HORIZON_SUMMARY_PATH = (
    OUTPUT_DIR
    /
    "model_08_horizon_metrics_summary.csv"
)

CAPACITY_BY_SEED_PATH = (
    OUTPUT_DIR
    /
    "model_08_capacity_metrics_by_seed.csv"
)

CAPACITY_SUMMARY_PATH = (
    OUTPUT_DIR
    /
    "model_08_capacity_metrics_summary.csv"
)

PRIMARY_HORIZON_PATH = (
    OUTPUT_DIR
    /
    "model_08_primary_logit_horizon_summary.csv"
)

PRIMARY_CAPACITY_PATH = (
    OUTPUT_DIR
    /
    "model_08_primary_logit_capacity_summary.csv"
)

SPLIT_PATH = (
    OUTPUT_DIR
    /
    "model_08_split_coverage.csv"
)

VALIDATION_RISK_PATH = (
    OUTPUT_DIR
    /
    "model_08_logit_validation_risk_scores_by_seed.csv"
)

README_PATH = (
    OUTPUT_DIR
    /
    "model_08_interpretation_notes.txt"
)


horizon_metrics_df.to_csv(
    HORIZON_BY_SEED_PATH,
    index=False,
)

horizon_summary_df.to_csv(
    HORIZON_SUMMARY_PATH,
    index=False,
)

capacity_metrics_df.to_csv(
    CAPACITY_BY_SEED_PATH,
    index=False,
)

capacity_summary_df.to_csv(
    CAPACITY_SUMMARY_PATH,
    index=False,
)

primary_horizon_summary_df.to_csv(
    PRIMARY_HORIZON_PATH,
    index=False,
)

primary_capacity_summary_df.to_csv(
    PRIMARY_CAPACITY_PATH,
    index=False,
)

split_df.to_csv(
    SPLIT_PATH,
    index=False,
)

validation_risk_scores_df.to_csv(
    VALIDATION_RISK_PATH,
    index=False,
)

README_PATH.write_text(
    (
        "FleetCare AI — Model 08\n"
        "Capacity-Constrained Risk Ranking\n\n"
        "Primary calibration method:\n"
        "LogitCalibrated (fixed from Test 07)\n\n"
        "Cumulative risks:\n"
        "Risk48 = 1 - P(C0)\n"
        "Risk24 = P(C2)+P(C3)+P(C4)\n"
        "Risk12 = P(C3)+P(C4)\n"
        "Risk6  = P(C4)\n\n"
        "Inspection budgets tested:\n"
        "1%, 2%, 5%, 10%, 20%\n\n"
        "No final inspection budget is automatically selected.\n"
        "The operating point must be chosen using real fleet "
        "inspection capacity / business rules.\n\n"
        "Primary files to inspect:\n"
        "model_08_primary_logit_horizon_summary.csv\n"
        "model_08_primary_logit_capacity_summary.csv\n"
    ),
    encoding="utf-8",
)


# ============================================================
# 36. FINAL GUIDE
# ============================================================

print("\n" + "=" * 115)
print("MODEL 8 COMPLETE")
print("=" * 115)

print(
    "\nTest 08 does NOT pick a final budget automatically."
)

print(
    "Use the PolicyDev tables first to understand the expected "
    "trade-off, then use OfficialValidation only as an external "
    "diagnostic."
)

print(
    "\nQuestions to answer from the result:"
)

print(
    "1) Which cumulative horizon produces the best PR-AUC / "
    "ranking quality?"
)

print(
    "2) At Top 1%, 2%, 5%, 10%, 20%, how many C4 vehicles are "
    "captured?"
)

print(
    "3) How many C3+C4 vehicles are captured?"
)

print(
    "4) How many false alerts are required for that recall?"
)

print(
    "5) What is the Number Needed to Inspect?"
)

print(
    "6) Are these results stable across five random seeds?"
)

print(
    "7) Once a real inspection capacity is known, which budget "
    "provides an acceptable operational trade-off?"
)

print("\nSaved files:")

for path in [
    HORIZON_BY_SEED_PATH,
    HORIZON_SUMMARY_PATH,
    CAPACITY_BY_SEED_PATH,
    CAPACITY_SUMMARY_PATH,
    PRIMARY_HORIZON_PATH,
    PRIMARY_CAPACITY_PATH,
    SPLIT_PATH,
    VALIDATION_RISK_PATH,
    README_PATH,
]:
    print(" ", path)
