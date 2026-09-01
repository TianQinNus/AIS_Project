# ============================================================
# FleetCare AI
# Test 10c — Direct Horizon-Specific Binary Models
# ============================================================
#
# Background
# ----------
# Test 09c rebased the c-track onto teammate's latest base:
#
#   Vehicle-Aware Sampling
#   + Multi-Readout Historical Features
#   + Multiclass XGBoost
#   + No Class Weighting
#
# Test 09c improved external ROC-AUC modestly at all horizons,
# but external AP / Top-K C4 Recall improved only slightly and
# inconsistently. Therefore the remaining bottleneck is still
# external near-term repair ranking.
#
# Test 10c asks:
#
#   Is it better to train the business target DIRECTLY?
#
# Instead of:
#
#   five-class model
#   -> calibrated P(C0..C4)
#   -> sum probabilities into Risk48/24/12/6
#
# we train four direct binary models:
#
#   Risk48: repair within 48 time units?
#   Risk24: repair within 24 time units?
#   Risk12: repair within 12 time units?
#   Risk6 : repair within 6 time units?
#
# Crucially, each binary horizon uses HORIZON-SPECIFIC
# CENSORING:
#
#   Non-repair negative for Risk48: observed repair-free > 48
#   Non-repair negative for Risk24: observed repair-free > 24
#   Non-repair negative for Risk12: observed repair-free > 12
#   Non-repair negative for Risk6 : observed repair-free > 6
#
# This lets shorter-horizon models use more safely-labelled
# negative observations.
#
# Experimental protocol
# ---------------------
# Same vehicle-level split and seeds as 09c:
#
#   70% Base Train
#   15% Calibration
#   15% Selection / Policy Dev
#   seeds = 42, 52, 62, 72, 82
#
# For a fair comparison, both:
#
#   A) Derived multiclass risk (09c-style)
#   B) Direct binary risk
#
# are evaluated on the SAME one-cutoff-per-vehicle PolicyDev
# snapshot and the SAME official validation set.
#
# Official validation is external diagnostic only.
#
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
    log_loss,
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIG
# ============================================================

SEEDS = [42, 52, 62, 72, 82]

CALIBRATION_FRACTION = 0.15
SELECTION_FRACTION = 0.15

HORIZONS = {
    "Risk48": 48,
    "Risk24": 24,
    "Risk12": 12,
    "Risk6": 6,
}

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

pd.set_option("display.max_columns", 350)
pd.set_option("display.width", 360)


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
# 3. LOAD DATA
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
print("TEST 10c — DIRECT HORIZON-SPECIFIC BINARY MODELS")
print("=" * 120)

print("\nResolved DATA_DIR:")
print(DATA_DIR)

for path in required_files:
    print(f"{path.name}: {path.exists()}")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

train_op = pd.read_csv(TRAIN_OP_PATH)
train_spec = pd.read_csv(TRAIN_SPEC_PATH)
train_tte = pd.read_csv(TRAIN_TTE_PATH)

val_op = pd.read_csv(VAL_OP_PATH)
val_spec = pd.read_csv(VAL_SPEC_PATH)
val_labels = pd.read_csv(VAL_LABEL_PATH)

RAW_FEATURE_COLS = [
    col
    for col in train_op.columns
    if col not in [
        "vehicle_id",
        "time_step",
    ]
]


# ============================================================
# 4. MULTI-READOUT HISTORY FEATURES
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

    current = (
        df[feature_cols]
        .to_numpy(dtype=np.float32)
    )

    previous_array = (
        previous
        .to_numpy(dtype=np.float32)
    )

    delta_array = (
        current
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
        current
        -
        lag3.to_numpy(dtype=np.float32),
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
        current
        -
        lag5.to_numpy(dtype=np.float32),
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


print("\nCreating multi-readout history features...")

train_history = create_history_features(
    train_op,
    RAW_FEATURE_COLS,
)

val_history = create_history_features(
    val_op,
    RAW_FEATURE_COLS,
)

del train_op
del val_op
gc.collect()


# ============================================================
# 5. MERGE TRAIN ENDPOINT INFORMATION
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

del train_history
gc.collect()


# ============================================================
# 6. FIVE-CLASS LABELS
# ============================================================

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

safe_c0_mask = (
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
    safe_c0_mask,
    "class_label"
] = 0

five_class_candidates = (
    train_full
    .dropna(
        subset=["class_label"]
    )
    .copy()
)

five_class_candidates["class_label"] = (
    five_class_candidates[
        "class_label"
    ]
    .astype(int)
)


# ============================================================
# 7. HORIZON-SPECIFIC BINARY LABELS
# ============================================================

def build_horizon_candidates(
    df,
    horizon,
):
    out = df.copy()

    out["binary_label"] = np.nan

    repair_valid = (
        (
            out["in_study_repair"]
            ==
            1
        )
        &
        (
            out["time_to_endpoint"]
            >=
            0
        )
    )

    out.loc[
        repair_valid,
        "binary_label"
    ] = (
        out.loc[
            repair_valid,
            "time_to_endpoint"
        ]
        <=
        horizon
    ).astype(int)

    safe_nonrepair_negative = (
        (
            out["in_study_repair"]
            ==
            0
        )
        &
        (
            out["time_to_endpoint"]
            >
            horizon
        )
    )

    out.loc[
        safe_nonrepair_negative,
        "binary_label"
    ] = 0

    out = (
        out
        .dropna(
            subset=["binary_label"]
        )
        .copy()
    )

    out["binary_label"] = (
        out["binary_label"]
        .astype(int)
    )

    return out


horizon_candidates = {
    name:
        build_horizon_candidates(
            train_full,
            horizon,
        )

    for name, horizon in (
        HORIZONS.items()
    )
}


# ============================================================
# 8. OFFICIAL VALIDATION SNAPSHOT
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

y_val_class = (
    val_last["class_label"]
    .astype(int)
    .to_numpy()
)


# ============================================================
# 9. VEHICLE SPLIT
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
            vehicle_df[
                "vehicle_id"
            ].to_numpy(),

            test_size=(
                CALIBRATION_FRACTION
                +
                SELECTION_FRACTION
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

    calibration_ids, selection_ids = (
        train_test_split(
            temp_table[
                "vehicle_id"
            ].to_numpy(),

            train_size=0.5,

            random_state=random_state,

            stratify=temp_table[
                "in_study_repair"
            ].to_numpy(),
        )
    )

    return (
        set(base_ids),
        set(calibration_ids),
        set(selection_ids),
    )


# ============================================================
# 10. SAMPLING
# ============================================================

def sample_five_class_vehicle_aware(
    df,
    random_state,
):
    repair = (
        df[
            df["in_study_repair"]
            ==
            1
        ]
        .copy()
    )

    nonrepair = (
        df[
            df["in_study_repair"]
            ==
            0
        ]
        .copy()
    )

    repair_sampled = (
        repair
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
    )

    nonrepair_sampled = (
        nonrepair
        .groupby(
            "vehicle_id",
            group_keys=False,
        )
        .sample(
            n=1,
            random_state=random_state,
        )
    )

    return (
        pd.concat(
            [
                repair_sampled,
                nonrepair_sampled,
            ],
            ignore_index=True,
        )
        .sample(
            frac=1,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )


def sample_binary_vehicle_aware(
    df,
    random_state,
):
    repair = (
        df[
            df["in_study_repair"]
            ==
            1
        ]
        .copy()
    )

    nonrepair = (
        df[
            df["in_study_repair"]
            ==
            0
        ]
        .copy()
    )

    # Repair vehicle can contribute at most:
    # one negative + one positive observation.
    repair_sampled = (
        repair
        .groupby(
            [
                "vehicle_id",
                "binary_label",
            ],
            group_keys=False,
        )
        .sample(
            n=1,
            random_state=random_state,
        )
    )

    # Non-repair contributes one safely-known negative.
    nonrepair_sampled = (
        nonrepair
        .groupby(
            "vehicle_id",
            group_keys=False,
        )
        .sample(
            n=1,
            random_state=random_state,
        )
    )

    return (
        pd.concat(
            [
                repair_sampled,
                nonrepair_sampled,
            ],
            ignore_index=True,
        )
        .sample(
            frac=1,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )


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

    df[weight_col] *= (
        len(df)
        /
        df[weight_col].sum()
    )

    return df


def one_cutoff_per_vehicle(
    df,
    random_state,
):
    out = (
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
        out["vehicle_id"].nunique()
        ==
        len(out)
    )

    return out


# ============================================================
# 11. MODEL BUILDERS
# ============================================================

NON_FEATURE_COLS = [
    "vehicle_id",
    "class_label",
    "binary_label",
    "length_of_study_time_step",
    "in_study_repair",
    "time_to_endpoint",
    "_weight",
]


def build_preprocessor(
    X_train,
):
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

    return ColumnTransformer(
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


def build_multiclass_pipeline(
    X_train,
    random_state,
):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    X_train
                ),
            ),
            (
                "classifier",
                XGBClassifier(
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
                ),
            ),
        ]
    )


def build_binary_pipeline(
    X_train,
    random_state,
):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    X_train
                ),
            ),
            (
                "classifier",
                XGBClassifier(
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

                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ============================================================
# 12. MULTICLASS LOGIT CALIBRATION
# ============================================================

def normalize_multiclass(
    proba,
):
    proba = np.clip(
        np.asarray(
            proba,
            dtype=np.float64,
        ),
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


def fit_multiclass_logit_calibrator(
    raw_proba,
    y,
    sample_weight,
    random_state,
):
    X_log = np.log(
        np.clip(
            normalize_multiclass(
                raw_proba
            ),
            EPS,
            1.0,
        )
    )

    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        random_state=random_state,
    )

    model.fit(
        X_log,
        y,
        sample_weight=sample_weight,
    )

    return model


def apply_multiclass_logit_calibrator(
    calibrator,
    raw_proba,
):
    X_log = np.log(
        np.clip(
            normalize_multiclass(
                raw_proba
            ),
            EPS,
            1.0,
        )
    )

    partial = (
        calibrator
        .predict_proba(
            X_log
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

    return normalize_multiclass(
        aligned
    )


# ============================================================
# 13. BINARY LOGIT CALIBRATION
# ============================================================

def binary_logit_feature(
    p,
):
    p = np.clip(
        np.asarray(
            p,
            dtype=np.float64,
        ),
        EPS,
        1.0 - EPS,
    )

    return (
        np.log(
            p
            /
            (1.0 - p)
        )
        .reshape(-1, 1)
    )


def fit_binary_logit_calibrator(
    raw_p,
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
        binary_logit_feature(
            raw_p
        ),
        y,
        sample_weight=sample_weight,
    )

    return calibrator


def apply_binary_logit_calibrator(
    calibrator,
    raw_p,
):
    return (
        calibrator
        .predict_proba(
            binary_logit_feature(
                raw_p
            )
        )[:, 1]
    )


# ============================================================
# 14. TARGET / RISK HELPERS
# ============================================================

def horizon_target_from_class(
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
        horizon_name
    )


def derived_risk(
    calibrated_proba,
    horizon_name,
):
    p = normalize_multiclass(
        calibrated_proba
    )

    if horizon_name == "Risk48":
        return (
            1.0
            -
            p[:, 0]
        )

    if horizon_name == "Risk24":
        return (
            p[:, 2]
            +
            p[:, 3]
            +
            p[:, 4]
        )

    if horizon_name == "Risk12":
        return (
            p[:, 3]
            +
            p[:, 4]
        )

    if horizon_name == "Risk6":
        return p[:, 4]

    raise ValueError(
        horizon_name
    )


# ============================================================
# 15. METRICS
# ============================================================

def safe_ap(
    y_true,
    score,
):
    if np.sum(y_true) == 0:
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


def evaluate_ranking(
    seed,
    split_name,
    horizon,
    model_type,
    score,
    y_true,
):
    return {
        "Seed":
            seed,

        "Split":
            split_name,

        "Horizon":
            horizon,

        "ModelType":
            model_type,

        "N":
            len(y_true),

        "PositiveCount":
            int(
                np.sum(y_true)
            ),

        "Prevalence":
            float(
                np.mean(y_true)
            ),

        "AveragePrecision":
            safe_ap(
                y_true,
                score,
            ),

        "ROCAUC":
            safe_auc(
                y_true,
                score,
            ),
    }


def evaluate_topk(
    seed,
    split_name,
    horizon,
    model_type,
    score,
    y_true,
    actual_class,
    budget,
):
    n = len(y_true)

    k = max(
        1,
        int(
            np.ceil(
                n
                *
                budget
            )
        )
    )

    order = np.argsort(
        -np.asarray(score),
        kind="mergesort",
    )

    selected = order[:k]

    tp = int(
        np.sum(
            y_true[selected]
        )
    )

    positive_count = int(
        np.sum(
            y_true
        )
    )

    precision = (
        tp / k
        if k > 0
        else np.nan
    )

    recall = (
        tp / positive_count
        if positive_count > 0
        else np.nan
    )

    prevalence = (
        positive_count / n
        if n > 0
        else np.nan
    )

    lift = (
        precision / prevalence
        if prevalence > 0
        else np.nan
    )

    c4_mask = (
        np.asarray(actual_class)
        ==
        4
    )

    c4_count = int(
        c4_mask.sum()
    )

    c4_captured = int(
        c4_mask[selected].sum()
    )

    c4_recall = (
        c4_captured / c4_count
        if c4_count > 0
        else np.nan
    )

    return {
        "Seed":
            seed,

        "Split":
            split_name,

        "Horizon":
            horizon,

        "ModelType":
            model_type,

        "BudgetFraction":
            budget,

        "BudgetPercent":
            budget * 100.0,

        "FleetSize":
            n,

        "AlertCount":
            k,

        "PrecisionAtK":
            precision,

        "RecallAtK":
            recall,

        "LiftAtK":
            lift,

        "FalseAlertsAtK":
            int(
                k - tp
            ),

        "C4Count":
            c4_count,

        "C4Captured":
            c4_captured,

        "C4RecallAtK":
            c4_recall,

        "RiskThresholdAtK":
            float(
                score[
                    selected[-1]
                ]
            ),
    }


# ============================================================
# 16. RESULT STORES
# ============================================================

data_expansion_rows = []

binary_calibration_rows = []

ranking_rows = []
topk_rows = []


# ============================================================
# 17. MULTI-SEED EXPERIMENT
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

    # ========================================================
    # A. MULTICLASS DERIVED-RISK BRANCH
    # ========================================================

    base_five = (
        five_class_candidates[
            five_class_candidates[
                "vehicle_id"
            ].isin(base_ids)
        ]
        .copy()
    )

    cal_five = (
        five_class_candidates[
            five_class_candidates[
                "vehicle_id"
            ].isin(calibration_ids)
        ]
        .copy()
    )

    sel_five = (
        five_class_candidates[
            five_class_candidates[
                "vehicle_id"
            ].isin(selection_ids)
        ]
        .copy()
    )

    base_five = (
        sample_five_class_vehicle_aware(
            base_five,
            seed,
        )
    )

    cal_five = (
        add_vehicle_balanced_weights(
            cal_five,
            "_weight",
        )
    )

    common_policy = (
        one_cutoff_per_vehicle(
            sel_five,
            seed,
        )
    )

    base_five = (
        base_five
        .merge(
            train_spec,
            on="vehicle_id",
            how="left",
            validate="many_to_one",
        )
    )

    cal_five = (
        cal_five
        .merge(
            train_spec,
            on="vehicle_id",
            how="left",
            validate="many_to_one",
        )
    )

    common_policy = (
        common_policy
        .merge(
            train_spec,
            on="vehicle_id",
            how="left",
            validate="one_to_one",
        )
    )

    X_train_multi = (
        base_five
        .drop(
            columns=NON_FEATURE_COLS,
            errors="ignore",
        )
    )

    y_train_multi = (
        base_five[
            "class_label"
        ]
        .astype(int)
        .to_numpy()
    )

    X_cal_multi = (
        cal_five
        .drop(
            columns=NON_FEATURE_COLS,
            errors="ignore",
        )
    )

    y_cal_multi = (
        cal_five[
            "class_label"
        ]
        .astype(int)
        .to_numpy()
    )

    cal_multi_weights = (
        cal_five[
            "_weight"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    X_policy_common = (
        common_policy
        .drop(
            columns=NON_FEATURE_COLS,
            errors="ignore",
        )
    )

    y_policy_class = (
        common_policy[
            "class_label"
        ]
        .astype(int)
        .to_numpy()
    )

    X_cal_multi = (
        X_cal_multi[
            X_train_multi.columns
        ]
    )

    X_policy_common = (
        X_policy_common[
            X_train_multi.columns
        ]
    )

    X_val_multi = (
        X_val_template[
            X_train_multi.columns
        ]
        .copy()
    )

    multi_pipeline = (
        build_multiclass_pipeline(
            X_train_multi,
            seed,
        )
    )

    multi_pipeline.fit(
        X_train_multi,
        y_train_multi,
    )

    cal_multi_raw = (
        multi_pipeline
        .predict_proba(
            X_cal_multi
        )
    )

    policy_multi_raw = (
        multi_pipeline
        .predict_proba(
            X_policy_common
        )
    )

    val_multi_raw = (
        multi_pipeline
        .predict_proba(
            X_val_multi
        )
    )

    multi_calibrator = (
        fit_multiclass_logit_calibrator(
            cal_multi_raw,
            y_cal_multi,
            cal_multi_weights,
            seed,
        )
    )

    policy_multi_cal = (
        apply_multiclass_logit_calibrator(
            multi_calibrator,
            policy_multi_raw,
        )
    )

    val_multi_cal = (
        apply_multiclass_logit_calibrator(
            multi_calibrator,
            val_multi_raw,
        )
    )

    # ========================================================
    # B. DIRECT BINARY BRANCH — EACH HORIZON
    # ========================================================

    for horizon_name, horizon_value in (
        HORIZONS.items()
    ):

        print(
            f"  Direct model: {horizon_name}"
        )

        horizon_df = (
            horizon_candidates[
                horizon_name
            ]
        )

        base_binary = (
            horizon_df[
                horizon_df[
                    "vehicle_id"
                ].isin(base_ids)
            ]
            .copy()
        )

        cal_binary = (
            horizon_df[
                horizon_df[
                    "vehicle_id"
                ].isin(calibration_ids)
            ]
            .copy()
        )

        # Data expansion diagnostic
        old_safe_rows = (
            five_class_candidates[
                five_class_candidates[
                    "vehicle_id"
                ].isin(base_ids)
            ]
        )

        data_expansion_rows.append(
            {
                "Seed":
                    seed,

                "Horizon":
                    horizon_name,

                "FiveClassSafeEligibleRows":
                    len(
                        old_safe_rows
                    ),

                "HorizonSpecificEligibleRows":
                    len(
                        base_binary
                    ),

                "EligibleRowExpansion":
                    (
                        len(base_binary)
                        -
                        len(old_safe_rows)
                    ),

                "EligibleRowExpansionRate":
                    (
                        len(base_binary)
                        /
                        len(old_safe_rows)
                        -
                        1.0
                    ),

                "HorizonSpecificEligibleVehicles":
                    base_binary[
                        "vehicle_id"
                    ].nunique(),
            }
        )

        base_binary = (
            sample_binary_vehicle_aware(
                base_binary,
                seed,
            )
        )

        cal_binary = (
            add_vehicle_balanced_weights(
                cal_binary,
                "_weight",
            )
        )

        base_binary = (
            base_binary
            .merge(
                train_spec,
                on="vehicle_id",
                how="left",
                validate="many_to_one",
            )
        )

        cal_binary = (
            cal_binary
            .merge(
                train_spec,
                on="vehicle_id",
                how="left",
                validate="many_to_one",
            )
        )

        X_train_bin = (
            base_binary
            .drop(
                columns=NON_FEATURE_COLS,
                errors="ignore",
            )
        )

        y_train_bin = (
            base_binary[
                "binary_label"
            ]
            .astype(int)
            .to_numpy()
        )

        X_cal_bin = (
            cal_binary
            .drop(
                columns=NON_FEATURE_COLS,
                errors="ignore",
            )
        )

        y_cal_bin = (
            cal_binary[
                "binary_label"
            ]
            .astype(int)
            .to_numpy()
        )

        cal_bin_weights = (
            cal_binary[
                "_weight"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        # Fair evaluation on SAME common policy snapshot.
        X_policy_bin = (
            common_policy
            .drop(
                columns=NON_FEATURE_COLS,
                errors="ignore",
            )
        )

        y_policy_binary = (
            horizon_target_from_class(
                y_policy_class,
                horizon_name,
            )
        )

        y_val_binary = (
            horizon_target_from_class(
                y_val_class,
                horizon_name,
            )
        )

        # Align to binary training schema.
        X_cal_bin = (
            X_cal_bin[
                X_train_bin.columns
            ]
        )

        X_policy_bin = (
            X_policy_bin[
                X_train_bin.columns
            ]
        )

        X_val_bin = (
            X_val_template[
                X_train_bin.columns
            ]
            .copy()
        )

        binary_pipeline = (
            build_binary_pipeline(
                X_train_bin,
                seed,
            )
        )

        binary_pipeline.fit(
            X_train_bin,
            y_train_bin,
        )

        cal_binary_raw = (
            binary_pipeline
            .predict_proba(
                X_cal_bin
            )[:, 1]
        )

        policy_binary_raw = (
            binary_pipeline
            .predict_proba(
                X_policy_bin
            )[:, 1]
        )

        val_binary_raw = (
            binary_pipeline
            .predict_proba(
                X_val_bin
            )[:, 1]
        )

        binary_calibrator = (
            fit_binary_logit_calibrator(
                cal_binary_raw,
                y_cal_bin,
                cal_bin_weights,
                seed,
            )
        )

        policy_binary_cal = (
            apply_binary_logit_calibrator(
                binary_calibrator,
                policy_binary_raw,
            )
        )

        val_binary_cal = (
            apply_binary_logit_calibrator(
                binary_calibrator,
                val_binary_raw,
            )
        )

        # Selection/calibration diagnostic is evaluated on
        # common policy snapshot, not official validation.
        for method_name, score in {
            "DirectRaw":
                policy_binary_raw,

            "DirectLogitCalibrated":
                policy_binary_cal,
        }.items():

            p = np.clip(
                score,
                EPS,
                1.0 - EPS,
            )

            binary_calibration_rows.append(
                {
                    "Seed":
                        seed,

                    "Horizon":
                        horizon_name,

                    "Method":
                        method_name,

                    "PolicyDevLogLoss":
                        log_loss(
                            y_policy_binary,
                            np.column_stack(
                                [
                                    1.0 - p,
                                    p,
                                ]
                            ),
                            labels=[0, 1],
                        ),
                }
            )

        # Derived multiclass scores.
        policy_derived = (
            derived_risk(
                policy_multi_cal,
                horizon_name,
            )
        )

        val_derived = (
            derived_risk(
                val_multi_cal,
                horizon_name,
            )
        )

        score_sets = {
            "DerivedMulticlass":
                (
                    policy_derived,
                    val_derived,
                ),

            "DirectBinaryRaw":
                (
                    policy_binary_raw,
                    val_binary_raw,
                ),

            "DirectBinaryCalibrated":
                (
                    policy_binary_cal,
                    val_binary_cal,
                ),
        }

        for model_type, (
            policy_score,
            validation_score,
        ) in score_sets.items():

            ranking_rows.append(
                evaluate_ranking(
                    seed=seed,
                    split_name="PolicyDev",
                    horizon=horizon_name,
                    model_type=model_type,
                    score=policy_score,
                    y_true=y_policy_binary,
                )
            )

            ranking_rows.append(
                evaluate_ranking(
                    seed=seed,
                    split_name="OfficialValidation",
                    horizon=horizon_name,
                    model_type=model_type,
                    score=validation_score,
                    y_true=y_val_binary,
                )
            )

            for budget in (
                INSPECTION_BUDGETS
            ):
                topk_rows.append(
                    evaluate_topk(
                        seed=seed,
                        split_name="PolicyDev",
                        horizon=horizon_name,
                        model_type=model_type,
                        score=policy_score,
                        y_true=y_policy_binary,
                        actual_class=y_policy_class,
                        budget=budget,
                    )
                )

                topk_rows.append(
                    evaluate_topk(
                        seed=seed,
                        split_name="OfficialValidation",
                        horizon=horizon_name,
                        model_type=model_type,
                        score=validation_score,
                        y_true=y_val_binary,
                        actual_class=y_val_class,
                        budget=budget,
                    )
                )

        del base_binary
        del cal_binary
        del X_train_bin
        del X_cal_bin
        del X_policy_bin
        del X_val_bin
        del binary_pipeline
        del binary_calibrator
        gc.collect()

    del base_five
    del cal_five
    del common_policy

    del X_train_multi
    del X_cal_multi
    del X_policy_common
    del X_val_multi

    del multi_pipeline
    del multi_calibrator

    gc.collect()


# ============================================================
# 18. RESULT TABLES
# ============================================================

data_expansion_df = pd.DataFrame(
    data_expansion_rows
)

binary_calibration_df = pd.DataFrame(
    binary_calibration_rows
)

ranking_df = pd.DataFrame(
    ranking_rows
)

topk_df = pd.DataFrame(
    topk_rows
)


# ============================================================
# 19. SUMMARIES
# ============================================================

data_expansion_summary_df = (
    data_expansion_df
    .groupby("Horizon")
    .agg(
        FiveClassSafeEligibleRows_Mean=(
            "FiveClassSafeEligibleRows",
            "mean",
        ),

        HorizonSpecificEligibleRows_Mean=(
            "HorizonSpecificEligibleRows",
            "mean",
        ),

        EligibleRowExpansion_Mean=(
            "EligibleRowExpansion",
            "mean",
        ),

        EligibleRowExpansionRate_Mean=(
            "EligibleRowExpansionRate",
            "mean",
        ),

        HorizonSpecificEligibleVehicles_Mean=(
            "HorizonSpecificEligibleVehicles",
            "mean",
        ),
    )
    .reset_index()
)


binary_calibration_summary_df = (
    binary_calibration_df
    .groupby(
        [
            "Horizon",
            "Method",
        ]
    )
    .agg(
        PolicyDevLogLoss_Mean=(
            "PolicyDevLogLoss",
            "mean",
        ),

        PolicyDevLogLoss_Std=(
            "PolicyDevLogLoss",
            "std",
        ),
    )
    .reset_index()
)


ranking_summary_df = (
    ranking_df
    .groupby(
        [
            "Split",
            "Horizon",
            "ModelType",
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


topk_summary_df = (
    topk_df
    .groupby(
        [
            "Split",
            "Horizon",
            "ModelType",
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

        C4RecallAtK_Mean=(
            "C4RecallAtK",
            "mean",
        ),

        C4RecallAtK_Std=(
            "C4RecallAtK",
            "std",
        ),
    )
    .reset_index()
)


# ============================================================
# 20. DIRECT VS DERIVED COMPARISON
# ============================================================

comparison_rows = []

for split_name in [
    "PolicyDev",
    "OfficialValidation",
]:

    split_ranking = (
        ranking_summary_df[
            ranking_summary_df[
                "Split"
            ]
            ==
            split_name
        ]
    )

    split_top5 = (
        topk_summary_df[
            (
                topk_summary_df[
                    "Split"
                ]
                ==
                split_name
            )
            &
            (
                np.isclose(
                    topk_summary_df[
                        "BudgetFraction"
                    ],
                    0.05,
                )
            )
        ]
    )

    for horizon_name in (
        HORIZONS.keys()
    ):

        derived = (
            split_ranking[
                (
                    split_ranking[
                        "Horizon"
                    ]
                    ==
                    horizon_name
                )
                &
                (
                    split_ranking[
                        "ModelType"
                    ]
                    ==
                    "DerivedMulticlass"
                )
            ]
            .iloc[0]
        )

        direct = (
            split_ranking[
                (
                    split_ranking[
                        "Horizon"
                    ]
                    ==
                    horizon_name
                )
                &
                (
                    split_ranking[
                        "ModelType"
                    ]
                    ==
                    "DirectBinaryCalibrated"
                )
            ]
            .iloc[0]
        )

        derived_top5 = (
            split_top5[
                (
                    split_top5[
                        "Horizon"
                    ]
                    ==
                    horizon_name
                )
                &
                (
                    split_top5[
                        "ModelType"
                    ]
                    ==
                    "DerivedMulticlass"
                )
            ]
            .iloc[0]
        )

        direct_top5 = (
            split_top5[
                (
                    split_top5[
                        "Horizon"
                    ]
                    ==
                    horizon_name
                )
                &
                (
                    split_top5[
                        "ModelType"
                    ]
                    ==
                    "DirectBinaryCalibrated"
                )
            ]
            .iloc[0]
        )

        comparison_rows.append(
            {
                "Split":
                    split_name,

                "Horizon":
                    horizon_name,

                "Derived_AP":
                    derived[
                        "AveragePrecision_Mean"
                    ],

                "Direct_AP":
                    direct[
                        "AveragePrecision_Mean"
                    ],

                "Delta_AP_DirectMinusDerived":
                    (
                        direct[
                            "AveragePrecision_Mean"
                        ]
                        -
                        derived[
                            "AveragePrecision_Mean"
                        ]
                    ),

                "Derived_ROCAUC":
                    derived[
                        "ROCAUC_Mean"
                    ],

                "Direct_ROCAUC":
                    direct[
                        "ROCAUC_Mean"
                    ],

                "Delta_ROCAUC_DirectMinusDerived":
                    (
                        direct[
                            "ROCAUC_Mean"
                        ]
                        -
                        derived[
                            "ROCAUC_Mean"
                        ]
                    ),

                "Derived_Top5Recall":
                    derived_top5[
                        "RecallAtK_Mean"
                    ],

                "Direct_Top5Recall":
                    direct_top5[
                        "RecallAtK_Mean"
                    ],

                "Delta_Top5Recall_DirectMinusDerived":
                    (
                        direct_top5[
                            "RecallAtK_Mean"
                        ]
                        -
                        derived_top5[
                            "RecallAtK_Mean"
                        ]
                    ),

                "Derived_Top5C4Recall":
                    derived_top5[
                        "C4RecallAtK_Mean"
                    ],

                "Direct_Top5C4Recall":
                    direct_top5[
                        "C4RecallAtK_Mean"
                    ],

                "Delta_Top5C4Recall_DirectMinusDerived":
                    (
                        direct_top5[
                            "C4RecallAtK_Mean"
                        ]
                        -
                        derived_top5[
                            "C4RecallAtK_Mean"
                        ]
                    ),
            }
        )


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# 21. PRINT
# ============================================================

print("\n" + "=" * 120)
print("HORIZON-SPECIFIC DATA EXPANSION")
print("=" * 120)

print(
    data_expansion_summary_df
    .round(6)
    .to_string(index=False)
)


print("\n" + "=" * 120)
print("DIRECT BINARY CALIBRATION")
print("=" * 120)

print(
    binary_calibration_summary_df
    .round(6)
    .to_string(index=False)
)


print("\n" + "=" * 120)
print("DIRECT VS DERIVED — POLICY DEV")
print("=" * 120)

print(
    comparison_df[
        comparison_df[
            "Split"
        ]
        ==
        "PolicyDev"
    ]
    .round(6)
    .to_string(index=False)
)


print("\n" + "=" * 120)
print("DIRECT VS DERIVED — OFFICIAL VALIDATION")
print("=" * 120)

print(
    comparison_df[
        comparison_df[
            "Split"
        ]
        ==
        "OfficialValidation"
    ]
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 22. SAVE
# ============================================================

paths = {
    "data_expansion":
        OUTPUT_DIR
        /
        "model_10c_horizon_data_expansion.csv",

    "binary_calibration":
        OUTPUT_DIR
        /
        "model_10c_binary_calibration_summary.csv",

    "ranking_by_seed":
        OUTPUT_DIR
        /
        "model_10c_ranking_metrics_by_seed.csv",

    "ranking_summary":
        OUTPUT_DIR
        /
        "model_10c_ranking_metrics_summary.csv",

    "topk_by_seed":
        OUTPUT_DIR
        /
        "model_10c_topk_metrics_by_seed.csv",

    "topk_summary":
        OUTPUT_DIR
        /
        "model_10c_topk_metrics_summary.csv",

    "direct_vs_derived":
        OUTPUT_DIR
        /
        "model_10c_direct_vs_derived_summary.csv",

    "notes":
        OUTPUT_DIR
        /
        "model_10c_interpretation_notes.txt",
}


data_expansion_summary_df.to_csv(
    paths["data_expansion"],
    index=False,
)

binary_calibration_summary_df.to_csv(
    paths["binary_calibration"],
    index=False,
)

ranking_df.to_csv(
    paths["ranking_by_seed"],
    index=False,
)

ranking_summary_df.to_csv(
    paths["ranking_summary"],
    index=False,
)

topk_df.to_csv(
    paths["topk_by_seed"],
    index=False,
)

topk_summary_df.to_csv(
    paths["topk_summary"],
    index=False,
)

comparison_df.to_csv(
    paths["direct_vs_derived"],
    index=False,
)

paths["notes"].write_text(
    (
        "FleetCare AI — Test 10c\n"
        "Direct Horizon-Specific Binary Models\n\n"
        "Primary comparison:\n"
        "DerivedMulticlass vs DirectBinaryCalibrated\n\n"
        "Selection logic:\n"
        "Do not select from official validation alone.\n"
        "A strong result means direct binary improves PolicyDev "
        "AND shows the same directional improvement externally.\n\n"
        "Key questions:\n"
        "1. Does horizon-specific censoring increase safe data?\n"
        "2. Does Direct Binary improve AP / ROC-AUC?\n"
        "3. Does Top-5% Recall improve?\n"
        "4. Is the improvement consistent across horizons/seeds?\n"
        "5. If only one or two horizons improve, consider a "
        "hybrid architecture instead of replacing the entire "
        "multiclass model.\n"
    ),
    encoding="utf-8",
)

print("\nSaved outputs:")
for path in paths.values():
    print(" ", path)

print("\nTEST 10c COMPLETE")
