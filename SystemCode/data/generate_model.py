# ============================================================
# FleetCare AI - Train and Save Final Model
#
# Final model:
#   Model: XGBoost 5-class classifier
#   Feature Set: F5_rate_plus_short_change
#   Parameters: P5_less_regularized
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np

# ------------------------------------------------------------
# IMPORTANT:
# Change this import to your actual Test 12c filename
# Example:
# import test_12c_no_cost as test12
# ------------------------------------------------------------
import test_12c_feature_parameter_search as test12

# Test 12c itself uses functions from Test 11c
test11 = test12.test11


# ============================================================
# 1. FINAL CONFIGURATION
# ============================================================

FINAL_FEATURE_SET = "F5_rate_plus_short_change"
FINAL_PARAMETER_SET = "P5_less_regularized"
FINAL_SEED = 42

OUTPUT_DIR = Path(test11.OUTPUT_DIR) / "final_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

print("=" * 60)
print("FleetCare AI - Final Model Training")
print("=" * 60)

print("\n[1/8] Loading and preparing data...")

data, candidates, official = test12.prepare_all_data()

# Official validation final readout times are used to construct
# PolicyDev-like landmark snapshots
target_times = official["time_step"].to_numpy()


# ============================================================
# 3. CREATE VEHICLE-LEVEL TRAIN / CAL / POLICY SPLITS
# ============================================================

print("\n[2/8] Creating vehicle-level splits...")

seed_data = test12.build_seed_data(
    candidates=candidates,
    official=official,
    target_times=target_times,
    train_tte=data["train_tte"],
    train_spec=data["train_spec"],
    seed=FINAL_SEED,
)

train_sample, cal_snapshot, policy_snapshot, official_snapshot = seed_data

print("Training samples:", len(train_sample))
print("Calibration vehicles:", len(cal_snapshot))
print("PolicyDev vehicles:", len(policy_snapshot))
print("Official Validation vehicles:", len(official_snapshot))


# ============================================================
# 4. GENERATE F5 FEATURE SET
# ============================================================

print("\n[3/8] Generating F5 features...")

frames = [
    test11.feature_frame(frame)
    for frame in seed_data
]

# Select exactly the F5 columns
feature_columns = test12.feature_columns(
    frames[0],
    FINAL_FEATURE_SET
)

print("Feature set:", FINAL_FEATURE_SET)
print("Number of raw selected features:", len(feature_columns))

# Force every split to have exactly the same columns/order
frames = [
    frame.reindex(columns=feature_columns)
    for frame in frames
]

train_frame = frames[0]
cal_frame = frames[1]
policy_frame = frames[2]
official_frame = frames[3]


# ============================================================
# 5. FIT PREPROCESSOR
# ============================================================

print("\n[4/8] Fitting preprocessor...")

# IMPORTANT:
# Fit ONLY using training data
preprocessor = test11.make_preprocessor(train_frame)

X_train = preprocessor.fit_transform(train_frame)

X_cal = preprocessor.transform(cal_frame)
X_policy = preprocessor.transform(policy_frame)
X_official = preprocessor.transform(official_frame)

y_train = train_sample["class_label"].astype(int).to_numpy()
y_cal = cal_snapshot["class_label"].astype(int).to_numpy()
y_policy = policy_snapshot["class_label"].astype(int).to_numpy()
y_official = official_snapshot["class_label"].astype(int).to_numpy()

print("X_train shape:", X_train.shape)
print("X_cal shape:", X_cal.shape)
print("X_policy shape:", X_policy.shape)
print("X_official shape:", X_official.shape)


# ============================================================
# 6. TRAIN FINAL XGBOOST MODEL
# ============================================================

print("\n[5/8] Training final XGBoost model...")

print("Parameter set:", FINAL_PARAMETER_SET)
print(test12.PARAMETER_SETS[FINAL_PARAMETER_SET])

model = test12.train_model(
    X_train,
    y_train,
    FINAL_PARAMETER_SET,
    FINAL_SEED
)

print("XGBoost training complete.")


# ============================================================
# 7. TRAIN CALIBRATOR
# ============================================================

print("\n[6/8] Training probability calibrator...")

raw_cal_proba = model.predict_proba(X_cal)

raw_cal_proba = test11.normalize_probabilities(
    raw_cal_proba
)

calibrator = test11.fit_calibrator(
    raw_cal_proba,
    y_cal,
    FINAL_SEED
)

print("Calibration complete.")


# ============================================================
# 8. EVALUATE BEFORE SAVING
# ============================================================

print("\n[7/8] Evaluating saved pipeline...")


# ------------------------------------------------------------
# PolicyDev
# ------------------------------------------------------------

policy_raw = model.predict_proba(X_policy)

policy_proba = test11.calibrated_proba(
    calibrator,
    policy_raw
)

policy_decision_metrics = test12.decision_metrics(
    y_policy,
    policy_proba
)

policy_ranking_metrics = test12.ranking_metrics(
    y_policy,
    policy_proba
)

print("\n---------------- PolicyDev ----------------")

print("\nClassification:")
for key, value in policy_decision_metrics.items():
    print(f"{key}: {value}")

print("\nRisk ranking:")
for key, value in policy_ranking_metrics.items():
    print(f"{key}: {value}")


# ------------------------------------------------------------
# Official Validation
# ------------------------------------------------------------

official_raw = model.predict_proba(X_official)

official_proba = test11.calibrated_proba(
    calibrator,
    official_raw
)

official_decision_metrics = test12.decision_metrics(
    y_official,
    official_proba
)

official_ranking_metrics = test12.ranking_metrics(
    y_official,
    official_proba
)

print("\n------------- Official Validation -------------")

print("\nClassification:")
for key, value in official_decision_metrics.items():
    print(f"{key}: {value}")

print("\nRisk ranking:")
for key, value in official_ranking_metrics.items():
    print(f"{key}: {value}")


# ============================================================
# 9. SAVE MODEL ARTIFACTS
# ============================================================

print("\n[8/8] Saving final model artifacts...")


# ------------------------------------------------------------
# A. XGBoost model
# ------------------------------------------------------------

MODEL_PATH = OUTPUT_DIR / "fleetcare_xgb_f5_p5.joblib"

joblib.dump(
    model,
    MODEL_PATH
)


# ------------------------------------------------------------
# B. Preprocessor
# ------------------------------------------------------------

PREPROCESSOR_PATH = OUTPUT_DIR / "fleetcare_preprocessor.joblib"

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)


# ------------------------------------------------------------
# C. Probability calibrator
# ------------------------------------------------------------

CALIBRATOR_PATH = OUTPUT_DIR / "fleetcare_calibrator.joblib"

joblib.dump(
    calibrator,
    CALIBRATOR_PATH
)


# ------------------------------------------------------------
# D. F5 feature columns
# ------------------------------------------------------------

FEATURE_COLUMNS_PATH = OUTPUT_DIR / "fleetcare_feature_columns.joblib"

joblib.dump(
    feature_columns,
    FEATURE_COLUMNS_PATH
)


# ------------------------------------------------------------
# E. Metadata
# ------------------------------------------------------------

METADATA_PATH = OUTPUT_DIR / "fleetcare_model_metadata.json"

metadata = {
    "model_type": "XGBClassifier",
    "objective": "multi:softprob",
    "num_classes": 5,

    "feature_set": FINAL_FEATURE_SET,
    "parameter_set": FINAL_PARAMETER_SET,

    "parameters": test12.PARAMETER_SETS[
        FINAL_PARAMETER_SET
    ],

    "training_seed": FINAL_SEED,

    "raw_feature_count": len(feature_columns),

    "class_mapping": {
        "0": ">48 operational units",
        "1": "24-48 operational units",
        "2": "12-24 operational units",
        "3": "6-12 operational units",
        "4": "0-6 operational units"
    },

    "risk_definition": {
        "Risk48": "P(C1)+P(C2)+P(C3)+P(C4)",
        "Risk24": "P(C2)+P(C3)+P(C4)",
        "Risk12": "P(C3)+P(C4)",
        "Risk6": "P(C4)"
    },

    "policydev_metrics": {
        **{
            k: float(v)
            if isinstance(v, (float, np.floating))
            else int(v)
            if isinstance(v, (int, np.integer))
            else v
            for k, v in policy_decision_metrics.items()
        },

        **{
            k: float(v)
            for k, v in policy_ranking_metrics.items()
        }
    },

    "official_validation_metrics": {
        **{
            k: float(v)
            if isinstance(v, (float, np.floating))
            else int(v)
            if isinstance(v, (int, np.integer))
            else v
            for k, v in official_decision_metrics.items()
        },

        **{
            k: float(v)
            for k, v in official_ranking_metrics.items()
        }
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
# 10. VERIFY THAT EVERYTHING CAN BE LOADED
# ============================================================

print("\nVerifying saved artifacts...")

loaded_model = joblib.load(MODEL_PATH)
loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
loaded_calibrator = joblib.load(CALIBRATOR_PATH)
loaded_features = joblib.load(FEATURE_COLUMNS_PATH)

# Test using first 10 Official Validation vehicles
test_frame = official_frame.iloc[:10].reindex(
    columns=loaded_features
)

test_X = loaded_preprocessor.transform(test_frame)

test_raw = loaded_model.predict_proba(test_X)

test_proba = test11.calibrated_proba(
    loaded_calibrator,
    test_raw
)

test_prediction = np.argmax(
    test_proba,
    axis=1
)

print("\nTest predictions:")
print(test_prediction)

print("\nTest probabilities:")
print(test_proba)

print(
    "\nProbability sums:",
    test_proba.sum(axis=1)
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("\nModel:")
print(MODEL_PATH)

print("\nPreprocessor:")
print(PREPROCESSOR_PATH)

print("\nCalibrator:")
print(CALIBRATOR_PATH)

print("\nFeature columns:")
print(FEATURE_COLUMNS_PATH)

print("\nMetadata:")
print(METADATA_PATH)

print(
    "\nFinal configuration:",
    FINAL_FEATURE_SET,
    "+",
    FINAL_PARAMETER_SET
)