# ============================================================
# FleetCare AI - Final System Pipeline
#
# 1. Load final F5 + P5 XGBoost model
# 2. Generate validation predictions
# 3. Evaluate final model
# 4. Generate evaluation plots
# 5. Apply deterministic maintenance rule engine
# 6. Generate frontend-ready CSV / JSON
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

import test_11c_generalization_robust_xgboost as test11


# ============================================================
# 0. CONFIGURATION
# ============================================================

MODEL_DIR = Path("processed/final_model")
RAW_DIR = Path("raw")
OUTPUT_DIR = Path("processed/system_output")
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_IDS = [0, 1, 2, 3, 4]

CLASS_NAMES = {
    0: "C0",
    1: "C1",
    2: "C2",
    3: "C3",
    4: "C4",
}

CLASS_DESCRIPTIONS = {
    0: "No imminent repair (>48 units)",
    1: "Early repair proximity (24-48 units)",
    2: "Increasing repair proximity (12-24 units)",
    3: "Near repair (6-12 units)",
    4: "Imminent repair (0-6 units)",
}

# ------------------------------------------------------------
# Prototype rule-engine threshold.
#
# IMPORTANT:
# This is a PROJECT POLICY, not an official SCANIA rule.
# It can later be tuned using validation performance.
# ------------------------------------------------------------

LOW_CONFIDENCE_THRESHOLD = 0.40


# ============================================================
# 1. LOAD SAVED MODEL COMPONENTS
# ============================================================

print("=" * 70)
print("FleetCare AI - Final System Pipeline")
print("=" * 70)

print("\n[1/10] Loading saved model...")

model = joblib.load(
    MODEL_DIR / "fleetcare_xgb_f5_p5.joblib"
)

preprocessor = joblib.load(
    MODEL_DIR / "fleetcare_preprocessor.joblib"
)

calibrator = joblib.load(
    MODEL_DIR / "fleetcare_calibrator.joblib"
)

feature_columns = joblib.load(
    MODEL_DIR / "fleetcare_feature_columns.joblib"
)

print("Model loaded successfully.")
print("Number of saved F5 features:", len(feature_columns))


# ============================================================
# 2. LOAD VALIDATION DATA
# ============================================================

print("\n[2/10] Loading validation data...")

raw_data = pd.read_csv(
    RAW_DIR / "validation_operational_readouts.csv"
)

specifications = pd.read_csv(
    RAW_DIR / "validation_specifications.csv"
)

labels = pd.read_csv(
    RAW_DIR / "validation_labels.csv"
)

print("Operational readouts:", raw_data.shape)
print("Specifications:", specifications.shape)
print("Labels:", labels.shape)

print("\nLabel columns:")
print(labels.columns.tolist())


# ============================================================
# 3. FIND LABEL COLUMN
# ============================================================

# Different releases/code may use different names.
# Try common possibilities automatically.

possible_label_columns = [
    "class_label",
    "label",
    "predicted_class",
    "class",
]

LABEL_COLUMN = None

for column in possible_label_columns:
    if column in labels.columns:
        LABEL_COLUMN = column
        break

if LABEL_COLUMN is None:

    non_id_columns = [
        c for c in labels.columns
        if c != "vehicle_id"
    ]

    if len(non_id_columns) == 1:
        LABEL_COLUMN = non_id_columns[0]
    else:
        raise ValueError(
            "Could not automatically identify the label column. "
            f"Available columns: {labels.columns.tolist()}"
        )

print("Using label column:", LABEL_COLUMN)


# ============================================================
# 4. GENERATE TEMPORAL FEATURES
# ============================================================

print("\n[3/10] Generating 6/12/24/48 temporal features...")

raw_features = [
    column
    for column in raw_data.columns
    if column not in {
        "vehicle_id",
        "time_step"
    }
]

temporal_data = test11.make_generalization_features(
    raw_data,
    raw_features
)

print("Temporal feature table:", temporal_data.shape)


# ============================================================
# 5. TAKE LATEST SNAPSHOT FOR EACH VEHICLE
# ============================================================

print("\n[4/10] Selecting latest snapshot for each vehicle...")

latest = (
    temporal_data
    .sort_values(["vehicle_id", "time_step"])
    .groupby("vehicle_id", as_index=False)
    .tail(1)
    .copy()
)

# Ensure deterministic order
latest = (
    latest
    .sort_values("vehicle_id")
    .reset_index(drop=True)
)

print("Vehicles to predict:", len(latest))


# ============================================================
# 6. ADD VEHICLE SPECIFICATIONS
# ============================================================

latest = test11.add_specs(
    latest,
    specifications
)

feature_frame = test11.feature_frame(
    latest
)


# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================

def predict_from_feature_frame(frame):

    # --------------------------------------------------------
    # Ensure exactly the same F5 columns/order as training
    # --------------------------------------------------------

    X = frame.reindex(
        columns=feature_columns
    )

    # --------------------------------------------------------
    # Same preprocessing as training
    # --------------------------------------------------------

    X_processed = preprocessor.transform(X)

    # --------------------------------------------------------
    # XGBoost raw probabilities
    # --------------------------------------------------------

    raw_proba = model.predict_proba(
        X_processed
    )

    # --------------------------------------------------------
    # Probability calibration
    # --------------------------------------------------------

    proba = test11.calibrated_proba(
        calibrator,
        raw_proba
    )

    # --------------------------------------------------------
    # C0-C4 hard prediction
    # --------------------------------------------------------

    predicted_class = np.argmax(
        proba,
        axis=1
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = np.max(
        proba,
        axis=1
    )

    # --------------------------------------------------------
    # Cumulative risk horizons
    # --------------------------------------------------------

    risk48 = (
        proba[:, 1]
        + proba[:, 2]
        + proba[:, 3]
        + proba[:, 4]
    )

    risk24 = (
        proba[:, 2]
        + proba[:, 3]
        + proba[:, 4]
    )

    risk12 = (
        proba[:, 3]
        + proba[:, 4]
    )

    risk6 = proba[:, 4]

    return pd.DataFrame({

        "predicted_class": predicted_class,

        "P_C0": proba[:, 0],
        "P_C1": proba[:, 1],
        "P_C2": proba[:, 2],
        "P_C3": proba[:, 3],
        "P_C4": proba[:, 4],

        "confidence": confidence,

        "Risk48": risk48,
        "Risk24": risk24,
        "Risk12": risk12,
        "Risk6": risk6,
    })


# ============================================================
# 8. RUN MODEL
# ============================================================

print("\n[5/10] Running final F5 + P5 XGBoost model...")

predictions = predict_from_feature_frame(
    feature_frame
)

predictions.insert(
    0,
    "vehicle_id",
    latest["vehicle_id"].to_numpy()
)

predictions["predicted_state"] = (
    predictions["predicted_class"]
    .map(CLASS_NAMES)
)

predictions["state_description"] = (
    predictions["predicted_class"]
    .map(CLASS_DESCRIPTIONS)
)

print("Prediction complete.")

print("\nExample predictions:")

print(
    predictions.head(10).to_string(
        index=False
    )
)


# ============================================================
# 9. MERGE GROUND TRUTH
# ============================================================

print("\n[6/10] Evaluating model...")

evaluation_df = predictions.merge(
    labels[
        ["vehicle_id", LABEL_COLUMN]
    ],
    on="vehicle_id",
    how="inner"
)

evaluation_df = evaluation_df.rename(
    columns={
        LABEL_COLUMN: "actual_class"
    }
)

evaluation_df["actual_class"] = (
    evaluation_df["actual_class"]
    .astype(int)
)

y_true = evaluation_df["actual_class"].to_numpy()

y_pred = (
    evaluation_df["predicted_class"]
    .to_numpy()
)

print(
    "Matched validation vehicles:",
    len(evaluation_df)
)


# ============================================================
# 10. BASIC EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=CLASS_IDS,
    average="macro",
    zero_division=0
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

weighted_f1 = f1_score(
    y_true,
    y_pred,
    labels=CLASS_IDS,
    average="weighted",
    zero_division=0
)

print("\n======================================")
print("FINAL MODEL EVALUATION")
print("======================================")

print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

report_text = classification_report(
    y_true,
    y_pred,
    labels=CLASS_IDS,
    target_names=[
        "C0",
        "C1",
        "C2",
        "C3",
        "C4"
    ],
    zero_division=0
)

print("\nClassification Report:")
print(report_text)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(report_text)


# Save report as CSV as well

report_dict = classification_report(
    y_true,
    y_pred,
    labels=CLASS_IDS,
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

report_df = (
    pd.DataFrame(report_dict)
    .transpose()
)

report_df.to_csv(
    OUTPUT_DIR / "classification_report.csv"
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\n[7/10] Creating evaluation plots...")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASS_IDS
)

fig, ax = plt.subplots(
    figsize=(8, 7)
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
    ax=ax,
    values_format="d",
    cmap="Blues",
    colorbar=False
)

ax.set_title(
    "FleetCare AI - Five-Class Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. NORMALIZED CONFUSION MATRIX
# ============================================================

cm_normalized = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASS_IDS,
    normalize="true"
)

fig, ax = plt.subplots(
    figsize=(8, 7)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=[
        "C0",
        "C1",
        "C2",
        "C3",
        "C4"
    ]
)

disp.plot(
    ax=ax,
    values_format=".2f",
    cmap="Blues",
    colorbar=False
)

ax.set_title(
    "FleetCare AI - Normalized Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "confusion_matrix_normalized.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. RISK-HORIZON EVALUATION
# ============================================================

HORIZONS = {

    "Risk48": [1, 2, 3, 4],

    "Risk24": [2, 3, 4],

    "Risk12": [3, 4],

    "Risk6": [4],
}

risk_metrics = []

fig, ax = plt.subplots(
    figsize=(9, 7)
)

for risk_name, positive_classes in HORIZONS.items():

    y_binary = (
        evaluation_df["actual_class"]
        .isin(positive_classes)
        .astype(int)
        .to_numpy()
    )

    scores = (
        evaluation_df[risk_name]
        .to_numpy()
    )

    ap = average_precision_score(
        y_binary,
        scores
    )

    try:
        auc = roc_auc_score(
            y_binary,
            scores
        )
    except ValueError:
        auc = np.nan

    precision, recall, _ = (
        precision_recall_curve(
            y_binary,
            scores
        )
    )

    ax.plot(
        recall,
        precision,
        label=(
            f"{risk_name} "
            f"(AP={ap:.3f})"
        )
    )

    # --------------------------------------------------------
    # Top 5% evaluation
    # --------------------------------------------------------

    k5 = max(
        1,
        int(np.ceil(
            len(scores) * 0.05
        ))
    )

    top5_idx = np.argsort(
        -scores,
        kind="stable"
    )[:k5]

    total_positive = y_binary.sum()

    if total_positive > 0:

        top5_recall = (
            y_binary[top5_idx].sum()
            / total_positive
        )

    else:
        top5_recall = np.nan

    top5_precision = (
        y_binary[top5_idx].mean()
    )

    # --------------------------------------------------------
    # Top 10%
    # --------------------------------------------------------

    k10 = max(
        1,
        int(np.ceil(
            len(scores) * 0.10
        ))
    )

    top10_idx = np.argsort(
        -scores,
        kind="stable"
    )[:k10]

    if total_positive > 0:

        top10_recall = (
            y_binary[top10_idx].sum()
            / total_positive
        )

    else:
        top10_recall = np.nan

    top10_precision = (
        y_binary[top10_idx].mean()
    )

    risk_metrics.append({

        "Horizon": risk_name,

        "AveragePrecision": ap,

        "ROCAUC": auc,

        "Top5Recall": top5_recall,

        "Top5Precision": top5_precision,

        "Top10Recall": top10_recall,

        "Top10Precision": top10_precision,
    })


ax.set_xlabel("Recall")
ax.set_ylabel("Precision")

ax.set_title(
    "FleetCare AI - Risk Horizon Precision-Recall Curves"
)

ax.legend()

ax.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "risk_precision_recall_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


risk_metrics_df = pd.DataFrame(
    risk_metrics
)

risk_metrics_df.to_csv(
    OUTPUT_DIR / "risk_metrics.csv",
    index=False
)

print("\nRisk Horizon Metrics:")
print(
    risk_metrics_df.to_string(
        index=False
    )
)


# ============================================================
# 15. RULE ENGINE
# ============================================================

print("\n[8/10] Applying maintenance rule engine...")


def maintenance_rule_engine(row):

    predicted_class = int(
        row["predicted_class"]
    )

    confidence = float(
        row["confidence"]
    )

    risk48 = float(
        row["Risk48"]
    )

    risk24 = float(
        row["Risk24"]
    )

    risk12 = float(
        row["Risk12"]
    )

    risk6 = float(
        row["Risk6"]
    )

    # --------------------------------------------------------
    # RULE 1
    #
    # High urgency + low confidence
    #
    # We do not automatically escalate uncertain C3/C4 cases.
    # Human review is requested.
    # --------------------------------------------------------

    if (
        predicted_class in [3, 4]
        and confidence < LOW_CONFIDENCE_THRESHOLD
    ):

        rule_id = "R1"

        action = "MANUAL_CHECK"

        priority = "REVIEW"

        reason = (
            "The model indicates a high-urgency "
            f"{CLASS_NAMES[predicted_class]} state, "
            "but prediction confidence is low. "
            "Human review is recommended before "
            "maintenance prioritisation."
        )

    # --------------------------------------------------------
    # RULE 2 - C4
    # --------------------------------------------------------

    elif predicted_class == 4:

        rule_id = "R2"

        action = "PRIORITY_INSPECT"

        priority = "CRITICAL"

        reason = (
            "The vehicle is classified as C4, "
            "the closest repair-proximity state "
            "(0-6 operational units)."
        )

    # --------------------------------------------------------
    # RULE 3 - C3
    # --------------------------------------------------------

    elif predicted_class == 3:

        rule_id = "R3"

        action = "INSPECT"

        priority = "HIGH"

        reason = (
            "The vehicle is classified as C3, "
            "indicating near-term repair proximity "
            "(6-12 operational units)."
        )

    # --------------------------------------------------------
    # RULE 4 - C2
    # --------------------------------------------------------

    elif predicted_class == 2:

        rule_id = "R4"

        action = "ENHANCED_MONITOR"

        priority = "MEDIUM"

        reason = (
            "The vehicle is classified as C2, "
            "indicating increasing repair proximity "
            "(12-24 operational units)."
        )

    # --------------------------------------------------------
    # RULE 5 - C1
    # --------------------------------------------------------

    elif predicted_class == 1:

        rule_id = "R5"

        action = "MONITOR"

        priority = "LOW"

        reason = (
            "The vehicle is classified as C1, "
            "indicating early repair proximity "
            "(24-48 operational units)."
        )

    # --------------------------------------------------------
    # RULE 6 - C0
    # --------------------------------------------------------

    else:

        rule_id = "R6"

        action = "CONTINUE"

        priority = "NORMAL"

        reason = (
            "The vehicle is classified as C0. "
            "No Component X repair is predicted "
            "within the next 48 operational units."
        )

    # --------------------------------------------------------
    # Rule trace
    # --------------------------------------------------------

    rule_trace = (
        f"{rule_id}: "
        f"class={CLASS_NAMES[predicted_class]}, "
        f"confidence={confidence:.3f}, "
        f"Risk48={risk48:.3f}, "
        f"Risk24={risk24:.3f}, "
        f"Risk12={risk12:.3f}, "
        f"Risk6={risk6:.3f} "
        f"-> {action}"
    )

    return pd.Series({

        "rule_id": rule_id,

        "action": action,

        "priority": priority,

        "rule_reason": reason,

        "rule_trace": rule_trace,
    })


rule_results = predictions.apply(
    maintenance_rule_engine,
    axis=1
)

system_df = pd.concat(
    [
        predictions,
        rule_results
    ],
    axis=1
)


# ============================================================
# 16. ADD FRONTEND DISPLAY FIELDS
# ============================================================

# Risk score used for default dashboard sorting
system_df["overall_risk_score"] = (
    system_df["Risk48"]
)

# Priority rank used by frontend sorting

PRIORITY_ORDER = {
    "CRITICAL": 5,
    "HIGH": 4,
    "REVIEW": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "NORMAL": 0,
}

system_df["priority_rank"] = (
    system_df["priority"]
    .map(PRIORITY_ORDER)
)

# Sort highest-priority vehicles first

system_df = (
    system_df
    .sort_values(
        [
            "priority_rank",
            "Risk6",
            "Risk12",
            "Risk48"
        ],
        ascending=[
            False,
            False,
            False,
            False
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 17. FLEET SUMMARY
# ============================================================

fleet_summary = {

    "total_vehicles": int(
        len(system_df)
    ),

    "class_distribution": {
        CLASS_NAMES[class_id]: int(
            (
                system_df["predicted_class"]
                == class_id
            ).sum()
        )
        for class_id in CLASS_IDS
    },

    "action_distribution": (
        system_df["action"]
        .value_counts()
        .to_dict()
    ),

    "priority_distribution": (
        system_df["priority"]
        .value_counts()
        .to_dict()
    ),

    "average_risk": {

        "Risk48": float(
            system_df["Risk48"].mean()
        ),

        "Risk24": float(
            system_df["Risk24"].mean()
        ),

        "Risk12": float(
            system_df["Risk12"].mean()
        ),

        "Risk6": float(
            system_df["Risk6"].mean()
        ),
    },

    "model": {

        "name": (
            "XGBoost F5 + P5"
        ),

        "feature_set": (
            "F5_rate_plus_short_change"
        ),

        "parameter_set": (
            "P5_less_regularized"
        ),
    },
}


# ============================================================
# 18. SAVE FRONTEND DATA
# ============================================================

print("\n[9/10] Generating frontend data...")

# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

system_df.to_csv(
    OUTPUT_DIR / "frontend_vehicle_data.csv",
    index=False
)


# ------------------------------------------------------------
# JSON
# ------------------------------------------------------------

frontend_records = []

for _, row in system_df.iterrows():

    record = {

        "vehicle_id": (
            row["vehicle_id"].item()
            if hasattr(row["vehicle_id"], "item")
            else row["vehicle_id"]
        ),

        "prediction": {

            "class_id": int(
                row["predicted_class"]
            ),

            "state": row[
                "predicted_state"
            ],

            "description": row[
                "state_description"
            ],

            "confidence": round(
                float(row["confidence"]),
                6
            ),

            "probabilities": {

                "C0": round(
                    float(row["P_C0"]),
                    6
                ),

                "C1": round(
                    float(row["P_C1"]),
                    6
                ),

                "C2": round(
                    float(row["P_C2"]),
                    6
                ),

                "C3": round(
                    float(row["P_C3"]),
                    6
                ),

                "C4": round(
                    float(row["P_C4"]),
                    6
                ),
            },
        },

        "risk": {

            "Risk48": round(
                float(row["Risk48"]),
                6
            ),

            "Risk24": round(
                float(row["Risk24"]),
                6
            ),

            "Risk12": round(
                float(row["Risk12"]),
                6
            ),

            "Risk6": round(
                float(row["Risk6"]),
                6
            ),
        },

        "decision": {

            "priority": row[
                "priority"
            ],

            "action": row[
                "action"
            ],

            "rule_id": row[
                "rule_id"
            ],

            "reason": row[
                "rule_reason"
            ],

            "trace": row[
                "rule_trace"
            ],
        },
    }

    frontend_records.append(
        record
    )


with open(
    OUTPUT_DIR / "frontend_vehicle_data.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        frontend_records,
        f,
        indent=2,
        ensure_ascii=False
    )


with open(
    OUTPUT_DIR / "fleet_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        fleet_summary,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 19. SAVE MODEL EVALUATION SUMMARY
# ============================================================

evaluation_summary = {

    "Accuracy": float(
        accuracy
    ),

    "MacroPrecision": float(
        macro_precision
    ),

    "MacroRecall": float(
        macro_recall
    ),

    "MacroF1": float(
        macro_f1
    ),

    "WeightedF1": float(
        weighted_f1
    ),
}

with open(
    OUTPUT_DIR / "evaluation_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation_summary,
        f,
        indent=2
    )


# ============================================================
# 20. PRINT FINAL SUMMARY
# ============================================================

print("\n[10/10] Pipeline complete.")

print("\n======================================")
print("FLEET SUMMARY")
print("======================================")

print(
    "Total vehicles:",
    fleet_summary["total_vehicles"]
)

print("\nPredicted classes:")

for key, value in (
    fleet_summary[
        "class_distribution"
    ].items()
):

    print(
        f"  {key}: {value}"
    )


print("\nMaintenance actions:")

for key, value in (
    fleet_summary[
        "action_distribution"
    ].items()
):

    print(
        f"  {key}: {value}"
    )


print("\n======================================")
print("OUTPUT FILES")
print("======================================")

print(
    OUTPUT_DIR
    / "frontend_vehicle_data.csv"
)

print(
    OUTPUT_DIR
    / "frontend_vehicle_data.json"
)

print(
    OUTPUT_DIR
    / "fleet_summary.json"
)

print(
    OUTPUT_DIR
    / "evaluation_summary.json"
)

print(
    OUTPUT_DIR
    / "classification_report.csv"
)

print(
    OUTPUT_DIR
    / "risk_metrics.csv"
)

print(
    PLOT_DIR
    / "confusion_matrix.png"
)

print(
    PLOT_DIR
    / "confusion_matrix_normalized.png"
)

print(
    PLOT_DIR
    / "risk_precision_recall_curves.png"
)

print("\nFleetCare AI pipeline finished successfully.")