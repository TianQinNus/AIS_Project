"""Test 50: GRU degradation-sequence reranker for C4.

Five individual readout intervals are consumed chronologically by a GRU. The
network produces an independent C4 sequence rank that is conservatively blended
with the existing tree ensemble. Validation chooses the blend; Test is loaded
only if the GRU improves the primary Validation objective.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fleetcare_c4_risk_inference import (
    MODEL_FILE,
    class_weights,
    ensemble_c4_risk_score,
    fit_spec_schema,
    last_rows,
    load_csv,
    percentile_rank,
    ranking_metrics,
    select_training,
    training_pool,
)
from test_48_interval_sequence_search import (
    interval_sequence_features,
    read_operational,
)


class SequenceGRU(nn.Module):
    def __init__(self, inputs: int, hidden: int = 48):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 24), nn.SiLU(),
            nn.Dropout(0.15), nn.Linear(24, 1),
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1]).squeeze(1)


def prepare_raw(op, selected, raw_columns):
    sequence, names = interval_sequence_features(op, selected, raw_columns)
    per_step = len(names) // 5
    if per_step * 5 != len(names):
        raise ValueError("Sequence feature count is not divisible by five")
    values = sequence.reshape(len(sequence), 5, per_step)
    first_names = names[:per_step]
    ratio_columns = np.asarray([
        name.startswith("step_ratio1:") for name in first_names
    ])
    values[:, :, ratio_columns] = np.log1p(
        np.clip(values[:, :, ratio_columns], 0, 1e3)
    )
    # interval_sequence_features is newest to oldest; GRU reads chronologically.
    return values[:, ::-1, :].copy(), first_names


def fit_scaler(values):
    with np.errstate(all="ignore"):
        median = np.nanmedian(values, axis=(0, 1))
        q25 = np.nanquantile(values, 0.25, axis=(0, 1))
        q75 = np.nanquantile(values, 0.75, axis=(0, 1))
    median = np.nan_to_num(median, nan=0.0)
    scale = np.nan_to_num(q75 - q25, nan=1.0, posinf=1.0, neginf=1.0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return median.astype(np.float32), scale.astype(np.float32)


def transform(values, median, scale):
    valid = np.isfinite(values)
    normalized = (np.where(valid, values, median) - median) / scale
    normalized = np.clip(normalized, -10, 10).astype(np.float32)
    coverage = valid.mean(axis=2, keepdims=True).astype(np.float32)
    return np.concatenate([normalized, coverage], axis=2)


def train_gru(x_train, y_train, sample_weight, x_validation=None, y_validation=None,
              epochs=15, fixed_epochs=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    model = SequenceGRU(x_train.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32)),
        torch.from_numpy(sample_weight.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    best_state = copy.deepcopy(model.state_dict())
    best_ap = -np.inf
    best_epoch = 1
    patience = 0
    total_epochs = fixed_epochs or epochs
    for epoch in range(1, total_epochs + 1):
        model.train()
        for xb, yb, wb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (criterion(model(xb), yb) * wb).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if fixed_epochs is not None:
            continue
        model.eval()
        with torch.no_grad():
            score = torch.sigmoid(model(torch.from_numpy(x_validation))).numpy()
        ap = average_precision_score(y_validation, score)
        print(f"GRU epoch={epoch} validation_AP={ap:.6f}", flush=True)
        if ap > best_ap + 1e-5:
            best_ap = ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if epoch >= 6 and patience >= 3:
            break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_ap


def predict(model, x):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), 2048):
            chunks.append(torch.sigmoid(model(torch.from_numpy(x[start:start + 2048]))).numpy())
    return np.concatenate(chunks)


def blend(first, second, second_weight):
    return np.power(np.clip(first, 1e-6, 1), 1 - second_weight) * np.power(
        np.clip(second, 1e-6, 1), second_weight
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=here / "raw")
    parser.add_argument(
        "--current-model-dir", type=Path,
        default=here / "processed" / "fleetcare_c4_risk_model",
    )
    parser.add_argument(
        "--refit-model-dir", type=Path,
        default=here / "processed" / "test_43_refit_train_validation",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=here / "processed" / "test_50_gru_sequence_reranker",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_op = read_operational(args.data_dir, "train")
    validation_op = read_operational(args.data_dir, "validation")
    train_spec = load_csv(args.data_dir, "train_specifications.csv")
    tte = load_csv(args.data_dir, "train_tte.csv")
    validation_labels = load_csv(args.data_dir, "validation_labels.csv")
    raw_columns = [c for c in train_op if c not in ("vehicle_id", "time_step")]
    schema = fit_spec_schema(train_spec)
    train_selected = select_training(training_pool(train_op, tte), seed=42)
    validation_selected = last_rows(validation_op, validation_labels)
    print("Building chronological GRU sequences...", flush=True)
    raw_train, sequence_names = prepare_raw(train_op, train_selected, raw_columns)
    raw_validation, validation_names = prepare_raw(
        validation_op, validation_selected, raw_columns
    )
    if sequence_names != validation_names:
        raise ValueError("Validation sequence schema differs")
    median, scale = fit_scaler(raw_train)
    x_train = transform(raw_train, median, scale)
    x_validation = transform(raw_validation, median, scale)
    labels = train_selected["class_label"].to_numpy(int)
    y_train = (labels == 4).astype(np.int8)
    y_validation = (
        validation_selected["class_label"].to_numpy(int) == 4
    ).astype(np.int8)
    row_weight = class_weights(
        y_train, train_selected["sw"].to_numpy(float), 0.35
    )
    row_weight = row_weight / row_weight.mean()
    model, best_epoch, best_ap = train_gru(
        x_train, y_train, row_weight, x_validation, y_validation
    )
    gru_rank = percentile_rank(predict(model, x_validation))

    # Reuse the already persisted first-stage ranks. Besides avoiding needless
    # recomputation, this keeps the Torch process independent from XGBoost's
    # OpenMP runtime on macOS.
    persisted_validation = pd.read_csv(
        args.current_model_dir / "validation_predictions.csv"
    )[["vehicle_id", "c4_risk_score"]]
    ordered_validation = validation_selected[["vehicle_id"]].merge(
        persisted_validation, on="vehicle_id", how="left", validate="one_to_one"
    )
    if ordered_validation["c4_risk_score"].isna().any():
        raise ValueError("Missing persisted Validation first-stage score")
    first_rank = ordered_validation["c4_risk_score"].to_numpy(float)
    candidates = [{
        "gru_weight": 0.0, **ranking_metrics(y_validation, first_rank)
    }]
    for weight in (0.15, 0.25, 0.50, 0.75, 1.0):
        candidates.append({
            "gru_weight": weight,
            **ranking_metrics(y_validation, blend(first_rank, gru_rank, weight)),
        })
    table = pd.DataFrame(candidates).sort_values(
        ["top5_c4_found", "top2_c4_found", "c4_average_precision", "c4_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    table.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    winner = table.iloc[0].to_dict()
    print("Validation GRU candidates:", flush=True)
    print(table.to_string(index=False), flush=True)
    if float(winner["gru_weight"]) == 0:
        result = {
            "best_epoch": best_epoch, "best_gru_ap": best_ap,
            "winner": winner, "test_loaded": False,
        }
        (args.output_dir / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print("GRU rejected; Test was not loaded.", flush=True)
        return

    # Refit for the selected number of epochs on Train plus Validation.
    raw_fit = np.concatenate([raw_train, raw_validation], axis=0)
    fit_labels = np.concatenate([
        labels, validation_selected["class_label"].to_numpy(int)
    ])
    fit_y = (fit_labels == 4).astype(np.int8)
    fit_base_weight = np.concatenate([
        train_selected["sw"].to_numpy(float), np.ones(len(validation_selected))
    ])
    fit_weight = class_weights(fit_y, fit_base_weight, 0.35)
    fit_weight = fit_weight / fit_weight.mean()
    final_median, final_scale = fit_scaler(raw_fit)
    x_fit = transform(raw_fit, final_median, final_scale)
    print(f"Refitting GRU for {best_epoch} epochs before Test...", flush=True)
    final_model, _, _ = train_gru(
        x_fit, fit_y, fit_weight, fixed_epochs=best_epoch
    )

    test_op = read_operational(args.data_dir, "test")
    test_labels = load_csv(args.data_dir, "test_labels.csv")
    test_selected = last_rows(test_op, test_labels)
    raw_test, test_names = prepare_raw(test_op, test_selected, raw_columns)
    if sequence_names != test_names:
        raise ValueError("Test sequence schema differs")
    x_test = transform(raw_test, final_median, final_scale)
    test_gru_rank = percentile_rank(predict(final_model, x_test))
    persisted_test = pd.read_csv(
        args.refit_model_dir / "test_predictions.csv"
    )[["vehicle_id", "c4_risk_score"]]
    ordered_test = test_selected[["vehicle_id"]].merge(
        persisted_test, on="vehicle_id", how="left", validate="one_to_one"
    )
    if ordered_test["c4_risk_score"].isna().any():
        raise ValueError("Missing persisted Test first-stage score")
    test_first = ordered_test["c4_risk_score"].to_numpy(float)
    final_score = blend(test_first, test_gru_rank, float(winner["gru_weight"]))
    y_test = (test_selected["class_label"].to_numpy(int) == 4).astype(np.int8)
    test_report = ranking_metrics(y_test, final_score)
    test43 = json.loads(
        (args.refit_model_dir / "comparison.json").read_text()
    )["candidate_train_plus_validation"]
    result = {
        "selection_rule": (
            "Validation lexicographic: Top-5 C4, Top-2 C4, AP, then AUC"
        ),
        "best_epoch": best_epoch,
        "validation_winner": winner,
        "test43_reference": test43,
        "test50_gru_sequence": test_report,
        "selected_for_deployment": False,
        "test_note": (
            "Test was loaded only after Validation selection in this experiment, "
            "but had been inspected in earlier project work."
        ),
    }
    predictions = pd.DataFrame({
        "vehicle_id": test_selected["vehicle_id"].to_numpy(),
        "actual_class": test_selected["class_label"].to_numpy(int),
        "first_stage_score": test_first,
        "gru_sequence_rank": test_gru_rank,
        "c4_risk_score": final_score,
        "ranking": pd.Series(final_score).rank(
            ascending=False, method="first"
        ).astype(int),
    }).sort_values("ranking")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    torch.save({
        "state_dict": final_model.state_dict(),
        "median": final_median,
        "scale": final_scale,
        "sequence_names": sequence_names,
        "gru_weight": winner["gru_weight"],
        "input_size": x_fit.shape[2],
    }, args.output_dir / "fleetcare_c4_gru_candidate.pt")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
