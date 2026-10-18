# FleetCare C4 Risk Ranking Ensemble

## 1. Train, save and evaluate

```bash
cd /Users/tianqin-gg/Desktop/AIS_Project
.venv/bin/python SystemCode/data/train_fleetcare_c4_risk_model.py
```

The command builds 534 past-only features, trains the C4 risk ranking ensemble
and the maintenance-horizon classifier, then packages all four internal
components into `fleetcare_c4_risk_ensemble.joblib`. Evaluation files are saved under
`SystemCode/data/processed/fleetcare_c4_risk_model/`.

Final test result: C4 AP `0.041185`, ROC-AUC `0.720488`, Precision@Top 2%
`6.93%`, Precision@Top 5% `4.35%`.

## 2. Preview the latest production prediction

```bash
.venv/bin/python SystemCode/data/fleetcare_c4_risk_inference.py
```

The script reads active, non-maintenance vehicles directly from PostgreSQL. It
calculates risk on each vehicle's latest five available readouts, applies the
temporal rule engine, and outputs only the decision for the latest readout.
The stored `prediction_at` is the production run timestamp plus one day.
Preview CSVs are saved under
`SystemCode/data/processed/fleetcare_c4_risk_inference/`.

## 3. Write the latest prediction to PostgreSQL

```bash
.venv/bin/python SystemCode/data/fleetcare_c4_risk_inference.py \
  --write-db \
  --replace-existing
```

`--replace-existing` replaces an existing prediction made on the same Singapore
calendar date for the eligible vehicle IDs. One analysis row per vehicle and
its top-five XGBoost C4-component SHAP rows are inserted in one transaction.
`prediction_step` is not used.

The rule order is `TRD → TR0 → TRV → TR1 → TR2 → TR3 → TR4 → TR5 → TR6`.
The C4 risk ensemble ranks the scarce inspection queue; the rules supply
temporal monitoring, data-quality review and traceable actions.
