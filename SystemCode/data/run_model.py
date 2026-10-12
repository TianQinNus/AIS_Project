"""Run one latest FleetCare prediction per vehicle."""
import argparse
import os
from pathlib import Path

from fleetcare_inference import run_fleetcare_prediction


MODEL_FILES = (
    "fleetcare_vehicle_aware_history_xgb.joblib",
    "fleetcare_history_feature_columns.joblib",
    "fleetcare_raw_feature_columns.joblib",
)


def resolve_model_dir(value):
    if value:
        folder = Path(value).expanduser().resolve()
        missing = [name for name in MODEL_FILES if not (folder / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Model directory {folder} is missing: {missing}")
        return folder

    here = Path(__file__).resolve()
    candidates = []
    for root in [here.parent, *here.parents]:
        candidates.extend(
            [
                root / "data" / "processed" / "final_model",
                root / "processed" / "final_model",
                root / "final_model",
            ]
        )
    for folder in candidates:
        if all((folder / name).is_file() for name in MODEL_FILES):
            print(f"Auto-detected model directory: {folder}")
            return folder
    raise FileNotFoundError(
        "Could not locate the three final model files automatically. Pass --model-dir."
    )


def main():
    parser = argparse.ArgumentParser(description="Latest FleetCare prediction per vehicle")
    parser.add_argument("--model-dir", default=None, help="Optional final-model folder")
    parser.add_argument("--username", default=os.getenv("PGUSER", "tianqin-gg"))
    parser.add_argument("--password", default=None, help="Uses PGPASSWORD when omitted")
    parser.add_argument("--host", default=os.getenv("PGHOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--database", default=os.getenv("PGDATABASE", "fleetcare"))
    parser.add_argument("--analysis-table", default="analysis")
    parser.add_argument("--shap-table", default="prediction_shap")
    parser.add_argument("--top-shap", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--as-of-date", default=None, help="Prediction batch date YYYY-MM-DD")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    return run_fleetcare_prediction(
        model_dir=model_dir,
        username=args.username,
        password=args.password,
        host=args.host,
        port=args.port,
        database=args.database,
        analysis_table=args.analysis_table,
        shap_table=args.shap_table,
        top_shap=args.top_shap,
        persist=not args.dry_run,
        as_of_date=args.as_of_date,
    )


if __name__ == "__main__":
    main()
