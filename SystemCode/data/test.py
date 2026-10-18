import joblib

model = joblib.load(
    "processed/final_model_xgboost/model.joblib"
)

for k, v in model.get_params().items():
    if k in {
        "n_estimators", "max_depth", "learning_rate",
        "subsample", "colsample_bytree",
        "min_child_weight", "reg_alpha", "reg_lambda",
        "objective", "tree_method"
    }:
        print(f"{k}: {v}")