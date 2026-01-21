
from sklearn.metrics import (
    precision_score,
    roc_auc_score,
    recall_score,
    f1_score, 
)

from util2 import num_features_selector, cat_features_selector
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from model_config import MODEL_CONFIG
from huggingface_hub import HfApi
import pandas as pd
import numpy as np
import mlflow
import time
import sys
import os

HF_REPO = os.getenv("HF_REPO")
hfApi = HfApi(token=os.getenv("HF_TOKEN"))
print(f"Loading Train/Test files from HF Dataset: {HF_REPO}")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    print("MLFLOW_TRACKING_URI is not set")
    sys.exit(1)

# Checking train/test splits are present or not
files = ["X_train", "y_train", "X_test", "y_test"]
for f in files:
    path = f"hf://datasets/{HF_REPO}/{f}.csv"
    try:
        pd.read_csv(path, nrows=1)
    except FileNotFoundError:
        print(f"{f}.csv missing @HF Dataset.")
        sys.exit(1)
    except Exception as e:
        print(f"Error Checking Path: {path} | Err: {e}")
        sys.exit(1)

X_train = pd.read_csv(f"hf://datasets/{HF_REPO}/X_train.csv")
y_train = pd.read_csv(f"hf://datasets/{HF_REPO}/y_train.csv")
X_test = pd.read_csv(f"hf://datasets/{HF_REPO}/X_test.csv")
y_test = pd.read_csv(f"hf://datasets/{HF_REPO}/y_test.csv")

print(X_train.info())

# column processors
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
])

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

col_processor = make_column_transformer(
    (cat_pipeline, cat_features_selector),
    (num_pipeline, num_features_selector),
    remainder = "drop"
)
col_processor.set_output(transform="pandas")

### model complexity & performance ###
def get_model_complexity(model):
    if hasattr(model, "estimators_"):
        return sum(
            est.tree_.node_count
            for est in model.estimators_
            if hasattr(est, "tree_")
        )
    if hasattr(model, "tree_"):
        return model.tree_.node_count
    if hasattr(model, "coef_"):
        return model.coef_.size
    return None

def measure_inference_latency(model, X, n_runs=100):
    X_sample = X.iloc[:1]

    # warm-up
    model.predict(X_sample)

    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(X_sample)
    end = time.perf_counter()

    return round((end - start) / n_runs * 1000, 4)
###

# model eval
results = []
EXPERIMENT_NAME = "wellness-purchase-propensity"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

for model_name, cfg in MODEL_CONFIG.items():
    run_name = f"{model_name}_{time.strftime('%Y-%m%d_%H%M')}"
    print(f"mlFlow Run: {run_name}")
    with mlflow.start_run(run_name=run_name):     
        mlflow.set_tag("model_name", model_name)

        pipeline = make_pipeline(col_processor, cfg["estimator"])
        search = RandomizedSearchCV(
            param_distributions=cfg["grid_params"],
            estimator=pipeline,
            scoring="roc_auc",
            random_state=42,
            n_iter=20,
            n_jobs=-1,
            verbose=1,
            cv=5,
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # predictions
        y_pred_proba = best_model.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # metrics
        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)

        # log metrics
        mlflow.log_metric("cv_auc_mean", search.best_score_)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_f1", test_f1)

        # log hyperparameters
        mlflow.log_params(search.best_params_)

        # feature importance
        perm = permutation_importance(
            best_model,
            X_test,
            y_test,
            scoring="roc_auc",
            random_state=42,
            n_repeats=5,
            n_jobs=-1
        )

        df_feature_importance = pd.DataFrame({
            "importance": perm.importances_mean,
            "feature": X_test.columns,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        df_feature_importance["importance_norm"] = df_feature_importance["importance"] / df_feature_importance["importance"].sum()
        df_feature_importance["cum_importance"] = df_feature_importance["importance_norm"].cumsum()

        # top (5 - 8) inputs
        MAX_FEATURES = 15
        top_k_features = df_feature_importance.loc[df_feature_importance["cum_importance"] <= 0.95, "feature"].head(MAX_FEATURES).to_list()

        mlflow.log_param("top_k_features", ",".join(top_k_features))
        mlflow.log_param("top_k_features_count", len(top_k_features))

        # model performance
        model_complexity = get_model_complexity(best_model)
        inference_latency_ms = measure_inference_latency(best_model, X_test)

        mlflow.log_param("model_complexity", model_complexity)
        mlflow.log_param("inference_latency_ms", inference_latency_ms)

        # agent payload
        results.append({
            "model_name": model_name,
            "top_features": top_k_features,
            "model_complexity": model_complexity,
            "feature_importance_coverage_pct": 0.95,
            "top_features_count": len(top_k_features),
            "best_hyperparameters": search.best_params_,
            "inference_latency_ms": inference_latency_ms,
            "input_features": best_model.feature_names_in_,
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "input_features_count": best_model.n_features_in_,
            "feature_importance_method": "permutation_importance",
            "metrics": {
                "cv_auc_mean": round(search.best_score_, 4),
                "test_precision": round(test_precision, 4),
                "test_recall": round(test_recall, 4),
                "test_auc": round(test_auc, 4),
                "test_f1": round(test_f1, 4)
            },
        })    
