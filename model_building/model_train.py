
from sklearn.metrics import (
    precision_score,
    roc_auc_score,
    recall_score,
    f1_score, 
)

from util import num_features_selector, cat_features_selector
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from model_config import MODEL_CONFIG
from huggingface_hub import HfApi
import pandas as pd
import mlflow
import time
import sys
import os

HF_REPO = os.getenv("HF_REPO")
hfApi = HfApi(token=os.getenv("HF_TOKEN"))
print(f"Loading Train/Test files from HF Dataset: {HF_REPO}")

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

# Column Processors
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

# model eval
results = []
EXPERIMENT_NAME = "wellness-purchase-propensity"
mlflow.set_experiment(EXPERIMENT_NAME)

for model_name, cfg in MODEL_CONFIG.items():
    run_name = f"{model_name}_rs_{time.strftime('%Y%m%d_%H%M')}"
    with mlflow.start_run(run_name=run_name):      
        pipeline = Pipeline([
            ("preprocess", col_processor),
            (model_name.lower(), cfg["estimator"])
        ])

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

        # Predictions
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Metrics
        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("cv_auc_mean", search.best_score_)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_f1", test_f1)

        # Log hyperparameters
        mlflow.log_params(search.best_params_)

        # agent payload
        results.append({
            "num_features": best_model.named_steps["preprocess"].transform(X_train).shape[1],
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "best_hyperparameters": search.best_params_,
            "model_name": model_name,
            "metrics": {
                "cv_auc_mean": round(search.best_score_, 4),
                "test_precision": round(test_precision, 4),
                "test_recall": round(test_recall, 4),
                "test_auc": round(test_auc, 4),
                "test_f1": round(test_f1, 4)
            },
        })

    print(results)
