
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
import mlflow
import uuid
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
def extract_model_structure(model):
    """
    Extracts structural indicators from the final estimator
    """
    # unwrap pipeline if needed
    if hasattr(model, "steps"):  # sklearn Pipeline
        estimator = model.steps[-1][1]
    else:
        estimator = model

    structure = {
        "model_family": estimator.__class__.__name__,
        "n_estimators": None,
        "total_tree_nodes": None,
        "n_coefficients": None,
    }

    if hasattr(estimator, "estimators_"):  # Tree ensembles
        structure["n_estimators"] = len(estimator.estimators_)
        structure["total_tree_nodes"] = sum(
            est.tree_.node_count
            for est in estimator.estimators_
            if hasattr(est, "tree_")
        )

    elif hasattr(estimator, "tree_"):  # Single tree
        structure["total_tree_nodes"] = estimator.tree_.node_count

    elif hasattr(estimator, "coef_"):  # Linear models
        structure["n_coefficients"] = estimator.coef_.size

    return structure

def classify_model_complexity(structure):
    """
    Converts structural indicators into complexity classes
    """

    model_family = structure["model_family"]
    total_nodes = structure.get("total_tree_nodes")
    n_estimators = structure.get("n_estimators") or 1

    # Tree-based ensembles
    if model_family in ["RandomForestClassifier", "BaggingClassifier", "AdaBoostClassifier"]:
        if n_estimators <= 100 and total_nodes and total_nodes <= 50_000:
            return "medium"
        return "high"

    # Single trees
    if model_family == "DecisionTreeClassifier":
        if total_nodes and total_nodes <= 5_000:
            return "low"
        return "medium"

    return "unknown"


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

PIPELINE_RUN_ID = os.getenv("GITHUB_RUN_ID")
GIT_SHA = os.getenv("GITHUB_SHA")
if not PIPELINE_RUN_ID:
    PIPELINE_RUN_ID = f"local_{uuid.uuid4().hex[:12]}"

# model eval
EXPERIMENT_NAME = "wellness-purchase-propensity"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

for model_name, cfg in MODEL_CONFIG.items():
    run_name = f"{model_name}_rs_{PIPELINE_RUN_ID}"
    print(f"mlFlow Run: {run_name}")
    with mlflow.start_run(run_name=run_name):     
        mlflow.set_tags({
            "model_name": model_name,
            "git_commit_sha": GIT_SHA,
            "pipeline_job": "model_train",
            "pipeline_run_id": PIPELINE_RUN_ID,
            "run_at": f"{time.strftime('%Y-%m-%d_%H:%M')}"
        })

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

        # log hyper-parameters
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

        # top k features
        MAX_FEATURES = 15
        COVERAGE_THRESHOLD = 0.95
        top_k_df = df_feature_importance.loc[df_feature_importance["cum_importance"] <= COVERAGE_THRESHOLD].head(MAX_FEATURES)
        top_k_features_artifact = {
            "top_k": len(top_k_df),
            "coverage_threshold": COVERAGE_THRESHOLD,
            "coverage_pct": round(top_k_df["importance_norm"].sum() * 100, 2),
            "features": [
                {
                    "name": row["feature"],
                    "importance": round(row["importance_norm"], 6),
                    "cumulative_importance": round(row["cum_importance"], 6)
                }
                for _, row in top_k_df.iterrows()
            ]
        }

        mlflow.log_dict(
            top_k_features_artifact,
            artifact_file="feature_analysis/top_k_features.json"
        )
        mlflow.set_tag("feature_importance_method", "permutation_importance")
        mlflow.log_metric("feature_importance_coverage_pct", COVERAGE_THRESHOLD)
        mlflow.log_metric("top_k_features_count", top_k_features_artifact["top_k"])

        # model performance
        inference_latency_ms = measure_inference_latency(best_model, X_test)
        mlflow.log_metric("inference_latency_ms", inference_latency_ms)           

        # model complexity
        structure = extract_model_structure(best_model)
        model_complexity = classify_model_complexity(structure)
        mlflow.log_param("model_complexity", model_complexity)

        # feature details
        raw_features = X_train.columns.to_list()
        raw_features_artifact = {
            "total_raw_features": len(raw_features),
            "features": raw_features
        }

        mlflow.log_dict(
            raw_features_artifact,
            artifact_file="feature_schema/raw_features.json"
        )
        mlflow.set_tag("total_raw_features", len(raw_features))

        transformed_features = best_model.named_steps["columntransformer"].get_feature_names_out().tolist()
        transformed_features_artifact = {
            "total_transformed_features": len(transformed_features),
            "features": transformed_features
        }

        mlflow.log_dict(
            transformed_features_artifact,
            artifact_file="feature_schema/transformed_features.json"
        )
        mlflow.log_metric("total_transformed_features", len(transformed_features))   
