
from agents.model_selector_agent.run import run_model_selector
from agents.agent_util import load_schema, validate_schema
from mlflow.tracking import MlflowClient
from datetime import datetime
from pprint import pprint
import asyncio
import mlflow
import os

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
if not MLFLOW_EXPERIMENT_NAME:
    raise RuntimeError("MLFLOW_EXPERIMENT_NAME not found")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI not found")

PIPELINE_RUN_ID = os.getenv("GITHUB_RUN_ID")
if not PIPELINE_RUN_ID:
    raise RuntimeError("PIPELINE_RUN_ID not found")

# get experiment results
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()
experiment = client.get_experiment_by_name("wellness-purchase-propensity")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.pipeline_run_id = '{PIPELINE_RUN_ID}'"
)
if not runs:
    raise RuntimeError(f"No MLflow runs found for pipeline_run_id={PIPELINE_RUN_ID}")

## agent payload ##
agent_payload = {
    "objective": {
        "business_goal": "Rank customers by likelihood of purchasing Wellness Tourism Package",
        "decision_type": "ranking",
        "primary_metric": "test_auc",
        "constraints": {
            "min_recall": 0.65,
            "min_precision": 0.45
        }
    },
    "pipeline_context": {
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
        "pipeline_run_id": PIPELINE_RUN_ID,
    },
    "candidates": []
}

for run in runs:
    data = run.data
    tags = data.tags
    agent_payload['candidates'].append({
        "model_name": tags.get("model_name"),
        "mlflow_run_id": run.info.run_id,
        "metrics": {
            "test_f1": data.metrics.get("test_f1"),
            "test_auc": data.metrics.get("test_auc"),
            "test_recall": data.metrics.get("test_recall"),
            "test_precision": data.metrics.get("test_precision"),
        },
        "complexity": {
            "model_complexity": tags.get("model_complexity"),
            "inference_latency_ms": data.metrics.get("inference_latency_ms"),
            "total_transformed_features": data.metrics.get("total_transformed_features")
        },
        "feature_profile": {
            "top_k_features_count": data.metrics.get("top_k_features_count"),
            "feature_importance_method": tags.get("feature_importance_method"),
            "coverage_pct": data.metrics.get("feature_importance_coverage_pct"),
        },
        "artifacts": {
            "top_k_features": "feature_analysis/top_k_features.json",
            "raw_features": "feature_schema/raw_features.json",
            "transformed_features": "feature_schema/transformed_features.json"
        }
    })

## calling agent ##
SCHEMA_PATH = "agents/model_selector_agent/select_model/config.json"

async def get_selection():
    try:
        decision = await run_model_selector(agent_payload)
        pprint(decision)

        schema = load_schema(SCHEMA_PATH)
        validate_schema(decision, schema)

        ## tagging selected model ##
        client.set_tag(decision["mlflow_run_id"], "selected_for_deployment", "true")
        client.set_tag(decision["mlflow_run_id"], "selection_timestamp", datetime.now().isoformat())
        client.set_tag(decision["mlflow_run_id"], "selection_justification", decision["justification"])
    except Exception as e:
        print(f"Agent failed: {e}")

asyncio.run(get_selection())
