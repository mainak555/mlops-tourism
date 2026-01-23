
from model_train import get_train_test_split, evaluate
from mlflow.tracking import MlflowClient
from model_config import MODEL_CONFIG
from huggingface_hub import HfApi
import mlflow
import json
import os

HF_REPO = os.getenv("HF_REPO")
hfApi = HfApi(token=os.getenv("HF_TOKEN"))
PIPELINE_RUN_ID = os.getenv("GITHUB_RUN_ID")

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
if not MLFLOW_EXPERIMENT_NAME:
    raise RuntimeError("MLFLOW_EXPERIMENT_NAME not found")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI not found")

## get selected model from experiment ##
client = MlflowClient()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.selected_for_deployment = 'true'"
)
if len(runs) != 1:
    raise RuntimeError(f"Expected exactly one selected run, found {len(runs)}")

# selected model details
tags = runs[0].data.tags
run_id = runs[0].info.run_id
model_name = tags.get("model_name")
TOP_k_FEATURE_PATH = "feature_analysis/top_k_features.json"
LOCAL_ARTIFACT_DIR = "./artifacts"

os.makedirs(LOCAL_ARTIFACT_DIR, exist_ok=True)
local_path = client.download_artifacts(
    run_id=run_id,
    path=TOP_k_FEATURE_PATH,
    dst_path=LOCAL_ARTIFACT_DIR
)

with open(local_path, "r") as f:
    top_k_artifact = json.load(f)

top_k_features = [
    f["name"] for f in top_k_artifact["features"]
]

## get data & train final model ##
X_train, y_train, X_test, y_test = get_train_test_split()
model_dict = evaluate(f"{PIPELINE_RUN_ID}_final", "model_deploy", {
    model_name: MODEL_CONFIG[model_name] #only selected model
}, X_train[top_k_features], y_train, X_test[top_k_features], y_test)

print('done upto here')
