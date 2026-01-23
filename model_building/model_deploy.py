
from model_train import get_train_test_split, evaluate
from mlflow.tracking import MlflowClient
from model_config import MODEL_CONFIG
from huggingface_hub import HfApi
from pathlib import Path
import joblib
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

filter_string = (
    "tags.selected_for_deployment = 'true' AND "
    f"tags.pipeline_run_id = '{PIPELINE_RUN_ID}'"
)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=filter_string
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

## model serialization ##
bin_path = f"{LOCAL_ARTIFACT_DIR}/{MLFLOW_EXPERIMENT_NAME}.joblib"
bin_name =f"{MLFLOW_EXPERIMENT_NAME}.joblib"
version = f"v1.0.0-build.{PIPELINE_RUN_ID}"

joblib.dump(model_dict[model_name], bin_path)

## deploy to HF ##
hfApi.upload_file(
    repo_id=HF_REPO,
    repo_type="model",
    path_in_repo=bin_name,
    path_or_fileobj=bin_path,
    commit_message=f"mlflow_run_id: {run_id}",
) # type: ignore
hfApi.create_tag(
    repo_type="model",
    repo_id=HF_REPO,
    tag=version,
)

## tagging mlflow ##
# log HF pointer mlflow artifact
client.log_dict(run_id, {
    "model_type": model_name,
    "artifact": bin_name,
    "hf_repo": HF_REPO,
    "tag": version,
}, artifact_file="hf_model")

# register model in mlflow registry
model_uri = f"runs:/{run_id}/hf_model"
registered_model = client.create_registered_model(name=MLFLOW_EXPERIMENT_NAME)
model_version = client.create_model_version(
    description=f"HF model {version}",
    name=MLFLOW_EXPERIMENT_NAME,
    source=model_uri,
    run_id=run_id,
)

# attach HF meta to registered model version
client.set_model_version_tag(
    version=model_version.version,
    name=MLFLOW_EXPERIMENT_NAME,
    key="hf_repo",
    value=HF_REPO
)
client.set_model_version_tag(
    version=model_version.version,
    name=MLFLOW_EXPERIMENT_NAME,
    key="version",
    value=version
)
client.set_model_version_tag(
    version=model_version.version,
    name=MLFLOW_EXPERIMENT_NAME,
    key="mlflow_run_id",
    value=run_id
)

print(f"Registered model (version: {model_version.version}) linked to (mlflow run: {run_id}) and (HF tag {version})")
