
from model_building.model_train import get_train_test_split
from agents.schema_generator_agent.run import run_schema_generator
from mlflow.tracking import MlflowClient
from huggingface_hub import HfApi
import asyncio
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
    #"tags.deployed = 'true' AND "
    f"tags.pipeline_run_id = '{PIPELINE_RUN_ID}'"
)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=filter_string
)
if len(runs) != 1:
    raise RuntimeError(f"Expected exactly one selected run, found {len(runs)}")

# deployed model details
tags = runs[0].data.tags
run_id = runs[0].info.run_id
model_name = tags.get("model_name")
print(f"deployed mlflow run: {runs[0].info.run_name}")

# get deployed model inputs
FEATURE_PATH = "feature_schema/raw_features.json"
LOCAL_ARTIFACT_DIR = f"./artifacts_{run_id}"

os.makedirs(LOCAL_ARTIFACT_DIR, exist_ok=True)
local_path = client.download_artifacts(
    dst_path=LOCAL_ARTIFACT_DIR,
    path=FEATURE_PATH,
    run_id=run_id,
)

with open(local_path, "r") as f:
    feature_schema = json.load(f)

df, _, _, _ = get_train_test_split()

async def generate_schema():
    result = await run_schema_generator(df, feature_schema["features"])
    print(result)

asyncio.run(generate_schema())
