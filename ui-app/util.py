
def get_artifacts():
    from huggingface_hub import hf_hub_download, HfApi
    import joblib
    import json
    import os

    ARTIFACT_PATH = "artifacts"
    SCHEMA_FILE = "input_schema.json"
    MODEL_REPO = os.getenv("MODEL_REPO") or os.getenv("HF_REPO") #local only
    MODEL_FILE = os.getenv("MODEL_FILE") or f"{os.getenv("MLFLOW_EXPERIMENT_NAME")}.joblib" #local only
    os.makedirs("artifacts", exist_ok=True)

    hfApi = HfApi()
    model_path = hfApi.hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=ARTIFACT_PATH
    )

    schema_path = hfApi.hf_hub_download(
        repo_id=MODEL_REPO,
        filename=SCHEMA_FILE,
        local_dir=ARTIFACT_PATH
    )

    model = joblib.load(model_path)
    with open(schema_path, "r") as f:
        schema = json.load(f)

    return model, schema
