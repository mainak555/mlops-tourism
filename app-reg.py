
# HF deployment script
from huggingface_hub import HfApi
from util import create_hf_repo
import os

hfApi = HfApi(token=os.getenv("HF_TOKEN"))
hf_repo = os.getenv('HF_REPO')

# Create HF App Space
create_hf_repo(hfApi, hf_repo, "space")

# Add HF App Secrets
hfApi.add_space_secret(
    key='HUGGING_FACE_API_KEY',
    value=os.getenv("HF_TOKEN"),
    repo_id=hf_repo,
)
hfApi.add_space_variable(
    key='MODEL_REPO',
    repo_id=hf_repo,
    value=hf_repo,
)
hfApi.add_space_variable(
    value=f"{os.getenv("MLFLOW_EXPERIMENT_NAME")}.joblib",
    key='MODEL_FILE',
    repo_id=hf_repo,
)

# Deploying App
hfApi.upload_folder(
    ignore_patterns=["*pycache**/", ".env"],
    folder_path="./ui-app",
    repo_type="space",
    repo_id=hf_repo
)
