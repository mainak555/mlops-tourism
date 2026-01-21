
from huggingface_hub import HfApi
from util import create_hf_repo
import os

hfApi = HfApi(token=os.getenv("HF_TOKEN"))
hf_repo = os.getenv('HF_REPO')

# Create HF Data Space
create_hf_repo(hfApi, hf_repo, "dataset")

# Upload tourism.csv to HF Data Space
hfApi.upload_folder(
    allow_patterns=["tourism.csv"],
    folder_path="./data",
    repo_type="dataset",
    repo_id=hf_repo,
)
