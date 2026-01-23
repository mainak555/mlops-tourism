
from huggingface_hub import HfApi
from util import create_hf_repo
import os

hfApi = HfApi(token=os.getenv("HF_TOKEN"))
hf_repo = os.getenv('HF_REPO')

# Create HF Model Space
create_hf_repo(hfApi, hf_repo, "model")
