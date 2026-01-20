
# helper functions will be used across pipelines
from huggingface_hub import create_repo, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

def create_hf_repo(hfApi: HfApi, repo_id: str, repo_type: str):
    try:
        hfApi.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' exists, Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new...")
        space_sdk = 'docker' if repo_type == 'space' else ''
        create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk=space_sdk, private=False)
