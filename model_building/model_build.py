
from model_train import get_train_test_split, evaluate
from model_config import MODEL_CONFIG
import uuid 
import os 

PIPELINE_RUN_ID = os.getenv("GITHUB_RUN_ID")
if not PIPELINE_RUN_ID:
    PIPELINE_RUN_ID = f"local_{uuid.uuid4().hex[:12]}"

X_train, y_train, X_test, y_test = get_train_test_split()
evaluate(PIPELINE_RUN_ID, "model_build", MODEL_CONFIG, X_train, y_train, X_test, y_test)
