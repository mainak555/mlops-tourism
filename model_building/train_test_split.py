
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import pandas as pd
import sys
import os

# loading dataset from Hugging Face data space
hfApi = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = f"hf://datasets/{os.getenv("HF_REPO")}/{os.getenv("CSV_DATA_FILE")}"

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
        print(f"{os.getenv("CSV_DATA_FILE")}.csv missing @HF Dataset")
        sys.exit(1)
except Exception as e:
    print(f"Error Checking Path: {DATASET_PATH} | Err: {e}")
    sys.exit(1)

print("\033[1mRows: {}\033[0m & \033[1mColumns: {}\033[0m".format(
            df.shape[0], df.shape[1]
    ))

# removing unnecessary column(s) & train-test split
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

X = df.drop(columns=['CustomerID', 'ProdTaken'])
y = df['ProdTaken']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40, stratify=y
)

# saving train/test split locally
X_train.to_csv("X_train.csv",index=False)
X_test.to_csv("X_test.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_test.to_csv("y_test.csv",index=False)

# uploading train and test datasets back to the Hugging Face data space
for file in ["X_train.csv","X_test.csv","y_train.csv","y_test.csv"]:
    hfApi.upload_file(
        repo_id=os.getenv('HF_REPO'),
        path_or_fileobj=file,
        repo_type="dataset",
        path_in_repo=file,
    )
