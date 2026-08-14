"""
export_hf_upload.py  —  push the staged folder to the Hugging Face repo.

Expects HF_TOKEN in the environment. Uploads README.md and modeling.py first, so
the page reads correctly while the 12 GB of weights are still going up.

Run:  python training_scripts/others/export_hf_upload.py
"""

import os
from pathlib import Path

from huggingface_hub import HfApi

OUT = Path(r"D:\chexpert-bench-hf")
REPO = "mamounyosef/chest-xray-bench"

api = HfApi(token=os.environ["HF_TOKEN"])

for small in ("README.md", "modeling.py", "models.json"):
    api.upload_file(path_or_fileobj=OUT / small, path_in_repo=small,
                    repo_id=REPO, repo_type="model",
                    commit_message=f"Add {small}")
    print(f"uploaded {small}")

api.upload_folder(
    folder_path=OUT,
    repo_id=REPO,
    repo_type="model",
    commit_message="Add 43 CheXpert models: weights, configs and thresholds",
    ignore_patterns=["*.tmp", ".git*"],
)
print("done")
