import os
from huggingface_hub import snapshot_download

# Target repository and subfolder
REPO_ID = "MasoudJTehrani/PCLA"
SUBFOLDER = "transfuserv6_pretrained"
LOCAL_DIR = "./transfuserv6_pretrained"  # Output directory on your local machine

print(f"Downloading '{SUBFOLDER}' from dataset repo '{REPO_ID}'...")

downloaded_path = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=f"{SUBFOLDER}/*",
    local_dir=LOCAL_DIR,
)

print(f"Download complete! Files saved to: {os.path.abspath(LOCAL_DIR)}")