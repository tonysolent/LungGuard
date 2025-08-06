# checkpoint_loader.py

import gdown
import zipfile
import os

def download_and_extract_checkpoints(gdrive_id: str, extract_to: str = "checkpoints"):
    """
    Downloads and extracts a zipped model checkpoint folder from Google Drive.

    Parameters:
    - gdrive_id (str): Google Drive file ID (e.g. from shared URL)
    - extract_to (str): Target folder where the zip will be extracted
    """

    zip_filename = "checkpoints.zip"

    if not os.path.exists(extract_to):
        print("🔽 Downloading model checkpoint archive from Google Drive...")
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, zip_filename, quiet=False)

        print("📦 Extracting checkpoint files...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        os.remove(zip_filename)
        print(f"✅ All models extracted to `{extract_to}/`")

    else:
        print(f"✅ Checkpoint directory `{extract_to}/` already exists. Skipping download.")
