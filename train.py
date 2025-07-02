import kagglehub
import os
import shutil
import tensorflow as tf

TARGET_DIR = "data"

if os.path.exists(TARGET_DIR):
    print("Dataset already present in 'data/' directory.")
else:
    print("Downloading dataset with kagglehub...")
    path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    source_path = os.path.join(path, TARGET_DIR)

    print(f"Copying from cache to {TARGET_DIR}...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    shutil.copytree(source_path, TARGET_DIR)

    print("Dataset copied to 'data/' directory.")


print("Beginning training...")
