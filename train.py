import kagglehub
import os
import shutil
import tensorflow as tf
from azure.storage.blob import BlobServiceClient
import sys
import re
import argparse
import subprocess

TARGET_DIR = "dataset"

def run_training_script(kaggle_url):
    process = subprocess.Popen(
        ["python","-u", "/app/train.py", "--kaggle_url", kaggle_url],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    with open("/app/train.log", "w") as log_file:
        if process.stdout is None:
            print("Error: stdout is None")
            return

        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            log_file.flush()
    
    process.wait()

def extract_kaggle_dataset_id(url_or_id):
    # If it's a full URL, extract the part after 'datasets/'
    match = re.search(r"datasets/([^/?#]+/[^/?#]+)", url_or_id)
    if match:
        return match.group(1)
    return url_or_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle_url", type=str, required=True, help="Kaggle dataset URL or ID")
    args = parser.parse_args()
    kaggle_url = args.kaggle_url

    print("Checking if model is already trained...")
    if os.path.exists("/app/model.h5"):
        print("Model already exists at /app/model.h5. Skipping training.")
        exit(0)

    print("Fetching dataset from Kaggle...")
    if os.path.exists(TARGET_DIR) and len(os.listdir(TARGET_DIR)) > 0:
        print(f"Dataset already present in {TARGET_DIR} directory.")
    else:
        print("Downloading dataset with kagglehub...")
        kaggle_url = extract_kaggle_dataset_id(kaggle_url)
        path = kagglehub.dataset_download(kaggle_url)

        print(f"Copying from cache to {TARGET_DIR}...")
        os.makedirs(TARGET_DIR, exist_ok=True)
        shutil.copytree(path, TARGET_DIR, dirs_exist_ok=True)
        print(f"Dataset copied to {TARGET_DIR} .")


    print("Checking dataset structure...")
    train_dir = None
    validation_dir = None
    DATASET_ROOT = None
    BATCH_SIZE = 16 
    IMG_SIZE = (224, 224)
    DATASET_ROOT = None
    for root, dirs, _ in os.walk(TARGET_DIR):
        DATASET_ROOT = root
        if "train" in dirs and "valid" in dirs:
            print(f"Found train and valid directories in {root}")
            train_dir = os.path.join(root, "train")
            validation_dir = os.path.join(root, "valid")
            break

    if train_dir and validation_dir:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        validation_ds = tf.keras.utils.image_dataset_from_directory(
            validation_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
        )
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_ROOT,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        validation_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_ROOT,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
        )

    num_classes = len(train_ds.class_names)
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    print("Normalizing datasets...")
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (
        train_ds
        .shuffle(100)
        .prefetch(buffer_size=AUTOTUNE)
    )

    validation_ds = (
        validation_ds
        .map(lambda x, y: (normalization_layer(x), y))
        .prefetch(buffer_size=AUTOTUNE)
    )

    print("Creating model...")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])


    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    print("Training model...")
    try: 
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=10,
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        exit(1)


    print("Saving model to /app/model.h5 ...")
    model.save("/app/model.h5")
    print("Model saved.")