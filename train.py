import kagglehub
import os
import shutil
import tensorflow as tf
from azure.storage.blob import BlobServiceClient

TARGET_DIR = "dataset"


print("Checking if model is already trained...")
if os.path.exists("/app/model.h5"):
    print("Model already exists at /app/model.h5. Skipping training.")
    exit(0)

print("Fetching dataset from Kaggle...")
if os.path.exists(TARGET_DIR) and len(os.listdir(TARGET_DIR)) > 0:
    print(f"Dataset already present in {TARGET_DIR} directory.")
else:
    print("Downloading dataset with kagglehub...")
    path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

    print(f"Copying from cache to {TARGET_DIR}...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    shutil.copytree(path, TARGET_DIR, dirs_exist_ok=True)
    print(f"Dataset copied to {TARGET_DIR} .")


print("Checking dataset structure...")
BATCH_SIZE = 32 
IMG_SIZE = (224, 224)
DATASET_ROOT = None
for root, dirs, _ in os.walk(TARGET_DIR):
    if "train" in dirs and "valid" in dirs:
        print(f"Found train and valid directories in {root}")
        DATASET_ROOT = root
        break
if DATASET_ROOT is None:
    raise ValueError("Train and valid directories not found in the dataset structure.")

train_dir = os.path.join(DATASET_ROOT, "train")
validation_dir = os.path.join(DATASET_ROOT, "valid")

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
    shuffle=True,
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
    exit(45)


print("Saving model to /app/model.h5 ...")
model.save("/app/model.h5")
print("Model saved.")