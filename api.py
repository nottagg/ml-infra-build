from fastapi import FastAPI
import numpy as np
from fastapi import File, UploadFile
from PIL import Image
import io
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model("/app/model.h5")
class_names = model.class_names if hasattr(model, 'class_names') else None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Plant Disease Detector API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    preds = model.predict(img_array)
    pred_class = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    # Optionally map to class name if available
    label = class_names[pred_class] if class_names else str(pred_class)

    return {"class": label, "confidence": confidence}