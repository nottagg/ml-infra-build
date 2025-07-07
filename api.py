import numpy as np
import tensorflow as tf
from train import run_training_script
from fastapi import File, UploadFile
from pydantic import BaseModel
from fastapi import BackgroundTasks, FastAPI


app = FastAPI()
class TrainRequest(BaseModel):
    kaggle_url: str



@app.get("/")
def read_root():
    return {"message": "Welcome to the Plant Disease Detector API"}

@app.post("/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training_script, request.kaggle_url)
    return {"status": "started", "message": f"Training started for {request.kaggle_url}"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return {"message": "Prediction endpoint is not implemented yet"}
