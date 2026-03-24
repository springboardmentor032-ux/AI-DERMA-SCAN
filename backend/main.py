from fastapi import FastAPI
from fastapi import UploadFile, File
import os
from model import model
import numpy as np
from preprocess import preprocess_image

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    # create uploads folder if not exists
    os.makedirs("../uploads", exist_ok=True)

    file_path = f"../uploads/{file.filename}"

    # save file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # preprocess
    processed = preprocess_image(file_path)

    # prediction
    preds = model.predict(processed)

    # get top class
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "class": class_index,
        "confidence": confidence
    }