from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from model import model
import numpy as np
from preprocess import preprocess_image
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve your frontend static files
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")

CLASS_NAMES = ["Clear Skin", "Dark Spots", "Puffy Eyes", "Wrinkles"]

# ✅ Serve index.html at root
@app.get("/")
def index():
    return FileResponse("../frontend/templates/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    os.makedirs("../uploads", exist_ok=True)
    file_path = f"../uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    processed = preprocess_image(file_path)
    preds = model.predict(processed)[0]
    class_index = int(np.argmax(preds))
    confidence = float(preds[class_index]) * 100
    return {
        "class": CLASS_NAMES[class_index],
        "confidence": round(confidence, 2),
        "all_probs": preds.tolist()
    }