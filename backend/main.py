from fastapi import FastAPI
from fastapi import UploadFile, File

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "type": file.content_type
    }