from tensorflow.keras.models import load_model
import numpy as np
import csv
import os
import datetime
from preprocess import preprocess_image

# ==============================
# Load Model (only once)
# ==============================
model = load_model("dermalscan_model.h5")

classes = ['clear_skin', 'dark_spots', 'puffy_eyes', 'wrinkles']

# ==============================
# Logging Function (CSV)
# ==============================
def log_prediction(label, confidence):
    
    file_exists = os.path.isfile("predictions_log.csv")

    with open("predictions_log.csv", "a", newline="") as f:
        writer = csv.writer(f)

        # Write header only once
        if not file_exists:
            writer.writerow(["Timestamp", "Prediction", "Confidence"])

        writer.writerow([
            datetime.datetime.now(),
            label,
            f"{confidence:.2f}%"
        ])

# ==============================
# Prediction Function
# ==============================
def predict_image(img):
    
    # Preprocess image
    processed = preprocess_image(img)

    # Predict
    prediction = model.predict(processed)

    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index] * 100
    label = classes[class_index]

    # Log prediction
    log_prediction(label, confidence)

    return label, confidence, prediction[0]