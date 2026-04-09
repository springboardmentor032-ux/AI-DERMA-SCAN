# frontend/utils.py

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224

# -----------------------------
# LOAD MODEL (CACHED)
# -----------------------------
@st.cache_resource
def load_my_model():
    return load_model("../dermalscan_model.keras")

model = load_my_model()

# -----------------------------
# CLASSES
# -----------------------------
classes = ["clear_skin", "dark_spots", "puffy_eyes", "wrinkles"]

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []
    final_probs = np.zeros(len(classes))

    if len(faces) == 0:
        return [], final_probs

    for (x, y, w, h) in faces:

        face = image[y:y+h, x:x+w]

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)

        prediction = model.predict(face, verbose=0)

        class_id = np.argmax(prediction)
        confidence = prediction[0][class_id]

        label = f"{classes[class_id]} ({confidence*100:.2f}%)"

        results.append((x, y, w, h, label))

        final_probs = prediction[0]

    return results, final_probs