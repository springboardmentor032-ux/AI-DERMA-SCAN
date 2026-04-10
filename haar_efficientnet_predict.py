import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==============================
# Load Model
# ==============================
model = load_model("dermalscan_model.h5")

classes = ['clear_skin', 'dark_spots', 'puffy_eyes', 'wrinkles']

# ==============================
# Load Haar Cascade
# ==============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==============================
# Image Path
# ==============================
img_path = "dataset/dark_spots/6.jpg"   # change if needed

img = cv2.imread(img_path)

# ==============================
# Safety Check
# ==============================
if img is None:
    print("❌ ERROR: Image not found:", img_path)
    exit()

# ==============================
# Convert to grayscale
# ==============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ==============================
# Face Detection
# ==============================
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=4,
    minSize=(50, 50)
)

# ==============================
# OPTION 3: Smart fallback
# ==============================
if len(faces) == 0:
    print("⚠ No face detected → using full image")

    face = cv2.resize(img, (224, 224))
    x, y, w, h = 0, 0, img.shape[1], img.shape[0]

else:
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))

# ==============================
# Preprocess
# ==============================
face = face.astype("float32") / 255.0
face = np.expand_dims(face, axis=0)

# ==============================
# Prediction
# ==============================
prediction = model.predict(face)

class_index = np.argmax(prediction)
confidence = prediction[0][class_index] * 100

label = f"{classes[class_index]} ({confidence:.2f}%)"

print("Predicted Skin Condition:", label)

# ==============================
# Draw Box + Label
# ==============================
cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.putText(
    img,
    label,
    (x, y-10 if y > 20 else y + 20),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 255, 0),
    2
)

# ==============================
# Show Output
# ==============================
cv2.imshow("DermalScan Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()