import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================
# Load Trained Model
# ==========================

model = load_model("best_model.keras")

classes = ['clear_skin', 'dark_spots', 'puffy_eyes', 'wrinkles']

IMG_SIZE = 224

# ==========================
# Load Haar Cascade
# ==========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ==========================
# Start Webcam
# ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face_array = np.array(face_resized, dtype=np.float32)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = preprocess_input(face_array)

        prediction = model.predict(face_array, verbose=0)

        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        label = f"{classes[class_index]}: {confidence:.2f}%"

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Put text
        cv2.putText(frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

    cv2.imshow("DermalScan - Real Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()