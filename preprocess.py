import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(50, 50)
    )

    # Fallback if no face
    if len(faces) == 0:
        face = cv2.resize(img, (224, 224))
    else:
        (x, y, w, h) = faces[0]
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))

    # Normalize
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    return face