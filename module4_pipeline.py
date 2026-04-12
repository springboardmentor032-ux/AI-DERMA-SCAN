import cv2
import numpy as np
import os
import random
from tensorflow.keras.models import load_model


model = load_model('skin_aging_model.h5')
categories = ['clear skin', 'dark spots', 'puffy eyes', 'wrinkles']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

dataset_path = r'C:\Users\User2\Desktop\NIVEDITA\dataset\wrinkles'
all_images = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
image_path = os.path.join(dataset_path, random.choice(all_images))

image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found!")
    exit()


image = cv2.resize(image, (900, 700)) 
kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
display_img = cv2.filter2D(image, -1, kernel)


gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

# 5. PIPELINE PROCESSING
if len(faces) > 0:
    for (x, y, w, h) in faces:
        # Crop the face for the AI
        roi = display_img[y:y+h, x:x+w]
        
        # AI Prediction
        img_ready = cv2.resize(roi, (224, 224))
        img_ready = img_ready.astype('float32') / 255.0
        img_ready = np.expand_dims(img_ready, axis=0)

        prediction = model.predict(img_ready)
        result_index = np.argmax(prediction)
        confidence = prediction[0][result_index] * 100
        
        # 6. DRAW PROFESSIONAL LABELS
        label = f"{categories[result_index].upper()}: {confidence:.1f}%"
        
        # Draw a thick Green box and a background for the text
        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.rectangle(display_img, (x, y-40), (x+w, y), (0, 255, 0), -1) # Green background for text
        cv2.putText(display_img, label, (x+5, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
else:
    print("No clear face found. Please try an image with better lighting.")

# 7. SHOW HD RESULT
cv2.imshow('Module 4: HD Prediction Result', display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()