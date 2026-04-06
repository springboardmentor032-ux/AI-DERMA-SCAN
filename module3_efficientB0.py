import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import numpy as np

# -----------------------------
# Paths & Parameters
# -----------------------------
data_dir = r"C:\Akshaya\internship_infosys\sample_folder"
results_dir = "module4_results"
os.makedirs(results_dir, exist_ok=True)

IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 30

# -----------------------------
# Haar Cascade
# -----------------------------
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)

# -----------------------------
# Helper function to crop faces
# -----------------------------
def detect_and_crop_faces(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80,80))
    face_imgs = []
    for (x,y,w,h) in faces:
        face = cv2.resize(img[y:y+h, x:x+w], IMG_SIZE)
        face_imgs.append(face)
    return face_imgs if len(face_imgs) > 0 else None

# -----------------------------
# Preprocess images for ImageDataGenerator
# -----------------------------
def preprocess_folder_with_haar(folder_path):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder_path))
    for label_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_folder):
            continue
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            faces = detect_and_crop_faces(img_path)
            if faces:
                for face in faces:
                    images.append(face)
                    labels.append(label_idx)
    images = np.array(images, dtype='float32')
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))
    return images, labels, class_names

# -----------------------------
# Load Data with Haar Cascade
# -----------------------------
X, y, class_names = preprocess_folder_with_haar(data_dir)
X = preprocess_input(X)
print("Data shape after Haar Cascade:", X.shape, y.shape)
print("Classes detected:", class_names)

# -----------------------------
# Train / Validation Split
# -----------------------------
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------
# Data Augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# -----------------------------
# Build EfficientNetB0 model
# -----------------------------
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# Training
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# -----------------------------
# Save Model
# -----------------------------
model.save(os.path.join(results_dir, "efficientnet_skin_model_haar.h5"))
print("\nModel saved successfully!")

# -----------------------------
# Save Training History
# -----------------------------
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(results_dir,"training_history.csv"), index=False)

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])
plt.savefig(os.path.join(results_dir,"accuracy_plot.png"))
plt.show()

# -----------------------------
# Plot Loss
# -----------------------------
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train","Validation"])
plt.savefig(os.path.join(results_dir,"loss_plot.png"))
plt.show()

print("\nAll results saved in:", results_dir)