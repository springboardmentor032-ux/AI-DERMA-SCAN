import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Dataset path
data_dir = "dataset"

# Pick random class
classes = os.listdir(data_dir)
random_class = random.choice(classes)

class_path = os.path.join(data_dir, random_class)
image_name = random.choice(os.listdir(class_path))
image_path = os.path.join(class_path, image_name)

print("Selected Image:", image_path)

# Load original image
original_img = load_img(image_path, target_size=(224, 224))
original_array = img_to_array(original_img)
original_array = np.expand_dims(original_array, axis=0)

# Define augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)

# Generate one augmented image
augmented_iterator = datagen.flow(original_array, batch_size=1)
augmented_image = next(augmented_iterator)[0]

# Plot comparison
plt.figure(figsize=(8,4))

# Original
plt.subplot(1,2,1)
plt.imshow(original_array[0] / 255.0)
plt.title("Original Image")
plt.axis("off")

# Augmented
plt.subplot(1,2,2)
plt.imshow(augmented_image / 255.0)
plt.title("Augmented Image")
plt.axis("off")

plt.tight_layout()
plt.show()