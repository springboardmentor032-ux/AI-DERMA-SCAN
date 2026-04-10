import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import random
import os

# Dataset path
data_dir = "dataset"

# Pick random class
classes = os.listdir(data_dir)
random_class = random.choice(classes)

class_path = os.path.join(data_dir, random_class)
image_name = random.choice(os.listdir(class_path))

image_path = os.path.join(class_path, image_name)

print("Selected Image:", image_path)

# Load image
img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Define augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
)

# Generate augmented images
augmented_images = [img_array[0]]

for batch in datagen.flow(img_array, batch_size=1):
    augmented_images.append(batch[0])
    if len(augmented_images) == 6:
        break

# Plot comparison
plt.figure(figsize=(12, 8))

titles = ["Original", "Augmented 1", "Augmented 2", "Augmented 3", "Augmented 4", "Augmented 5"]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(augmented_images[i] / 255.0)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()