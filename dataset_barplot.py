import os
import matplotlib.pyplot as plt

dataset_path = "dataset"

class_names = []
image_counts = []

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):
        count = len([
            file for file in os.listdir(class_path)
            if file.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        class_names.append(class_name)
        image_counts.append(count)

# Bar Plot
plt.figure(figsize=(8,6))
plt.bar(class_names, image_counts)
plt.title("Image Distribution Per Class")
plt.xlabel("Class Names")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()