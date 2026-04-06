import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATASET_PATH = r"C:\Akshaya\internship_infosys\sample_data"
classes = [
    "clear skin",
    "dark spots",
    "puffy eyes",
    "wrinkles"
]
plt.figure(figsize=(8, 8))
for i, cls in enumerate(classes):
    folder = os.path.join(DATASET_PATH, cls)
    images = os.listdir(folder)
    if len(images) == 0:
        print(f"No images found in {cls}")
        continue
    img_path = os.path.join(folder, images[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")
plt.tight_layout()
plt.show()
counts = {}
for cls in classes:
    folder = os.path.join(DATASET_PATH, cls)
    counts[cls] = len(os.listdir(folder))
df = pd.DataFrame.from_dict(counts, orient="index", columns=["Image Count"])
print("\nClass Distribution:")
print(df)
plt.figure(figsize=(6, 4))
sns.barplot(x=df.index, y=df["Image Count"])
plt.xticks(rotation=20)
plt.title("Class Distribution of Facial Skin Dataset")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()