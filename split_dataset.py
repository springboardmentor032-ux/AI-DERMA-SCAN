import os
import shutil
import random

source_dir = "dataset"
base_dir = "data"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for cls in os.listdir(source_dir):
    cls_path = os.path.join(source_dir, cls)

    images = os.listdir(cls_path)
    random.shuffle(images)

    train_split = int(len(images) * train_ratio)
    val_split = int(len(images) * val_ratio)

    train_imgs = images[:train_split]
    val_imgs = images[train_split:train_split + val_split]
    test_imgs = images[train_split + val_split:]

    for folder in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, folder, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img),
                    os.path.join(base_dir, "train", cls, img))

    for img in val_imgs:
        shutil.copy(os.path.join(cls_path, img),
                    os.path.join(base_dir, "val", cls, img))

    for img in test_imgs:
        shutil.copy(os.path.join(cls_path, img),
                    os.path.join(base_dir, "test", cls, img))

print("Dataset split completed!")