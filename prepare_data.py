import os
import shutil
import random

SOURCE_DIR = "data"
OUTPUT_DIR = "splits"

classes = [
    "bacterial_spot",
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot",
    "spider_mites",
    "target_spot",
    "yellow_leaf_curl_virus"
]

random.seed(42)

TRAIN_RATIO = 0.80

for class_name in classes:

    source_folder = os.path.join(
        SOURCE_DIR,
        class_name
    )

    train_folder = os.path.join(
        OUTPUT_DIR,
        "train",
        class_name
    )

    test_folder = os.path.join(
        OUTPUT_DIR,
        "test",
        class_name
    )

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    images = []

    for filename in os.listdir(source_folder):

        filepath = os.path.join(
            source_folder,
            filename
        )

        if os.path.isfile(filepath):
            images.append(filename)

    random.shuffle(images)

    train_count = int(
        len(images) * TRAIN_RATIO
    )

    train_images = images[:train_count]
    test_images = images[train_count:]

    print("\nClass:", class_name)
    print("Total:", len(images))
    print("Training:", len(train_images))
    print("Testing:", len(test_images))

    for filename in train_images:

        source = os.path.join(
            source_folder,
            filename
        )

        destination = os.path.join(
            train_folder,
            filename
        )

        shutil.copy2(source, destination)

    for filename in test_images:

        source = os.path.join(
            source_folder,
            filename
        )

        destination = os.path.join(
            test_folder,
            filename
        )

        shutil.copy2(source, destination)


print("\n==============================")
print("DATA SPLIT COMPLETE")
print("==============================")