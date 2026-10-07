import os
import numpy as np
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# -----------------------------
# SETTINGS
# -----------------------------

DATA_DIR = "data"

classes = [
    "healthy",
    "early_blight",
    "late_blight",
    "leaf_mold"
]

IMAGE_SIZE = (224, 224)


# -----------------------------
# LOAD CNN
# -----------------------------

print("Loading MobileNetV2...")

model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

print("MobileNetV2 loaded!")


# -----------------------------
# EXTRACT FEATURES
# -----------------------------

features = []
labels = []

for class_index, class_name in enumerate(classes):

    folder = os.path.join(DATA_DIR, class_name)

    print(f"\nReading: {class_name}")

    for filename in os.listdir(folder):

        filepath = os.path.join(folder, filename)

        try:

            image = Image.open(filepath).convert("RGB")
            image = image.resize(IMAGE_SIZE)

            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)

            image_array = preprocess_input(image_array)

            feature = model.predict(image_array, verbose=0)

            features.append(feature[0])
            labels.append(class_index)

        except Exception as e:

            print("Skipping:", filename)
            print("Reason:", e)


# -----------------------------
# SAVE FEATURES
# -----------------------------

features = np.array(features)
labels = np.array(labels)

print("\nFinished!")
print("Feature shape:", features.shape)
print("Label shape:", labels.shape)

np.save("features.npy", features)
np.save("labels.npy", labels)

print("\nSaved:")
print("features.npy")
print("labels.npy")