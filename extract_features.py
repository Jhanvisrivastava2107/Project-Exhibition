import os
import numpy as np
import tensorflow as tf
from PIL import Image


# -------------------------------
# Settings
# -------------------------------

DATA_DIR = "data"
IMAGE_SIZE = (224, 224)


# -------------------------------
# Load MobileNetV2
# -------------------------------

model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

model.trainable = False


# -------------------------------
# Store features and labels
# -------------------------------

features = []
labels = []

class_names = sorted(os.listdir(DATA_DIR))

print("Classes:", class_names)


# -------------------------------
# Process images
# -------------------------------

for label, class_name in enumerate(class_names):

    class_path = os.path.join(DATA_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    for filename in os.listdir(class_path):

        image_path = os.path.join(
            class_path,
            filename
        )

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(IMAGE_SIZE)

            image_array = np.array(image)

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                image_array
            )

            feature = model.predict(
                image_array,
                verbose=0
            )[0]

            features.append(feature)
            labels.append(label)

        except Exception as e:
            print("Skipping:", image_path)
            print("Reason:", e)


# -------------------------------
# Convert to NumPy arrays
# -------------------------------

features = np.array(features)
labels = np.array(labels)


print("\nFinished!")
print("Feature shape:", features.shape)
print("Label shape:", labels.shape)


# -------------------------------
# Save
# -------------------------------

np.save("cnn_features.npy", features)
np.save("labels.npy", labels)

print("\nSaved:")
print("cnn_features.npy")
print("labels.npy")