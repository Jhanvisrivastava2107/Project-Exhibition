import tensorflow as tf
import numpy as np
from PIL import Image


# -------------------------------
# 1. Load MobileNetV2
# -------------------------------

model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

model.trainable = False

print("MobileNetV2 loaded successfully.")


# -------------------------------
# 2. Choose one image
# -------------------------------

image_path = "data/healthy/WhatsApp Image 2026-09-16 at 9.47.02 PM (1).jpeg"


# -------------------------------
# 3. Load image
# -------------------------------

image = Image.open(image_path).convert("RGB")

image = image.resize((224, 224))

image_array = np.array(image)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)


# -------------------------------
# 4. Preprocess
# -------------------------------

image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    image_array
)


# -------------------------------
# 5. Extract features
# -------------------------------

features = model.predict(
    image_array,
    verbose=0
)

print("Feature shape:", features.shape)

print("First 10 features:")
print(features[0][:10])

import tensorflow as tf

print("TensorFlow version:", tf.__version__)

model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

print("MobileNetV2 loaded successfully!")