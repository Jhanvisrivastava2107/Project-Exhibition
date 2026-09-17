import tensorflow as tf
import numpy as np

# Load MobileNetV2 without the final classification layer
model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)
model.trainable = False

print("MobileNetV2 loaded successfully")
