import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

TRAIN_DIR = "splits/train"
TEST_DIR = "splits/test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 9


train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)


test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


print("\nClasses:")
print(train_dataset.class_names)


base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False


model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    layers.Lambda(preprocess_input),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.3),

    layers.Dense(NUM_CLASSES, activation="softmax")

])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


print("\nTraining CNN...")


history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)


print("\nEvaluating CNN...")


loss, accuracy = model.evaluate(test_dataset)


print("\n==============================")
print("CNN RESULTS")
print("==============================")

print("Accuracy:", accuracy)
print("Accuracy %:", accuracy * 100)


model.save("model/classical_cnn.keras")

print("\nCNN model saved!")