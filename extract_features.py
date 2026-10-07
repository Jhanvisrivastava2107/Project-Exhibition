from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_DIR = PROJECT_ROOT / "data" / "split" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "split" / "test"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FEATURE_DIR = PROJECT_ROOT / "features"

CLASS_NAMES_FILE = ARTIFACT_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK PATHS
# ============================================================

print("=" * 70)
print("STEP 2: MOBILE NET V2 FEATURE EXTRACTION")
print("=" * 70)

print(f"\nProject root:")
print(PROJECT_ROOT)

print(f"\nTraining directory:")
print(TRAIN_DIR)

print(f"\nTesting directory:")
print(TEST_DIR)


if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"\nTraining directory not found:\n{TRAIN_DIR}\n\n"
        "Run prepare_data.py first."
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"\nTesting directory not found:\n{TEST_DIR}\n\n"
        "Run prepare_data.py first."
    )

if not CLASS_NAMES_FILE.exists():
    raise FileNotFoundError(
        f"\nClass names file not found:\n{CLASS_NAMES_FILE}\n\n"
        "Run prepare_data.py first."
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
    class_names = json.load(f)


print("\nDetected classes:")

for index, class_name in enumerate(class_names):
    print(f"  {index}: {class_name}")


if len(class_names) != 9:
    raise ValueError(
        f"Expected exactly 9 classes, but found {len(class_names)}."
    )


# ============================================================
# LOAD MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2...")

model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

model.trainable = False

print("MobileNetV2 loaded successfully.")
print("Feature dimension: 1280")


# ============================================================
# IMAGE DATA GENERATOR
# ============================================================

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


# ============================================================
# TRAIN GENERATOR
# ============================================================

print("\nReading training images...")

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=class_names,
    shuffle=False
)


# ============================================================
# TEST GENERATOR
# ============================================================

print("\nReading testing images...")

test_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=class_names,
    shuffle=False
)


# ============================================================
# VERIFY CLASS MAPPING
# ============================================================

print("\nClass mapping used by TensorFlow:")

for class_name, index in train_generator.class_indices.items():
    print(f"  {index}: {class_name}")


# ============================================================
# EXTRACT TRAIN FEATURES
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING TRAINING FEATURES")
print("=" * 70)

train_generator.reset()

X_train = model.predict(
    train_generator,
    verbose=1
)

y_train = train_generator.classes.astype(np.int64)


# ============================================================
# EXTRACT TEST FEATURES
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING TEST FEATURES")
print("=" * 70)

test_generator.reset()

X_test = model.predict(
    test_generator,
    verbose=1
)

y_test = test_generator.classes.astype(np.int64)


# ============================================================
# VERIFY SHAPES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE EXTRACTION RESULTS")
print("=" * 70)

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

print(f"\nX_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")


# ============================================================
# SAVE FEATURES
# ============================================================

np.save(FEATURE_DIR / "X_train.npy", X_train)
np.save(FEATURE_DIR / "y_train.npy", y_train)

np.save(FEATURE_DIR / "X_test.npy", X_test)
np.save(FEATURE_DIR / "y_test.npy", y_test)


# ============================================================
# SAVE CLASS NAMES AGAIN
# ============================================================

with open(
    FEATURE_DIR / "class_names.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(class_names, f, indent=4)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print("\nSaved files:")

print(FEATURE_DIR / "X_train.npy")
print(FEATURE_DIR / "y_train.npy")
print(FEATURE_DIR / "X_test.npy")
print(FEATURE_DIR / "y_test.npy")
print(FEATURE_DIR / "class_names.json")

print("\nExpected feature dimension: 1280")
print("\nNext step:")
print("    python train_classical.py")