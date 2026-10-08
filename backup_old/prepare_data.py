import json
import random
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_CLASSES = 9

TEST_SIZE = 0.20

# This file is directly inside:
# C:\Users\Sanjeev\Desktop\project exhibition\prepare_data.py
#
# Therefore Path(__file__).resolve().parent gives:
# C:\Users\Sanjeev\Desktop\project exhibition

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SPLIT_DATA_DIR = PROJECT_ROOT / "data" / "split"

TRAIN_DIR = SPLIT_DATA_DIR / "train"

TEST_DIR = SPLIT_DATA_DIR / "test"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

SPLIT_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed():

    random.seed(SEED)

    np.random.seed(SEED)


# ============================================================
# FIND CLASSES
# ============================================================

def get_classes():

    print(
        f"\nLooking for dataset at:\n"
        f"{RAW_DATA_DIR}\n"
    )

    if not RAW_DATA_DIR.exists():

        raise FileNotFoundError(
            "\nRaw dataset directory was not found.\n\n"
            f"Expected location:\n"
            f"{RAW_DATA_DIR}\n\n"
            "Please create this folder and put your "
            "9 disease-class folders inside it.\n\n"
            "Example:\n"
            f"{RAW_DATA_DIR}\\Class1\n"
            f"{RAW_DATA_DIR}\\Class2\n"
            f"...\n"
            f"{RAW_DATA_DIR}\\Class9\n"
        )

    classes = sorted(
        [
            folder.name
            for folder in RAW_DATA_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    if len(classes) != NUM_CLASSES:

        raise ValueError(
            "\nExpected exactly "
            f"{NUM_CLASSES} classes, "
            f"but found {len(classes)}.\n\n"
            "Detected folders:\n"
            + "\n".join(
                f"  {i + 1}. {name}"
                for i, name in enumerate(classes)
            )
        )

    return classes


# ============================================================
# FIND IMAGES
# ============================================================

def get_images(class_dir):

    images = []

    for path in class_dir.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            images.append(path)

    return sorted(images)


# ============================================================
# RESET SPLIT DIRECTORIES
# ============================================================

def reset_split_directories():

    if TRAIN_DIR.exists():

        print(
            f"\nRemoving old train directory:\n"
            f"{TRAIN_DIR}"
        )

        shutil.rmtree(TRAIN_DIR)

    if TEST_DIR.exists():

        print(
            f"\nRemoving old test directory:\n"
            f"{TEST_DIR}"
        )

        shutil.rmtree(TEST_DIR)

    TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print("=" * 70)
    print("STEP 1: PREPARING DATASET")
    print("=" * 70)

    print(
        f"\nProject root:\n"
        f"{PROJECT_ROOT}"
    )

    print(
        f"\nRaw dataset:\n"
        f"{RAW_DATA_DIR}"
    )

    # --------------------------------------------------------
    # Find classes
    # --------------------------------------------------------

    classes = get_classes()

    print(
        "\nDetected exactly "
        f"{len(classes)} classes:"
    )

    for index, class_name in enumerate(classes):

        print(
            f"  {index}: {class_name}"
        )

    # --------------------------------------------------------
    # Reset split directories
    # --------------------------------------------------------

    reset_split_directories()

    # --------------------------------------------------------
    # Split each class
    # --------------------------------------------------------

    total_images = 0

    total_train = 0

    total_test = 0

    for class_name in classes:

        source_dir = (
            RAW_DATA_DIR / class_name
        )

        images = get_images(
            source_dir
        )

        if len(images) < 2:

            raise ValueError(
                f"\nClass '{class_name}' "
                "contains fewer than 2 images."
            )

        train_images, test_images = (
            train_test_split(
                images,
                test_size=TEST_SIZE,
                random_state=SEED,
                shuffle=True,
            )
        )

        train_class_dir = (
            TRAIN_DIR / class_name
        )

        test_class_dir = (
            TEST_DIR / class_name
        )

        train_class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        test_class_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Copy training images
        # ----------------------------------------------------

        for image_path in train_images:

            destination = (
                train_class_dir
                / image_path.name
            )

            shutil.copy2(
                image_path,
                destination,
            )

        # ----------------------------------------------------
        # Copy testing images
        # ----------------------------------------------------

        for image_path in test_images:

            destination = (
                test_class_dir
                / image_path.name
            )

            shutil.copy2(
                image_path,
                destination,
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total = len(images)

        train_count = len(
            train_images
        )

        test_count = len(
            test_images
        )

        total_images += total

        total_train += train_count

        total_test += test_count

        print("\n" + "-" * 60)

        print(
            f"Class: {class_name}"
        )

        print(
            f"Total : {total}"
        )

        print(
            f"Train : {train_count}"
        )

        print(
            f"Test  : {test_count}"
        )

    # --------------------------------------------------------
    # Save class names
    # --------------------------------------------------------

    class_names_path = (
        ARTIFACT_DIR
        / "class_names.json"
    )

    with open(
        class_names_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            classes,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal images : {total_images}"
    )

    print(
        f"Training     : {total_train}"
    )

    print(
        f"Testing      : {total_test}"
    )

    print(
        f"\nTrain directory:\n"
        f"{TRAIN_DIR}"
    )

    print(
        f"\nTest directory:\n"
        f"{TEST_DIR}"
    )

    print(
        f"\nClass names saved to:\n"
        f"{class_names_path}"
    )

    print("\nClass mapping:")

    for index, class_name in enumerate(
        classes
    ):

        print(
            f"  {index} -> {class_name}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":

    main()