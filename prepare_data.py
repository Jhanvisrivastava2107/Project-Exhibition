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


# ============================================================
# PATHS
# ============================================================

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
# CREATE DIRECTORIES
# ============================================================

SPLIT_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
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
            f"\nRaw dataset directory not found:\n"
            f"{RAW_DATA_DIR}\n\n"
            "Expected structure:\n"
            "data\\raw\\class_name\\image.jpg"
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
            f"\nExpected exactly {NUM_CLASSES} classes, "
            f"but found {len(classes)}.\n\n"
            "Detected classes:\n"
            + "\n".join(
                f"  {i}: {name}"
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

    if SPLIT_DATA_DIR.exists():

        print(
            "\nRemoving old split dataset..."
        )

        shutil.rmtree(
            SPLIT_DATA_DIR
        )

    TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# COPY IMAGE WITH UNIQUE NAME
# ============================================================

def copy_images(
    image_paths,
    destination_dir,
    prefix
):

    destination_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We deliberately add an index to every filename.
        #
        # This prevents files with identical filenames
        # from different raw subfolders overwriting each other.
        # ----------------------------------------------------

        new_filename = (
            f"{prefix}_{index:04d}_"
            f"{image_path.name}"
        )

        destination = (
            destination_dir
            / new_filename
        )

        shutil.copy2(
            image_path,
            destination
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
    # FIND CLASSES
    # --------------------------------------------------------

    classes = get_classes()

    print(
        "\nDetected exactly "
        f"{len(classes)} classes:"
    )

    for index, class_name in enumerate(
        classes
    ):

        print(
            f"  {index}: {class_name}"
        )

    # --------------------------------------------------------
    # RESET SPLIT
    # --------------------------------------------------------

    reset_split_directories()

    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------

    total_images = 0

    total_train = 0

    total_test = 0

    print(
        "\n" + "=" * 70
    )

    print(
        "CREATING TRAIN / TEST SPLIT"
    )

    print(
        "=" * 70
    )

    for class_index, class_name in enumerate(
        classes
    ):

        source_dir = (
            RAW_DATA_DIR
            / class_name
        )

        images = get_images(
            source_dir
        )

        # ----------------------------------------------------
        # VERIFY RAW COUNT
        # ----------------------------------------------------

        print(
            f"\nClass: {class_name}"
        )

        print(
            f"Raw images: {len(images)}"
        )

        if len(images) == 0:

            raise ValueError(
                f"Class '{class_name}' "
                "contains no images."
            )

        # ----------------------------------------------------
        # TRAIN / TEST SPLIT
        # ----------------------------------------------------

        train_images, test_images = (
            train_test_split(
                images,
                test_size=TEST_SIZE,
                random_state=SEED,
                shuffle=True
            )
        )

        # ----------------------------------------------------
        # DESTINATION DIRECTORIES
        # ----------------------------------------------------

        train_class_dir = (
            TRAIN_DIR
            / class_name
        )

        test_class_dir = (
            TEST_DIR
            / class_name
        )

        # ----------------------------------------------------
        # COPY TRAINING IMAGES
        # ----------------------------------------------------

        copy_images(
            train_images,
            train_class_dir,
            f"class{class_index}_train"
        )

        # ----------------------------------------------------
        # COPY TESTING IMAGES
        # ----------------------------------------------------

        copy_images(
            test_images,
            test_class_dir,
            f"class{class_index}_test"
        )

        # ----------------------------------------------------
        # COUNTS
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

        print(
            f"Train images: {train_count}"
        )

        print(
            f"Test images : {test_count}"
        )

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if (
            train_count
            + test_count
            != total
        ):

            raise RuntimeError(
                f"Split count mismatch "
                f"for class '{class_name}'."
            )

    # ========================================================
    # SAVE CLASS NAMES
    # ========================================================

    class_names_path = (
        ARTIFACT_DIR
        / "class_names.json"
    )

    with open(
        class_names_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            classes,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "VERIFYING SPLIT DATASET"
    )

    print(
        "=" * 70
    )

    for class_name in classes:

        train_count = len(
            get_images(
                TRAIN_DIR / class_name
            )
        )

        test_count = len(
            get_images(
                TEST_DIR / class_name
            )
        )

        print(
            f"{class_name:30s}"
            f" Train: {train_count:4d}"
            f" | Test: {test_count:4d}"
        )

        if train_count + test_count == 0:

            raise RuntimeError(
                f"No images found after "
                f"splitting class '{class_name}'."
            )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "DATA PREPARATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal raw images : {total_images}"
    )

    print(
        f"Total train      : {total_train}"
    )

    print(
        f"Total test       : {total_test}"
    )

    print(
        f"\nTrain directory:"
    )

    print(
        TRAIN_DIR
    )

    print(
        f"\nTest directory:"
    )

    print(
        TEST_DIR
    )

    print(
        f"\nClass names:"
    )

    print(
        class_names_path
    )

    print(
        "\nClass mapping:"
    )

    for index, class_name in enumerate(
        classes
    ):

        print(
            f"  {index} -> {class_name}"
        )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()