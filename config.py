from pathlib import Path
import random

import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# DATA PATHS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

SPLIT_DATA_DIR = DATA_DIR / "split"

TRAIN_DIR = SPLIT_DATA_DIR / "train"

TEST_DIR = SPLIT_DATA_DIR / "test"


# ============================================================
# OUTPUT PATHS
# ============================================================

FEATURE_DIR = PROJECT_ROOT / "features"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ============================================================
# DATASET SETTINGS
# ============================================================

NUM_CLASSES = 9

IMAGE_SIZE = (224, 224)


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42


# ============================================================
# PCA
# ============================================================

PCA_COMPONENTS = 8


# ============================================================
# QUANTUM SETTINGS
# ============================================================

N_QUBITS = 8

N_Q_LAYERS = 2

QUANTUM_EPOCHS = 30

QUANTUM_BATCH_SIZE = 16

QUANTUM_LEARNING_RATE = 0.02

QUANTUM_PATIENCE = 7


# ============================================================
# CREATE DIRECTORIES
# ============================================================

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed():

    random.seed(SEED)

    np.random.seed(SEED)