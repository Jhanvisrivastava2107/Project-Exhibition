from pathlib import Path
import json

import numpy as np
import pennylane as qml

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from quantum_model import (
    N_QUBITS,
    N_LAYERS,
    NUM_CLASSES,
    to_angles,
    quantum_circuit,
    softmax,
    save_quantum_model
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURE_DIR = BASE_DIR / "features"
ARTIFACT_DIR = BASE_DIR / "artifacts"

X_PATH = FEATURE_DIR / "X_train_pca.npy"
Y_PATH = FEATURE_DIR / "y_train.npy"

MODEL_PATH = ARTIFACT_DIR / "quantum_model.npz"
HISTORY_PATH = ARTIFACT_DIR / "quantum_training_history.json"


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

RANDOM_SEED = 42

LEARNING_RATE = 0.01

EPOCHS = 60

BATCH_SIZE = 16

VALIDATION_SIZE = 0.15

PATIENCE = 10


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(RANDOM_SEED)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("VQC V2 TRAINING")
print("=" * 70)

print("\nLoading PCA training data...")

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nExpected:")
print("  Features:", N_QUBITS)
print("  Classes :", NUM_CLASSES)
print("  Layers  :", N_LAYERS)


# ============================================================
# CHECK DATA
# ============================================================

if X.shape[1] != N_QUBITS:
    raise ValueError(
        f"Expected {N_QUBITS} features, "
        f"but received {X.shape[1]}"
    )

unique_classes = np.unique(y)

if len(unique_classes) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(unique_classes)}"
    )


# ============================================================
# CONVERT FEATURES TO QUANTUM ANGLES
# ============================================================

print("\nConverting PCA features to quantum angles...")

X_angles = to_angles(X)

print("Angle feature shape:", X_angles.shape)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\nCreating validation split...")

X_train, X_val, y_train, y_val = train_test_split(
    X_angles,
    y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

print("Training samples  :", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# INITIALIZE PARAMETERS
# ============================================================

print("\nInitializing quantum parameters...")

rng = np.random.default_rng(
    RANDOM_SEED
)

quantum_weights = qml.numpy.array(
    0.05 * rng.standard_normal(
        (
            N_LAYERS,
            N_QUBITS,
            3
        )
    ),
    requires_grad=True
)

readout_weights = qml.numpy.array(
    0.05 * rng.standard_normal(
        (
            NUM_CLASSES,
            N_QUBITS
        )
    ),
    requires_grad=True
)

bias = qml.numpy.zeros(
    NUM_CLASSES,
    requires_grad=True
)


# ============================================================
# FORWARD PASS
# ============================================================

def forward_batch(
    X_batch,
    quantum_weights,
    readout_weights,
    bias
):

    logits_list = []

    for x in X_batch:

        x = qml.numpy.array(
            x,
            requires_grad=False
        )

        quantum_features = quantum_circuit(
            x,
            quantum_weights
        )

        quantum_features = qml.numpy.stack(
            quantum_features
        )

        logits = (
            qml.numpy.dot(
                readout_weights,
                quantum_features
            )
            + bias
        )

        logits_list.append(logits)

    return qml.numpy.stack(
        logits_list
    )


# ============================================================
# CROSS ENTROPY LOSS
# ============================================================

def cross_entropy_loss(
    logits,
    labels
):

    # Numerically stable softmax
    shifted_logits = (
        logits
        - qml.numpy.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    exp_logits = qml.numpy.exp(
        shifted_logits
    )

    probabilities = (
        exp_logits
        / qml.numpy.sum(
            exp_logits,
            axis=1,
            keepdims=True
        )
    )

    selected = probabilities[
        qml.numpy.arange(
            len(labels)
        ),
        labels
    ]

    return -qml.numpy.mean(
        qml.numpy.log(
            selected + 1e-10
        )
    )

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = qml.AdamOptimizer(
    stepsize=LEARNING_RATE
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
}


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_val_loss = float("inf")

best_quantum_weights = None
best_readout_weights = None
best_bias = None

epochs_without_improvement = 0


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # SHUFFLE TRAINING DATA
    # --------------------------------------------------------

    permutation = rng.permutation(
        len(X_train)
    )

    X_train_shuffled = X_train[
        permutation
    ]

    y_train_shuffled = y_train[
        permutation
    ]

    # --------------------------------------------------------
    # MINI-BATCH TRAINING
    # --------------------------------------------------------

    batch_losses = []

    for start in range(
        0,
        len(X_train_shuffled),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_train_shuffled)
        )

        X_batch = X_train_shuffled[
            start:end
        ]

        y_batch = y_train_shuffled[
            start:end
        ]

        def batch_cost(
            qw,
            rw,
            b
        ):

            logits = forward_batch(
                X_batch,
                qw,
                rw,
                b
            )

            return cross_entropy_loss(
                logits,
                y_batch
            )

        (
            quantum_weights,
            readout_weights,
            bias
        ), loss_value = optimizer.step_and_cost(
            batch_cost,
            quantum_weights,
            readout_weights,
            bias
        )

        batch_losses.append(
            float(loss_value)
        )

    # --------------------------------------------------------
    # TRAINING METRICS
    # --------------------------------------------------------

    train_logits = forward_batch(
        X_train,
        quantum_weights,
        readout_weights,
        bias
    )

    train_probabilities = np.asarray(
        softmax(
            np.asarray(
                train_logits,
                dtype=np.float64
            )
        )
    )
    

    train_predictions = np.argmax(
        train_probabilities,
        axis=1
    )

    train_loss = float(
        cross_entropy_loss(
            train_logits,
            y_train
        )
    )

    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    # --------------------------------------------------------
    # VALIDATION METRICS
    # --------------------------------------------------------

    val_logits = forward_batch(
        X_val,
        quantum_weights,
        readout_weights,
        bias
    )

    
    

    val_probabilities = np.asarray(
        softmax(
            np.asarray(
                val_logits,
                dtype=np.float64
            )
        )
    )

    val_predictions = np.argmax(
        val_probabilities,
        axis=1
    )

    val_loss = float(
        cross_entropy_loss(
            val_logits,
            y_val
        )
    )

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["val_loss"].append(
        val_loss
    )

    history["train_accuracy"].append(
        float(train_accuracy)
    )

    history["val_accuracy"].append(
        float(val_accuracy)
    )

    # --------------------------------------------------------
    # PRINval_probT PROGRESS
    # --------------------------------------------------------

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_quantum_weights = np.asarray(
            quantum_weights
        ).copy()

        best_readout_weights = np.asarray(
            readout_weights
        ).copy()

        best_bias = np.asarray(
            bias
        ).copy()

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch}."
        )

        break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_quantum_weights is None:

    raise RuntimeError(
        "No valid best model was saved."
    )


quantum_weights = best_quantum_weights
readout_weights = best_readout_weights
bias = best_bias


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING BEST VQC MODEL")
print("=" * 70)

save_quantum_model(
    MODEL_PATH,
    quantum_weights,
    readout_weights,
    bias
)

print(
    "\nSaved model:"
)

print(
    MODEL_PATH
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    HISTORY_PATH,
    "w"
) as file:

    json.dump(
        history,
        file,
        indent=2
    )

print(
    "\nSaved training history:"
)

print(
    HISTORY_PATH
)


# ============================================================
# FINAL VALIDATION RESULT
# ============================================================

best_epoch = (
    int(
        np.argmin(
            history["val_loss"]
        )
    )
    + 1
)

best_val_accuracy = (
    history["val_accuracy"][
        best_epoch - 1
    ]
)

print("\n" + "=" * 70)
print("VQC V2 TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest epoch       : {best_epoch}"
)

print(
    f"Best val accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best val loss    : "
    f"{best_val_loss:.4f}"
)

print("\nModel configuration:")
print(
    f"  Qubits : {N_QUBITS}"
)
print(
    f"  Layers : {N_LAYERS}"
)
print(
    f"  Classes: {NUM_CLASSES}"
)
print(
    f"  LR     : {LEARNING_RATE}"
)
print(
    f"  Epochs : {EPOCHS}"
)

print("\nTraining complete.")