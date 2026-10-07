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
    quantum_circuit,
    to_angles,
    softmax,
    save_quantum_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

FEATURE_DIR = PROJECT_ROOT / "features"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

X_TRAIN_FILE = FEATURE_DIR / "X_train_pca.npy"
Y_TRAIN_FILE = FEATURE_DIR / "y_train.npy"

MODEL_FILE = ARTIFACT_DIR / "quantum_model.npz"
HISTORY_FILE = ARTIFACT_DIR / "quantum_training_history.json"

EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.02
PATIENCE = 7

VALIDATION_SIZE = 0.15

SEED = 42


# ============================================================
# SEED
# ============================================================

np.random.seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 4: VARIATIONAL QUANTUM CLASSIFIER TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading PCA features...")

X = np.load(X_TRAIN_FILE)
y = np.load(Y_TRAIN_FILE)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# VALIDATION SPLIT
# ============================================================

print("\nCreating validation split...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    stratify=y
)

print(f"Training samples  : {len(X_train)}")
print(f"Validation samples: {len(X_val)}")


# ============================================================
# ANGLE ENCODING
# ============================================================

X_train_angles = to_angles(X_train)
X_val_angles = to_angles(X_val)

print("\nQuantum input dimension:")
print(X_train_angles.shape[1])

if X_train_angles.shape[1] != N_QUBITS:
    raise ValueError(
        f"Expected {N_QUBITS} PCA features, "
        f"but found {X_train_angles.shape[1]}."
    )


# ============================================================
# INITIALIZE PARAMETERS
# ============================================================

print("\nInitializing quantum parameters...")

quantum_weights = qml.numpy.array(
    0.01 * np.random.randn(
        N_LAYERS,
        N_QUBITS,
        3
    ),
    requires_grad=True
)

readout_weights = qml.numpy.array(
    0.01 * np.random.randn(
        NUM_CLASSES,
        N_QUBITS
    ),
    requires_grad=True
)

bias = qml.numpy.zeros(
    NUM_CLASSES,
    requires_grad=True
)


# ============================================================
# LOSS FUNCTION
# ============================================================

def batch_loss(
    q_weights,
    r_weights,
    b,
    X_batch,
    y_batch
):

    losses = []

    for x, label in zip(
        X_batch,
        y_batch
    ):

        x = qml.numpy.array(
            x,
            requires_grad=False
        )

        quantum_features = quantum_circuit(
            x,
            q_weights
        )

        quantum_features = qml.numpy.stack(
            quantum_features
        )

        logits = qml.numpy.dot(
            r_weights,
            quantum_features
        ) + b

        probabilities = qml.numpy.exp(
            logits - qml.numpy.max(logits)
        )

        probabilities = probabilities / qml.numpy.sum(
            probabilities
        )

        probability_of_true_class = (
            probabilities[int(label)] + 1e-8
        )

        losses.append(
            -qml.numpy.log(
                probability_of_true_class
            )
        )

    return qml.numpy.mean(
        qml.numpy.stack(losses)
    )


# ============================================================
# VALIDATION LOSS
# ============================================================

def validation_loss(
    q_weights,
    r_weights,
    b,
    X_data,
    y_data
):

    losses = []

    for x, label in zip(
        X_data,
        y_data
    ):

        x = qml.numpy.array(
            x,
            requires_grad=False
        )

        quantum_features = quantum_circuit(
            x,
            q_weights
        )

        quantum_features = qml.numpy.stack(
            quantum_features
        )

        logits = qml.numpy.dot(
            r_weights,
            quantum_features
        ) + b

        probabilities = qml.numpy.exp(
            logits - qml.numpy.max(logits)
        )

        probabilities = probabilities / qml.numpy.sum(
            probabilities
        )

        losses.append(
            -qml.numpy.log(
                probabilities[int(label)] + 1e-8
            )
        )

    return float(
        qml.numpy.mean(
            qml.numpy.stack(losses)
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
    "validation_loss": [],
    "validation_accuracy": []
}


best_validation_loss = float("inf")

best_quantum_weights = np.array(
    quantum_weights
)

best_readout_weights = np.array(
    readout_weights
)

best_bias = np.array(
    bias
)

patience_counter = 0


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 70)
print("STARTING QUANTUM TRAINING")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    # Shuffle training data
    permutation = np.random.permutation(
        len(X_train)
    )

    X_train_shuffled = X_train_angles[
        permutation
    ]

    y_train_shuffled = y_train[
        permutation
    ]

    batch_losses = []

    # --------------------------------------------------------
    # Mini-batches
    # --------------------------------------------------------

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

        def cost_function(
            q_weights,
            r_weights,
            b
        ):

            return batch_loss(
                q_weights,
                r_weights,
                b,
                X_batch,
                y_batch
            )

        (
            quantum_weights,
            readout_weights,
            bias
        ), loss_value = optimizer.step_and_cost(
            cost_function,
            quantum_weights,
            readout_weights,
            bias
        )

        batch_losses.append(
            float(loss_value)
        )

    # --------------------------------------------------------
    # Training loss
    # --------------------------------------------------------

    train_loss = float(
        np.mean(batch_losses)
    )

    # --------------------------------------------------------
    # Validation loss
    # --------------------------------------------------------

    val_loss = validation_loss(
        quantum_weights,
        readout_weights,
        bias,
        X_val_angles,
        y_val
    )

    # --------------------------------------------------------
    # Validation accuracy
    # --------------------------------------------------------

    validation_predictions = []

    for x in X_val_angles:

        x = qml.numpy.array(
            x,
            requires_grad=False
        )

        quantum_features = quantum_circuit(
            x,
            quantum_weights
        )

        quantum_features = np.asarray(
            quantum_features,
            dtype=np.float64
        )

        logits = (
            np.asarray(
                readout_weights,
                dtype=np.float64
            )
            @ quantum_features
            + np.asarray(
                bias,
                dtype=np.float64
            )
        )

        validation_predictions.append(
            np.argmax(logits)
        )

    validation_predictions = np.asarray(
        validation_predictions
    )

    val_accuracy = accuracy_score(
        y_val,
        validation_predictions
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["validation_loss"].append(
        val_loss
    )

    history["validation_accuracy"].append(
        float(val_accuracy)
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if val_loss < best_validation_loss:

        best_validation_loss = val_loss

        best_quantum_weights = np.array(
            quantum_weights
        )

        best_readout_weights = np.array(
            readout_weights
        )

        best_bias = np.array(
            bias
        )

        patience_counter = 0

        print("  -> Best model updated.")

    else:

        patience_counter += 1

        print(
            f"  -> No improvement "
            f"({patience_counter}/{PATIENCE})"
        )

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")

            break


# ============================================================
# SAVE BEST MODEL
# ============================================================

save_quantum_model(
    MODEL_FILE,
    best_quantum_weights,
    best_readout_weights,
    best_bias
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    HISTORY_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("QUANTUM TRAINING COMPLETE")
print("=" * 70)

print("\nQuantum model saved to:")
print(MODEL_FILE)

print("\nTraining history saved to:")
print(HISTORY_FILE)

print("\nArchitecture:")
print(f"  PCA features : {N_QUBITS}")
print(f"  Qubits       : {N_QUBITS}")
print(f"  Q-layers     : {N_LAYERS}")
print(f"  Output classes: {NUM_CLASSES}")

print("\nNext step:")
print("  python evaluate.py")