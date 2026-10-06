import numpy as np
import pennylane as qml

from pennylane import numpy as pnp


# -----------------------------
# LOAD PCA DATA
# -----------------------------

X_train = np.load("X_train_pca.npy")
X_test = np.load("X_test_pca.npy")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")


# -----------------------------
# SETTINGS
# -----------------------------

n_qubits = 4

print("Number of qubits:", n_qubits)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# QUANTUM DEVICE
# -----------------------------

dev = qml.device(
    "default.qubit",
    wires=n_qubits
)


# -----------------------------
# QUANTUM CIRCUIT
# -----------------------------

@qml.qnode(dev)
def circuit(features, weights):

    # Encode the 4 features
    qml.AngleEmbedding(
        features,
        wires=range(n_qubits),
        rotation="Y"
    )

    # Trainable rotations
    for i in range(n_qubits):
        qml.RY(
            weights[i],
            wires=i
        )

    # Connect the qubits
    for i in range(n_qubits - 1):
        qml.CNOT(
            wires=[i, i + 1]
        )

    # Measurements
    return [
        qml.expval(qml.PauliZ(i))
        for i in range(n_qubits)
    ]


# -----------------------------
# INITIAL WEIGHTS
# -----------------------------

weights = pnp.array(
    np.random.uniform(
        -0.1,
        0.1,
        n_qubits
    ),
    requires_grad=True
)


# -----------------------------
# PREDICTION
# -----------------------------

def predict(features, weights):

    output = circuit(
        features,
        weights
    )

    return pnp.stack(output)


# -----------------------------
# LOSS FUNCTION
# -----------------------------

def loss_function(weights):

    total_loss = 0

    for features, label in zip(
        X_train,
        y_train
    ):

        output = predict(
            features,
            weights
        )

        # Convert label into one-hot form
        target = pnp.zeros(4)
        target = qml.numpy.array(target)

        target[label] = 1.0

        total_loss += pnp.mean(
            (output - target) ** 2
        )

    return total_loss / len(X_train)


# -----------------------------
# OPTIMIZER
# -----------------------------

optimizer = qml.AdamOptimizer(
    stepsize=0.05
)


# -----------------------------
# TRAINING
# -----------------------------

print("\nTraining quantum model...")

epochs = 20

for epoch in range(epochs):

    weights, current_loss = optimizer.step_and_cost(
        loss_function,
        weights
    )

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Loss: {current_loss:.4f}"
    )


# -----------------------------
# TESTING
# -----------------------------

quantum_predictions = []

for features in X_test:

    output = predict(
        features,
        weights
    )

    predicted_class = int(
        pnp.argmax(output)
    )

    quantum_predictions.append(
        predicted_class
    )


quantum_predictions = np.array(
    quantum_predictions
)


# -----------------------------
# SAVE RESULTS
# -----------------------------

np.save(
    "quantum_predictions.npy",
    quantum_predictions
)

np.save(
    "quantum_weights.npy",
    np.array(weights)
)

print("\nQuantum predictions:")
print(quantum_predictions)

print("\nActual labels:")
print(y_test)

print("\nQuantum model training completed!")