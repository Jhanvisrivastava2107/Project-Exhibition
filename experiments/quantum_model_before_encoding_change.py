from pathlib import Path

import numpy as np
import pennylane as qml


# ============================================================
# QUANTUM MODEL CONFIGURATION
# ============================================================

N_QUBITS = 8
N_LAYERS = 2
NUM_CLASSES = 9


# ============================================================
# QUANTUM DEVICE
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# QUANTUM CIRCUIT
# ============================================================

@qml.qnode(
    dev,
    interface="autograd",
    diff_method="backprop"
)
def quantum_circuit(inputs, weights):

    # --------------------------------------------------------
    # Angle encoding
    # --------------------------------------------------------

    for i in range(N_QUBITS):
        qml.RY(
            inputs[i],
            wires=i
        )

    # --------------------------------------------------------
    # Trainable quantum layers
    # --------------------------------------------------------

    for layer in range(N_LAYERS):

        for qubit in range(N_QUBITS):

            qml.RX(
                weights[layer, qubit, 0],
                wires=qubit
            )

            qml.RY(
                weights[layer, qubit, 1],
                wires=qubit
            )

            qml.RZ(
                weights[layer, qubit, 2],
                wires=qubit
            )

        # ----------------------------------------------------
        # Ring entanglement
        # ----------------------------------------------------

        for qubit in range(N_QUBITS - 1):

            qml.CNOT(
                wires=[
                    qubit,
                    qubit + 1
                ]
            )

        qml.CNOT(
            wires=[
                N_QUBITS - 1,
                0
            ]
        )

    # --------------------------------------------------------
    # Measurements
    # --------------------------------------------------------

    return [
        qml.expval(
            qml.PauliZ(i)
        )
        for i in range(N_QUBITS)
    ]


# ============================================================
# PCA FEATURES TO QUANTUM ANGLES
# ============================================================

def to_angles(X):

    X = np.asarray(
        X,
        dtype=np.float64
    )

    return np.tanh(X) * np.pi


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits):

    logits = np.asarray(
        logits,
        dtype=np.float64
    )

    shifted = (
        logits
        - np.max(
            logits,
            axis=-1,
            keepdims=True
        )
    )

    exp_values = np.exp(
        shifted
    )

    return (
        exp_values
        / np.sum(
            exp_values,
            axis=-1,
            keepdims=True
        )
    )


# ============================================================
# SINGLE SAMPLE FORWARD PASS
# ============================================================

def forward_single(
    x,
    quantum_weights,
    readout_weights,
    bias
):

    x = qml.numpy.array(
        x,
        requires_grad=False
    )

    quantum_weights = qml.numpy.array(
        quantum_weights,
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

    return logits


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

def predict_proba(
    X,
    quantum_weights,
    readout_weights,
    bias
):

    X = np.asarray(
        X,
        dtype=np.float64
    )

    all_logits = []

    for sample in X:

        logits = forward_single(
            sample,
            quantum_weights,
            readout_weights,
            bias
        )

        all_logits.append(
            np.asarray(
                logits,
                dtype=np.float64
            )
        )

    all_logits = np.asarray(
        all_logits,
        dtype=np.float64
    )

    return softmax(
        all_logits
    )


# ============================================================
# PREDICT CLASS
# ============================================================

def predict(
    X,
    quantum_weights,
    readout_weights,
    bias
):

    probabilities = predict_proba(
        X,
        quantum_weights,
        readout_weights,
        bias
    )

    return np.argmax(
        probabilities,
        axis=1
    )


# ============================================================
# SAVE QUANTUM MODEL
# ============================================================

def save_quantum_model(
    path,
    quantum_weights,
    readout_weights,
    bias
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        path,
        quantum_weights=np.asarray(
            quantum_weights
        ),
        readout_weights=np.asarray(
            readout_weights
        ),
        bias=np.asarray(
            bias
        )
    )


# ============================================================
# LOAD QUANTUM MODEL
# ============================================================

def load_quantum_model(path):

    data = np.load(
        path
    )

    quantum_weights = data[
        "quantum_weights"
    ]

    readout_weights = data[
        "readout_weights"
    ]

    bias = data[
        "bias"
    ]

    return (
        quantum_weights,
        readout_weights,
        bias
    )