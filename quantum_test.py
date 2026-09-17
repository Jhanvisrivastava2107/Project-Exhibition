import pennylane as qml
from pennylane import numpy as np


# Number of qubits
n_qubits = 4


# Create quantum simulator
dev = qml.device(
    "default.qubit",
    wires=n_qubits
)


@qml.qnode(dev)
def circuit(features, weights):

    # Encode data
    qml.AngleEmbedding(
        features,
        wires=range(n_qubits),
        rotation="Y"
    )

    # Trainable gates
    for i in range(n_qubits):
        qml.RY(
            weights[i],
            wires=i
        )

    # Entangle qubits
    for i in range(n_qubits - 1):
        qml.CNOT(
            wires=[i, i + 1]
        )

    # Measurement
    return qml.expval(
        qml.PauliZ(0)
    )


# Example input
features = np.array([
    0.3,
    -0.7,
    1.2,
    0.5
])


# Example trainable parameters
weights = np.array([
    0.1,
    0.2,
    0.3,
    0.4
])


# Run circuit
result = circuit(
    features,
    weights
)


print("Quantum output:")
print(result)


print("\nCircuit:")
print(
    qml.draw(circuit)(
        features,
        weights
    )
)