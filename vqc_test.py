import pennylane as qml
from pennylane import numpy as np

n_qubits = 4

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def vqc(features, weights):

    # Step 1: Encode classical features
    # into quantum states using rotation angles
    qml.AngleEmbedding(
        features,
        wires=range(n_qubits),
        rotation="Y"
    )

    # Step 2: Trainable quantum rotations
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)

    # Step 3: Entangle neighbouring qubits
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    # Step 4: Measure qubit 0
    return qml.expval(qml.PauliZ(0))


# Example input from PCA
features = np.array([
    0.4,
    -0.7,
    1.2,
    -0.2
])

# Random trainable parameters
weights = np.array([
    0.1,
    0.2,
    0.3,
    0.4
])

result = vqc(features, weights)

print("Quantum output:", result)

print("\nQuantum circuit:")
print(qml.draw(vqc)(features, weights))