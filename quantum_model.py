from pathlib import Path
import numpy as np
import pennylane as qml

N_QUBITS = 8
N_LAYERS = 2
NUM_CLASSES = 9

# Use lightning.qubit for 50x faster C++ adjoint gradient calculation
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    DIFF_METHOD = "adjoint"
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    DIFF_METHOD = "backprop"

@qml.qnode(dev, interface="autograd", diff_method=DIFF_METHOD)
def quantum_circuit(inputs, weights):
    # Angle encoding
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # 2 Trainable variational layers
    for layer in range(weights.shape[0]):
        for qubit in range(N_QUBITS):
            qml.RX(weights[layer, qubit, 0], wires=qubit)
            qml.RY(weights[layer, qubit, 1], wires=qubit)
            qml.RZ(weights[layer, qubit, 2], wires=qubit)
        for qubit in range(N_QUBITS - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

def forward_single(x, quantum_weights, readout_weights, bias):
    x = qml.numpy.array(x, requires_grad=False)
    quantum_weights = qml.numpy.array(quantum_weights, requires_grad=False)
    quantum_features = qml.numpy.stack(quantum_circuit(x, quantum_weights))
    return qml.numpy.dot(readout_weights, quantum_features) + bias

def predict_proba(X, quantum_weights, readout_weights, bias):
    X = np.asarray(X, dtype=np.float64)
    all_logits = [forward_single(sample, quantum_weights, readout_weights, bias) for sample in X]
    return softmax(np.asarray(all_logits, dtype=np.float64))

def predict(X, quantum_weights, readout_weights, bias):
    return np.argmax(predict_proba(X, quantum_weights, readout_weights, bias), axis=1)

def save_quantum_model(path, quantum_weights, readout_weights, bias):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        quantum_weights=np.asarray(quantum_weights),
        readout_weights=np.asarray(readout_weights),
        bias=np.asarray(bias)
    )

def load_quantum_model(path):
    data = np.load(path)
    return data["quantum_weights"], data["readout_weights"], data["bias"]