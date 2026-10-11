from pathlib import Path
import json
import joblib
import numpy as np
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from quantum_model import (
    N_QUBITS,
    N_LAYERS,
    NUM_CLASSES,
    quantum_circuit,
    save_quantum_model
)

BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "features"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = FEATURE_DIR / "X_train_pca.npy"
Y_PATH = FEATURE_DIR / "y_train.npy"
MODEL_PATH = ARTIFACT_DIR / "quantum_model.npz"
SCALER_PATH = ARTIFACT_DIR / "quantum_scaler.joblib"
HISTORY_PATH = ARTIFACT_DIR / "quantum_training_history.json"

RANDOM_SEED = 42
LEARNING_RATE = 0.02
EPOCHS = 15
BATCH_SIZE = 32
PATIENCE = 10

np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

X_raw = np.load(X_PATH)
y = np.load(Y_PATH)

# Fit and save explicit angle scaler into [-pi/2, pi/2]
scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
X_scaled = scaler.fit_transform(X_raw)
joblib.dump(scaler, SCALER_PATH)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
)

# Proper initialization scale: gives healthy gradient flow into quantum layers
quantum_weights = qml.numpy.array(
    0.1 * rng.standard_normal((N_LAYERS, N_QUBITS, 3)), requires_grad=True
)
readout_weights = qml.numpy.array(
    0.5 * rng.standard_normal((NUM_CLASSES, N_QUBITS)), requires_grad=True
)
bias = qml.numpy.zeros(NUM_CLASSES, requires_grad=True)

def forward_batch(X_batch, qw, rw, b):
    logits_list = []
    for x in X_batch:
        x_arr = qml.numpy.array(x, requires_grad=False)
        q_feats = qml.numpy.stack(quantum_circuit(x_arr, qw))
        logits_list.append(qml.numpy.dot(rw, q_feats) + b)
    return qml.numpy.stack(logits_list)

def cross_entropy_loss(logits, labels):
    shifted = logits - qml.numpy.max(logits, axis=1, keepdims=True)
    exp_l = qml.numpy.exp(shifted)
    probs = exp_l / qml.numpy.sum(exp_l, axis=1, keepdims=True)
    selected = probs[qml.numpy.arange(len(labels)), labels]
    return -qml.numpy.mean(qml.numpy.log(selected + 1e-10))

opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)

history = {"train_loss": [], "val_accuracy": []}
best_val_acc = 0.0
best_weights = None
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    perm = rng.permutation(len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]
    
    batch_losses = []
    for start in range(0, len(X_shuffled), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X_shuffled))
        X_b = X_shuffled[start:end]
        y_b = y_shuffled[start:end]
        
        (quantum_weights, readout_weights, bias), loss_val = opt.step_and_cost(
            lambda qw, rw, b: cross_entropy_loss(forward_batch(X_b, qw, rw, b), y_b),
            quantum_weights, readout_weights, bias
        )
        batch_losses.append(float(loss_val))
        
    train_loss = float(np.mean(batch_losses))
    val_logits = forward_batch(X_val, quantum_weights, readout_weights, bias)
    val_preds = np.argmax(val_logits, axis=1)
    val_acc = float(accuracy_score(y_val, val_preds))
    
    history["train_loss"].append(train_loss)
    history["val_accuracy"].append(val_acc)
    
    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = (np.array(quantum_weights), np.array(readout_weights), np.array(bias))
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

qw_best, rw_best, b_best = best_weights
save_quantum_model(MODEL_PATH, qw_best, rw_best, b_best)
with open(HISTORY_PATH, "w") as f:
    json.dump(history, f, indent=2)

print(f"\nBest Validation Accuracy Saved: {best_val_acc * 100:.2f}%")