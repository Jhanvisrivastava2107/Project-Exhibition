from pathlib import Path
import json
import numpy as np


ROOT = Path(__file__).resolve().parent


def load_npy(filename):
    path = ROOT / filename

    if not path.exists():
        print(f"[MISSING] {filename}")
        return None

    arr = np.load(path, allow_pickle=True)

    print(f"\n{filename}")
    print("-" * 50)
    print("Shape:", arr.shape)
    print("Dtype:", arr.dtype)

    if arr.size > 0:
        print("Min:", np.min(arr))
        print("Max:", np.max(arr))

    return arr


print("=" * 70)
print("QUANTUM / TEST DATA CHECK")
print("=" * 70)


# ---------------------------------------------------------
# Class names
# ---------------------------------------------------------

class_file = ROOT / "artifacts" / "class_names.json"

if class_file.exists():

    with open(class_file, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    print("\nCLASS NAMES")
    print("-" * 50)

    if isinstance(class_names, dict):
        for key, value in class_names.items():
            print(f"{key}: {value}")
        print("Number of classes:", len(class_names))

    elif isinstance(class_names, list):
        for i, name in enumerate(class_names):
            print(f"{i}: {name}")
        print("Number of classes:", len(class_names))

else:
    print("\n[MISSING] artifacts/class_names.json")


# ---------------------------------------------------------
# Test labels
# ---------------------------------------------------------

y_test = load_npy("y_test.npy")


if y_test is not None:

    print("\nTEST LABELS")
    print("-" * 50)

    unique, counts = np.unique(y_test, return_counts=True)

    for label, count in zip(unique, counts):
        print(f"Class {label}: {count} samples")

    print("Number of unique labels:", len(unique))


# ---------------------------------------------------------
# Test PCA features
# ---------------------------------------------------------

X_test = load_npy("X_test_pca.npy")


# ---------------------------------------------------------
# Quantum predictions
# ---------------------------------------------------------

quantum_predictions = load_npy("quantum_predictions.npy")


# ---------------------------------------------------------
# Quantum weights
# ---------------------------------------------------------

quantum_weights = load_npy("quantum_weights.npy")


# ---------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("CONSISTENCY CHECKS")
print("=" * 70)


if y_test is not None:

    print("\nTest samples:", len(y_test))

if X_test is not None:

    print("X_test samples:", len(X_test))

    if y_test is not None:
        print(
            "X_test / y_test:",
            "OK" if len(X_test) == len(y_test) else "MISMATCH"
        )


if quantum_predictions is not None:

    print("\nQuantum prediction shape:", quantum_predictions.shape)

    if y_test is not None:

        if len(quantum_predictions) == len(y_test):
            print("Quantum predictions / y_test: OK")
        else:
            print("Quantum predictions / y_test: MISMATCH")


if y_test is not None:

    unique_labels = np.unique(y_test)

    print("\nLabels found in y_test:")

    for label in unique_labels:
        print(label)

    if len(unique_labels) == 9:
        print("\n9-class test set detected.")

    else:
        print(
            f"\nWARNING: y_test contains {len(unique_labels)} classes, "
            "not 9."
        )


print("\n")
print("=" * 70)
print("CHECK COMPLETE")
print("=" * 70)