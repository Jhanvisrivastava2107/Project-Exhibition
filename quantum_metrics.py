import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# -----------------------------
# LOAD RESULTS
# -----------------------------

y_test = np.load("y_test.npy")
predictions = np.load(
    "quantum_predictions.npy"
)


# -----------------------------
# METRICS
# -----------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


# -----------------------------
# DISPLAY
# -----------------------------

print("\n================================")
print("QUANTUM MODEL RESULTS")
print("================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Healthy",
            "Early Blight",
            "Late Blight",
            "Leaf Mold"
        ],
        zero_division=0
    )
)