"""Compute accuracy, precision, recall and F1-score for the quantum (VQC) model.

Put this file in the project folder and run:  python quantum_metrics.py
It needs y_test.npy and quantum_predictions.npy (saved by train_quantum.py).
"""
import os
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

y_test = np.load("y_test.npy")
preds = np.load("quantum_predictions.npy")

# If the file holds probabilities / scores (one column per class), take the top class
if preds.ndim == 2:
    preds = np.argmax(preds, axis=1)

y_test = y_test.astype(int).ravel()
preds = preds.astype(int).ravel()

if len(preds) != len(y_test):
    raise SystemExit(
        f"Length mismatch: {len(preds)} predictions vs {len(y_test)} test labels. "
        "Re-run train_quantum.py so predictions match the current test split."
    )

# Class names from the training folders, if they match the label count
names = None
if os.path.isdir("splits/train"):
    folders = sorted(d for d in os.listdir("splits/train")
                     if os.path.isdir(os.path.join("splits/train", d)))
    if len(folders) == len(np.unique(np.concatenate([y_test, preds]))):
        names = folders

acc = accuracy_score(y_test, preds)
print("=== VQC (quantum) results on the test set ===")
print(f"Test samples : {len(y_test)}")
print(f"Accuracy     : {acc * 100:.2f}%")
for avg in ("weighted", "macro"):
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, preds, average=avg, zero_division=0)
    print(f"{avg.capitalize():9s} -> Precision: {p * 100:.2f}%  "
          f"Recall: {r * 100:.2f}%  F1: {f1 * 100:.2f}%")

print("\nPer-class report:")
print(classification_report(y_test, preds, target_names=names, zero_division=0))
print("Confusion matrix (rows = true, columns = predicted):")
print(confusion_matrix(y_test, preds))
