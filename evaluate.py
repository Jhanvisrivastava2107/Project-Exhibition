from pathlib import Path
import json
import time

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

from quantum_model import (
    load_quantum_model,
    predict as quantum_predict,
    predict_proba as quantum_predict_proba,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

FEATURE_DIR = PROJECT_ROOT / "features"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

X_TEST_FILE = FEATURE_DIR / "X_test_pca.npy"
Y_TEST_FILE = FEATURE_DIR / "y_test.npy"

CLASS_NAMES_FILE = ARTIFACT_DIR / "class_names.json"

SVM_FILE = ARTIFACT_DIR / "svm.joblib"
QUANTUM_FILE = ARTIFACT_DIR / "quantum_model.npz"
QUANTUM_SCALER_FILE = ARTIFACT_DIR / "quantum_scaler.joblib"

SVM_CM_FILE = OUTPUT_DIR / "confusion_matrix_svm.png"
VQC_CM_FILE = OUTPUT_DIR / "confusion_matrix_vqc.png"
METRICS_FILE = OUTPUT_DIR / "final_metrics.json"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 5: FINAL MODEL EVALUATION")
print("=" * 70)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    X_TEST_FILE,
    Y_TEST_FILE,
    CLASS_NAMES_FILE,
    SVM_FILE,
    QUANTUM_FILE,
    QUANTUM_SCALER_FILE,
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    raise FileNotFoundError(
        "The following required files are missing:\n"
        + "\n".join(f"  - {file}" for file in missing_files)
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test = np.load(X_TEST_FILE)
y_test = np.load(Y_TEST_FILE)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("\nClasses:")
for index, name in enumerate(class_names):
    print(f"  {index}: {name}")


# ============================================================
# VALIDATE DATA
# ============================================================

if X_test.ndim != 2:
    raise ValueError(
        f"Expected 2D PCA test features, got shape {X_test.shape}"
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "The number of test feature rows does not match "
        "the number of test labels."
    )

if X_test.shape[1] != 8:
    raise ValueError(
        f"Expected 8 PCA features for the VQC, but received {X_test.shape[1]}."
    )

if len(class_names) != 9:
    raise ValueError(f"Expected 9 class names, got {len(class_names)}.")

if not np.all(np.isin(y_test, np.arange(len(class_names)))):
    raise ValueError("Test labels contain class indices outside the class-name list.")


# ============================================================
# LOAD SVM
# ============================================================

print("\nLoading SVM model...")
svm_model = joblib.load(SVM_FILE)


# ============================================================
# LOAD QUANTUM MODEL AND QUANTUM SCALER
# ============================================================

print("Loading quantum model...")
quantum_weights, readout_weights, bias = load_quantum_model(QUANTUM_FILE)

print("Loading fitted quantum scaler...")
quantum_scaler = joblib.load(QUANTUM_SCALER_FILE)

if not hasattr(quantum_scaler, "n_features_in_"):
    raise ValueError(
        "The quantum scaler does not expose n_features_in_. "
        "Confirm that quantum_scaler.joblib is a fitted scikit-learn scaler."
    )

if quantum_scaler.n_features_in_ != X_test.shape[1]:
    raise ValueError(
        "Quantum scaler input dimension does not match X_test. "
        f"Scaler expects {quantum_scaler.n_features_in_} features, "
        f"but X_test has {X_test.shape[1]}."
    )


# ============================================================
# SVM PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("SVM EVALUATION")
print("=" * 70)

svm_start = time.perf_counter()

svm_predictions = svm_model.predict(X_test)
svm_probabilities = svm_model.predict_proba(X_test)

svm_time = time.perf_counter() - svm_start


# ============================================================
# VQC PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("VQC EVALUATION")
print("=" * 70)

print("\nApplying the fitted quantum scaler to PCA test features...")

# X_test already contains PCA-transformed features.
# This scaler must be the same fitted scaler used during VQC training.
# Do not fit or refit a scaler on the test set.
X_test_quantum = quantum_scaler.transform(X_test)

print(f"Quantum test feature shape: {X_test_quantum.shape}")
print("\nRunning quantum circuit on test set...")
print("This may take some time because the circuit is simulated classically.")

vqc_start = time.perf_counter()

vqc_predictions = quantum_predict(
    X_test_quantum,
    quantum_weights,
    readout_weights,
    bias,
)

vqc_probabilities = quantum_predict_proba(
    X_test_quantum,
    quantum_weights,
    readout_weights,
    bias,
)

vqc_time = time.perf_counter() - vqc_start


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# ============================================================
# CALCULATE METRICS
# ============================================================

svm_metrics = calculate_metrics(y_test, svm_predictions)
vqc_metrics = calculate_metrics(y_test, vqc_predictions)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nSVM Results:")
for metric, value in svm_metrics.items():
    print(f"{metric:20s}: {value:.4f}")
print(f"{'Inference time (s)':20s}: {svm_time:.4f}")

print("\nVQC Results:")
for metric, value in vqc_metrics.items():
    print(f"{metric:20s}: {value:.4f}")
print(f"{'Inference time (s)':20s}: {vqc_time:.4f}")


# ============================================================
# CLASSIFICATION REPORTS
# ============================================================

print("\n" + "=" * 70)
print("SVM CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        svm_predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
)

print("\n" + "=" * 70)
print("VQC CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        vqc_predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# SVM CONFUSION MATRIX
# ============================================================

svm_cm = confusion_matrix(
    y_test,
    svm_predictions,
    labels=np.arange(len(class_names)),
)

fig, ax = plt.subplots(figsize=(12, 9))

ConfusionMatrixDisplay(
    confusion_matrix=svm_cm,
    display_labels=class_names,
).plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)

ax.set_title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig(SVM_CM_FILE, dpi=300, bbox_inches="tight")
plt.close(fig)


# ============================================================
# VQC CONFUSION MATRIX
# ============================================================

vqc_cm = confusion_matrix(
    y_test,
    vqc_predictions,
    labels=np.arange(len(class_names)),
)

fig, ax = plt.subplots(figsize=(12, 9))

ConfusionMatrixDisplay(
    confusion_matrix=vqc_cm,
    display_labels=class_names,
).plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Purples",
    values_format="d",
    colorbar=False,
)

ax.set_title("VQC Confusion Matrix")
plt.tight_layout()
plt.savefig(VQC_CM_FILE, dpi=300, bbox_inches="tight")
plt.close(fig)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

final_metrics = {
    "svm": {
        **svm_metrics,
        "inference_time_seconds": float(svm_time),
    },
    "vqc": {
        **vqc_metrics,
        "inference_time_seconds": float(vqc_time),
    },
    "num_test_samples": int(len(y_test)),
    "num_classes": int(len(class_names)),
    "class_names": class_names,
}

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=4)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nCreated:")
print(f"  {SVM_CM_FILE}")
print(f"  {VQC_CM_FILE}")
print(f"  {METRICS_FILE}")

print("\nBoth models were evaluated on the same test set.")
print("\nProject pipeline:")
print("  MobileNetV2 -> StandardScaler -> PCA -> SVM")
print("  MobileNetV2 -> StandardScaler -> PCA -> QuantumScaler -> VQC")
