from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

FEATURE_DIR = PROJECT_ROOT / "features"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

X_TRAIN_FILE = FEATURE_DIR / "X_train.npy"
Y_TRAIN_FILE = FEATURE_DIR / "y_train.npy"
X_TEST_FILE = FEATURE_DIR / "X_test.npy"
Y_TEST_FILE = FEATURE_DIR / "y_test.npy"

SCALER_FILE = ARTIFACT_DIR / "scaler.joblib"
PCA_FILE = ARTIFACT_DIR / "pca.joblib"
SVM_FILE = ARTIFACT_DIR / "svm.joblib"

X_TRAIN_PCA_FILE = FEATURE_DIR / "X_train_pca.npy"
X_TEST_PCA_FILE = FEATURE_DIR / "X_test_pca.npy"


PCA_COMPONENTS = 8


# ============================================================
# CREATE DIRECTORIES
# ============================================================

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 3: CLASSICAL MACHINE LEARNING BASELINE")
print("=" * 70)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading MobileNetV2 features...")

X_train = np.load(X_TRAIN_FILE)
y_train = np.load(Y_TRAIN_FILE)

X_test = np.load(X_TEST_FILE)
y_test = np.load(Y_TEST_FILE)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")


# ============================================================
# STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZATION")
print("=" * 70)

print("\nFitting StandardScaler ONLY on training data...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Standardization complete.")


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(scaler, SCALER_FILE)

print(f"\nScaler saved to:")
print(SCALER_FILE)


# ============================================================
# PCA
# ============================================================

print("\n" + "=" * 70)
print("PCA DIMENSIONALITY REDUCTION")
print("=" * 70)

print(f"\nOriginal feature dimension: {X_train.shape[1]}")
print(f"PCA components: {PCA_COMPONENTS}")

pca = PCA(
    n_components=PCA_COMPONENTS,
    random_state=42
)

X_train_pca = pca.fit_transform(X_train_scaled)

X_test_pca = pca.transform(X_test_scaled)


print(f"\nReduced training shape: {X_train_pca.shape}")
print(f"Reduced testing shape : {X_test_pca.shape}")

explained_variance = np.sum(pca.explained_variance_ratio_)

print(
    f"\nTotal explained variance by "
    f"{PCA_COMPONENTS} components: "
    f"{explained_variance:.4f}"
)


# ============================================================
# SAVE PCA
# ============================================================

joblib.dump(pca, PCA_FILE)

np.save(X_TRAIN_PCA_FILE, X_train_pca)
np.save(X_TEST_PCA_FILE, X_test_pca)

print("\nPCA saved to:")
print(PCA_FILE)

print("\nPCA features saved to:")
print(X_TRAIN_PCA_FILE)
print(X_TEST_PCA_FILE)


# ============================================================
# TRAIN SVM
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MULTICLASS SVM")
print("=" * 70)

print("\nParameters:")
print("Kernel      : RBF")
print("C           : 10")
print("Gamma       : scale")
print("Probability : True")
print("Class weight: balanced")

svm_model = SVC(
    C=10,
    kernel="rbf",
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=42
)

svm_model.fit(X_train_pca, y_train)

print("\nSVM training complete.")


# ============================================================
# SAVE SVM
# ============================================================

joblib.dump(svm_model, SVM_FILE)

print("\nSVM model saved to:")
print(SVM_FILE)


# ============================================================
# TEST PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("SVM TEST EVALUATION")
print("=" * 70)

y_pred = svm_model.predict(X_test_pca)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision_macro = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print(f"\nAccuracy       : {accuracy:.4f}")
print(f"Macro Precision: {precision_macro:.4f}")
print(f"Macro Recall   : {recall_macro:.4f}")
print(f"Macro F1       : {f1_macro:.4f}")
print(f"Weighted F1    : {f1_weighted:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("=" * 70)
print("CLASSICAL MODEL COMPLETE")
print("=" * 70)

print("\nCreated files:")

print("  artifacts/scaler.joblib")
print("  artifacts/pca.joblib")
print("  artifacts/svm.joblib")
print("  features/X_train_pca.npy")
print("  features/X_test_pca.npy")

print("\nNext step:")
print("  python train_quantum.py")