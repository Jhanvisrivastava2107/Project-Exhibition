from pathlib import Path
import json
import joblib


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


print("=" * 70)
print("MODEL ARTIFACT CHECK")
print("=" * 70)


# ---------------------------------------------------------
# CLASS NAMES
# ---------------------------------------------------------

class_file = ARTIFACTS / "class_names.json"

with open(
    class_file,
    "r",
    encoding="utf-8"
) as f:

    class_names = json.load(f)


if isinstance(class_names, dict):

    class_names = [
        class_names[str(i)] if str(i) in class_names
        else class_names[i]
        for i in range(len(class_names))
    ]


print("\nCLASS NAMES")
print("-" * 70)

for i, name in enumerate(class_names):

    print(f"{i}: {name}")

print(
    "\nNumber of classes:",
    len(class_names)
)


# ---------------------------------------------------------
# SCALER
# ---------------------------------------------------------

scaler_path = ARTIFACTS / "scaler.joblib"

if scaler_path.exists():

    scaler = joblib.load(scaler_path)

    print("\nSCALER")
    print("-" * 70)

    print(
        "Type:",
        type(scaler).__name__
    )

    print(
        "Expected input features:",
        getattr(
            scaler,
            "n_features_in_",
            "unknown"
        )
    )

else:

    print(
        "\n[MISSING] scaler.joblib"
    )


# ---------------------------------------------------------
# PCA
# ---------------------------------------------------------

pca_path = ARTIFACTS / "pca.joblib"

if pca_path.exists():

    pca = joblib.load(pca_path)

    print("\nPCA")
    print("-" * 70)

    print(
        "Type:",
        type(pca).__name__
    )

    print(
        "Input features:",
        getattr(
            pca,
            "n_features_in_",
            "unknown"
        )
    )

    print(
        "Output components:",
        getattr(
            pca,
            "n_components_",
            getattr(
                pca,
                "n_components",
                "unknown"
            )
        )
    )

else:

    print(
        "\n[MISSING] pca.joblib"
    )


# ---------------------------------------------------------
# SVM
# ---------------------------------------------------------

svm_path = ARTIFACTS / "svm.joblib"

if svm_path.exists():

    svm = joblib.load(svm_path)

    print("\nSVM")
    print("-" * 70)

    print(
        "Type:",
        type(svm).__name__
    )

    print(
        "Input features:",
        getattr(
            svm,
            "n_features_in_",
            "unknown"
        )
    )

    print(
        "Classes:",
        svm.classes_
    )

    print(
        "Number of classes:",
        len(svm.classes_)
    )

else:

    print(
        "\n[MISSING] svm.joblib"
    )


print("\n")
print("=" * 70)
print("CHECK COMPLETE")
print("=" * 70)