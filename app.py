
from pathlib import Path
import json

import joblib
import numpy as np
import streamlit as st
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from quantum_model import (
    predict_proba as quantum_predict_proba,
    load_quantum_model,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tomato Leaf Disease Classification",
    page_icon="🍅",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "artifacts"

CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.json"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
PCA_PATH = ARTIFACT_DIR / "pca.joblib"
SVM_PATH = ARTIFACT_DIR / "svm.joblib"
QUANTUM_MODEL_PATH = ARTIFACT_DIR / "quantum_model.npz"

# This scaler must be fitted on PCA-transformed training data
# and used when training the VQC.
QUANTUM_SCALER_PATH = ARTIFACT_DIR / "quantum_scaler.joblib"


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    CLASS_NAMES_PATH,
    SCALER_PATH,
    PCA_PATH,
    SVM_PATH,
    QUANTUM_MODEL_PATH,
    QUANTUM_SCALER_PATH,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    st.error("Required model files are missing:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.warning(
        "The quantum scaler must be created during quantum "
        "training. Do not create it by fitting a scaler on "
        "test images or uploaded images."
    )

    st.stop()


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

if len(class_names) != 9:
    st.error(
        f"Expected 9 classes, but found {len(class_names)} "
        "in class_names.json."
    )
    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    # --------------------------------------------------------
    # MobileNetV2 feature extractor
    # --------------------------------------------------------

    feature_extractor = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )

    feature_extractor.trainable = False

    # --------------------------------------------------------
    # Existing SVM preprocessing
    # --------------------------------------------------------

    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    svm_model = joblib.load(SVM_PATH)

    # --------------------------------------------------------
    # Quantum preprocessing
    # --------------------------------------------------------

    quantum_scaler = joblib.load(QUANTUM_SCALER_PATH)

    # --------------------------------------------------------
    # Quantum model
    # --------------------------------------------------------

    (
        quantum_weights,
        readout_weights,
        bias,
    ) = load_quantum_model(QUANTUM_MODEL_PATH)

    # --------------------------------------------------------
    # Validate preprocessing dimensions
    # --------------------------------------------------------

    if pca.n_components_ != 8:
        raise ValueError(
            f"Expected 8 PCA components, got {pca.n_components_}."
        )

    if quantum_scaler.n_features_in_ != 8:
        raise ValueError(
            "The quantum scaler must expect 8 PCA features. "
            f"It expects {quantum_scaler.n_features_in_}."
        )

    if quantum_weights.shape[1] != 8:
        raise ValueError(
            "Quantum model configuration does not match "
            "the expected 8 qubits."
        )

    if readout_weights.shape[0] != len(class_names):
        raise ValueError(
            "Quantum readout output dimension does not match "
            "the number of classes."
        )

    return (
        feature_extractor,
        scaler,
        pca,
        quantum_scaler,
        svm_model,
        quantum_weights,
        readout_weights,
        bias,
    )


try:
    (
        feature_extractor,
        scaler,
        pca,
        quantum_scaler,
        svm_model,
        quantum_weights,
        readout_weights,
        bias,
    ) = load_models()

except Exception as error:
    st.error(f"Could not load the models: {error}")
    st.stop()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image):
    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to MobileNetV2 input dimensions
    image = image.resize((224, 224))

    # Convert to NumPy array
    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    # Apply MobileNetV2 preprocessing
    image_array = preprocess_input(image_array)

    # Extract 1280-dimensional features
    features = feature_extractor.predict(
        image_array,
        verbose=0,
    )

    return features


# ============================================================
# CLASSICAL + QUANTUM PREDICTION
# ============================================================

def predict_image(image):
    # --------------------------------------------------------
    # MobileNetV2
    # --------------------------------------------------------

    features = extract_features(image)

    # --------------------------------------------------------
    # Existing StandardScaler
    # --------------------------------------------------------

    scaled_features = scaler.transform(features)

    # --------------------------------------------------------
    # Existing PCA
    # --------------------------------------------------------

    pca_features = pca.transform(scaled_features)

    # --------------------------------------------------------
    # SVM
    # SVM continues to use PCA features directly.
    # --------------------------------------------------------

    svm_probabilities = svm_model.predict_proba(
        pca_features
    )[0]

    svm_prediction = int(
        np.argmax(svm_probabilities)
    )

    # --------------------------------------------------------
    # VQC
    # Apply the quantum scaler fitted during quantum training.
    # --------------------------------------------------------

    quantum_features = quantum_scaler.transform(
        pca_features
    )

    vqc_probabilities = quantum_predict_proba(
        quantum_features,
        quantum_weights,
        readout_weights,
        bias,
    )[0]

    vqc_prediction = int(
        np.argmax(vqc_probabilities)
    )

    return (
        svm_prediction,
        svm_probabilities,
        vqc_prediction,
        vqc_probabilities,
    )


# ============================================================
# HEADER
# ============================================================

st.title("🍅 Tomato Leaf Disease Classification")

st.markdown(
    """
### Classical SVM vs Variational Quantum Classifier

Upload a tomato leaf image to compare predictions from
the classical SVM and the simulated quantum VQC.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Project Information")

    st.write("**Dataset:** Tomato Leaf Disease")
    st.write(f"**Number of classes:** {len(class_names)}")

    st.divider()

    st.subheader("Feature Pipeline")

    st.write("MobileNetV2")
    st.write("↓")
    st.write("StandardScaler")
    st.write("↓")
    st.write("PCA → 8 features")

    st.divider()

    st.subheader("Classical Model")
    st.write("RBF SVM")

    st.caption(
        "Display a measured test accuracy here only after "
        "confirming it from your final evaluation results."
    )

    st.divider()

    st.subheader("Quantum Model")
    st.write("8-qubit VQC")
    st.write("2 quantum layers")
    st.write("9 output classes")

    st.caption(
        "The VQC is simulated using a classical "
        "quantum simulator."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"],
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # ----------------------------------------------------
        # Display uploaded image
        # ----------------------------------------------------

        st.subheader("Uploaded Image")
        st.image(image, width=400)

        # ----------------------------------------------------
        # Run both models
        # ----------------------------------------------------

        with st.spinner(
            "Running SVM and VQC predictions..."
        ):
            (
                svm_prediction,
                svm_probabilities,
                vqc_prediction,
                vqc_probabilities,
            ) = predict_image(image)

        # ----------------------------------------------------
        # Resolve predicted class names
        # ----------------------------------------------------

        svm_class = class_names[svm_prediction]
        vqc_class = class_names[vqc_prediction]

        svm_probability = float(
            svm_probabilities[svm_prediction]
        )

        vqc_probability = float(
            vqc_probabilities[vqc_prediction]
        )

        # ====================================================
        # MAIN RESULTS
        # ====================================================

        st.divider()
        st.header("Prediction Results")

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # SVM result
        # ----------------------------------------------------

        with col1:
            st.subheader("🔵 Classical SVM")
            st.success(svm_class)

            st.metric(
                "Predicted probability",
                f"{svm_probability * 100:.2f}%",
            )

        # ----------------------------------------------------
        # VQC result
        # ----------------------------------------------------

        with col2:
            st.subheader("🟣 Quantum VQC")
            st.info(vqc_class)

            st.metric(
                "Predicted probability",
                f"{vqc_probability * 100:.2f}%",
            )

        # ====================================================
        # MODEL AGREEMENT
        # ====================================================

        st.divider()
        st.header("Model Comparison")

        if svm_prediction == vqc_prediction:
            st.success(
                "Both models predicted the same class."
            )
        else:
            st.warning(
                "The models predicted different classes."
            )

        # ====================================================
        # TOP-3 PREDICTIONS
        # ====================================================

        st.divider()
        st.header("Top-3 Predictions")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("SVM")

            svm_top3 = np.argsort(
                svm_probabilities
            )[::-1][:3]

            for rank, index in enumerate(
                svm_top3,
                start=1,
            ):
                st.write(
                    f"**{rank}. {class_names[index]}** — "
                    f"{svm_probabilities[index] * 100:.2f}%"
                )

        with col2:
            st.subheader("VQC")

            vqc_top3 = np.argsort(
                vqc_probabilities
            )[::-1][:3]

            for rank, index in enumerate(
                vqc_top3,
                start=1,
            ):
                st.write(
                    f"**{rank}. {class_names[index]}** — "
                    f"{vqc_probabilities[index] * 100:.2f}%"
                )

        # ====================================================
        # TECHNICAL DETAILS
        # ====================================================

        st.divider()

        with st.expander("Technical Details"):
            st.write("**Input image:** 224 × 224 RGB")
            st.write("**Feature extractor:** MobileNetV2")
            st.write("**CNN features:** 1280")
            st.write("**StandardScaler:** Applied")
            st.write("**PCA components:** 8")
            st.write("**SVM kernel:** RBF")
            st.write("**VQC qubits:** 8")
            st.write("**VQC layers:** 2")
            st.write("**VQC classes:** 9")
            st.write(
                "**Quantum scaler:** Applied to PCA features"
            )
            st.write(
                "**Quantum backend:** PennyLane "
                "`default.qubit` simulator"
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
