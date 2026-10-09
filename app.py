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
    load_quantum_model
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tomato Leaf Disease Classification",
    page_icon="🍅",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "artifacts"
FEATURE_DIR = BASE_DIR / "features"


CLASS_NAMES_PATH = (
    ARTIFACT_DIR / "class_names.json"
)

SCALER_PATH = (
    ARTIFACT_DIR / "scaler.joblib"
)

PCA_PATH = (
    ARTIFACT_DIR / "pca.joblib"
)

SVM_PATH = (
    ARTIFACT_DIR / "svm.joblib"
)

QUANTUM_MODEL_PATH = (
    ARTIFACT_DIR / "quantum_model.npz"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    CLASS_NAMES_PATH,
    SCALER_PATH,
    PCA_PATH,
    SVM_PATH,
    QUANTUM_MODEL_PATH
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:

    st.error(
        "The following required files are missing:"
    )

    for file in missing_files:
        st.write(f"- {file}")

    st.stop()


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r"
) as file:

    class_names = json.load(
        file
    )


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
        input_shape=(
            224,
            224,
            3
        )
    )

    feature_extractor.trainable = False


    # --------------------------------------------------------
    # Classical preprocessing
    # --------------------------------------------------------

    scaler = joblib.load(
        SCALER_PATH
    )

    pca = joblib.load(
        PCA_PATH
    )


    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    svm_model = joblib.load(
        SVM_PATH
    )


    # --------------------------------------------------------
    # VQC
    # --------------------------------------------------------

    (
        quantum_weights,
        readout_weights,
        bias
    ) = load_quantum_model(
        QUANTUM_MODEL_PATH
    )


    return (
        feature_extractor,
        scaler,
        pca,
        svm_model,
        quantum_weights,
        readout_weights,
        bias
    )


(
    feature_extractor,
    scaler,
    pca,
    svm_model,
    quantum_weights,
    readout_weights,
    bias
) = load_models()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    image
):

    # Convert to RGB
    image = image.convert(
        "RGB"
    )

    # Resize
    image = image.resize(
        (224, 224)
    )

    # Convert to numpy array
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # MobileNetV2 preprocessing
    image_array = preprocess_input(
        image_array
    )

    # Extract 1280-dimensional features
    features = feature_extractor.predict(
        image_array,
        verbose=0
    )

    return features


# ============================================================
# CLASSICAL + QUANTUM PREDICTION
# ============================================================

def predict_image(
    image
):

    # --------------------------------------------------------
    # MobileNetV2
    # --------------------------------------------------------

    features = extract_features(
        image
    )


    # --------------------------------------------------------
    # StandardScaler
    # --------------------------------------------------------

    scaled_features = scaler.transform(
        features
    )


    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca_features = pca.transform(
        scaled_features
    )


    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    svm_probabilities = (
        svm_model.predict_proba(
            pca_features
        )[0]
    )

    svm_prediction = int(
        np.argmax(
            svm_probabilities
        )
    )


    # --------------------------------------------------------
    # VQC
    # --------------------------------------------------------

    vqc_probabilities = (
        quantum_predict_proba(
            pca_features,
            quantum_weights,
            readout_weights,
            bias
        )[0]
    )

    vqc_prediction = int(
        np.argmax(
            vqc_probabilities
        )
    )


    return (
        svm_prediction,
        svm_probabilities,
        vqc_prediction,
        vqc_probabilities
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🍅 Tomato Leaf Disease Classification"
)

st.markdown(
    """
### Classical SVM vs Variational Quantum Classifier

Upload a tomato leaf image to compare predictions
from the classical SVM and the simulated quantum VQC.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Project Information"
    )

    st.write(
        "**Dataset:** Tomato Leaf Disease"
    )

    st.write(
        "**Number of classes:** 9"
    )

    st.write(
        "**Test images:** 900"
    )

    st.write(
        "**Images per class:** 100"
    )

    st.divider()

    st.subheader(
        "Feature Pipeline"
    )

    st.write(
        "MobileNetV2"
    )

    st.write(
        "↓"
    )

    st.write(
        "StandardScaler"
    )

    st.write(
        "↓"
    )

    st.write(
        "PCA → 8 features"
    )

    st.divider()

    st.subheader(
        "Classical Model"
    )

    st.write(
        "RBF SVM"
    )

    st.write(
        "**Test Accuracy: 72.11%**"
    )

    st.divider()

    st.subheader(
        "Quantum Model"
    )

    st.write(
        "8-qubit VQC"
    )

    st.write(
        "2 quantum layers"
    )

    st.write(
        "**Test Accuracy: 9.11%**"
    )

    st.caption(
        "The VQC is simulated using a classical "
        "quantum simulator."
    )


# ============================================================
# UPLOAD IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )


    # --------------------------------------------------------
    # Display image
    # --------------------------------------------------------

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        width=400
    )


    # --------------------------------------------------------
    # Run models
    # --------------------------------------------------------

    with st.spinner(
        "Running SVM and VQC predictions..."
    ):

        (
            svm_prediction,
            svm_probabilities,
            vqc_prediction,
            vqc_probabilities
        ) = predict_image(
            image
        )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    svm_class = class_names[
        svm_prediction
    ]

    vqc_class = class_names[
        vqc_prediction
    ]

    svm_probability = (
        svm_probabilities[
            svm_prediction
        ]
    )

    vqc_probability = (
        vqc_probabilities[
            vqc_prediction
        ]
    )


    # ========================================================
    # MAIN RESULTS
    # ========================================================

    st.divider()

    st.header(
        "Prediction Results"
    )


    col1, col2 = st.columns(
        2
    )


    # --------------------------------------------------------
    # SVM RESULT
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "🔵 Classical SVM"
        )

        st.success(
            svm_class
        )

        st.metric(
            "Predicted probability",
            f"{svm_probability * 100:.2f}%"
        )

        st.caption(
            "Test accuracy: 72.11%"
        )


    # --------------------------------------------------------
    # VQC RESULT
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "🟣 Quantum VQC"
        )

        st.info(
            vqc_class
        )

        st.metric(
            "Predicted probability",
            f"{vqc_probability * 100:.2f}%"
        )

        st.caption(
            "Test accuracy: 9.11%"
        )


    # ========================================================
    # AGREEMENT
    # ========================================================

    st.divider()

    st.header(
        "Model Comparison"
    )

    if svm_prediction == vqc_prediction:

        st.success(
            "✓ Both models predicted the same class."
        )

    else:

        st.warning(
            "⚠ The models predicted different classes."
        )


    # ========================================================
    # TOP-3 PREDICTIONS
    # ========================================================

    st.divider()

    st.header(
        "Top-3 Predictions"
    )


    col1, col2 = st.columns(
        2
    )


    # --------------------------------------------------------
    # SVM TOP 3
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "SVM"
        )

        svm_top3 = np.argsort(
            svm_probabilities
        )[::-1][:3]

        for rank, index in enumerate(
            svm_top3,
            start=1
        ):

            st.write(
                f"**{rank}. "
                f"{class_names[index]}** — "
                f"{svm_probabilities[index] * 100:.2f}%"
            )


    # --------------------------------------------------------
    # VQC TOP 3
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "VQC"
        )

        vqc_top3 = np.argsort(
            vqc_probabilities
        )[::-1][:3]

        for rank, index in enumerate(
            vqc_top3,
            start=1
        ):

            st.write(
                f"**{rank}. "
                f"{class_names[index]}** — "
                f"{vqc_probabilities[index] * 100:.2f}%"
            )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    st.divider()

    with st.expander(
        "Technical Details"
    ):

        st.write(
            "**Input image:** 224 × 224 RGB"
        )

        st.write(
            "**Feature extractor:** MobileNetV2"
        )

        st.write(
            "**CNN features:** 1280"
        )

        st.write(
            "**StandardScaler:** Applied"
        )

        st.write(
            "**PCA components:** 8"
        )

        st.write(
            "**SVM kernel:** RBF"
        )

        st.write(
            "**VQC qubits:** 8"
        )

        st.write(
            "**VQC layers:** 2"
        )

        st.write(
            "**VQC classes:** 9"
        )

        st.write(
            "**Quantum backend:** PennyLane "
            "`default.qubit` simulator"
        )