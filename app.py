from pathlib import Path
import json

import joblib
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

CLASS_NAMES_FILE = ARTIFACT_DIR / "class_names.json"
SVM_FILE = ARTIFACT_DIR / "svm.joblib"
SCALER_FILE = ARTIFACT_DIR / "scaler.joblib"
PCA_FILE = ARTIFACT_DIR / "pca.joblib"
QUANTUM_FILE = ARTIFACT_DIR / "quantum_model.npz"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Quantum Plant Disease Detector",
    page_icon="🌿",
    layout="centered",
)


# ============================================================
# UI
# ============================================================

st.title("🌿 Quantum-Based Plant Disease Detector")

st.write(
    "Hybrid classical-quantum classification "
    "of tomato leaf diseases."
)


# ============================================================
# CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


class_names = load_class_names()


# ============================================================
# MOBILENETV2
# ============================================================

@st.cache_resource
def load_feature_extractor():

    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2

    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )

    model.trainable = False

    return model


# ============================================================
# CLASSICAL MODELS
# ============================================================

@st.cache_resource
def load_classical_models():

    scaler = joblib.load(SCALER_FILE)
    pca = joblib.load(PCA_FILE)
    svm = joblib.load(SVM_FILE)

    return scaler, pca, svm


# ============================================================
# QUANTUM MODEL
# ============================================================

@st.cache_resource
def load_vqc():

    from quantum_model import load_quantum_model

    return load_quantum_model(QUANTUM_FILE)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"],
)


# ============================================================
# WAIT FOR IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a tomato leaf image to start prediction."
    )

    st.divider()

    st.subheader("Model Information")

    st.write(f"Number of classes: {len(class_names)}")
    st.write("Feature extractor: MobileNetV2")
    st.write("PCA components: 8")
    st.write("Quantum qubits: 8")
    st.write("Quantum layers: 2")
    st.write("Classical baseline: RBF SVM")

    st.stop()


# ============================================================
# DISPLAY IMAGE
# ============================================================

image = Image.open(uploaded_file).convert("RGB")

st.image(
    image,
    caption="Uploaded leaf image",
    use_container_width=True,
)


# ============================================================
# LOAD MODELS ONLY NOW
# ============================================================

with st.spinner("Loading AI models..."):

    feature_extractor = load_feature_extractor()

    scaler, pca, svm_model = load_classical_models()

    quantum_weights, readout_weights, bias = load_vqc()


# ============================================================
# PREPROCESS IMAGE
# ============================================================

with st.spinner("Analyzing leaf image..."):

    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input,
    )

    resized_image = image.resize((224, 224))

    image_array = np.asarray(
        resized_image,
        dtype=np.float32,
    )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    image_array = preprocess_input(image_array)


    # ========================================================
    # MOBILENET FEATURES
    # ========================================================

    features = feature_extractor.predict(
        image_array,
        verbose=0,
    )


    # ========================================================
    # SCALER
    # ========================================================

    scaled_features = scaler.transform(features)


    # ========================================================
    # PCA
    # ========================================================

    pca_features = pca.transform(
        scaled_features,
    )


    # ========================================================
    # SVM
    # ========================================================

    svm_probabilities = svm_model.predict_proba(
        pca_features
    )[0]

    svm_prediction = int(
        np.argmax(svm_probabilities)
    )


    # ========================================================
    # VQC
    # ========================================================

    from quantum_model import predict_proba as quantum_predict_proba

    vqc_probabilities = quantum_predict_proba(
        pca_features,
        quantum_weights,
        readout_weights,
        bias,
    )[0]

    vqc_prediction = int(
        np.argmax(vqc_probabilities)
    )


# ============================================================
# RESULTS
# ============================================================

st.divider()

st.subheader("Prediction Results")


# ============================================================
# SVM RESULT
# ============================================================

st.markdown("### Classical SVM")

st.success(
    f"Prediction: **{class_names[svm_prediction]}**"
)


# ============================================================
# VQC RESULT
# ============================================================

st.markdown("### Variational Quantum Classifier")

st.success(
    f"Prediction: **{class_names[vqc_prediction]}**"
)


# ============================================================
# TOP 3 SVM
# ============================================================

st.subheader("SVM Top 3 Predictions")

svm_top3 = np.argsort(
    svm_probabilities
)[::-1][:3]

for index in svm_top3:

    probability = float(
        svm_probabilities[index]
    )

    st.write(
        f"**{class_names[index]}**: "
        f"{probability * 100:.2f}%"
    )

    st.progress(probability)


# ============================================================
# TOP 3 VQC
# ============================================================

st.subheader("VQC Top 3 Predictions")

vqc_top3 = np.argsort(
    vqc_probabilities
)[::-1][:3]

for index in vqc_top3:

    probability = float(
        vqc_probabilities[index]
    )

    st.write(
        f"**{class_names[index]}**: "
        f"{probability * 100:.2f}%"
    )

    st.progress(probability)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model Information")

    st.write(
        f"Number of classes: {len(class_names)}"
    )

    st.write(
        "Feature extractor: MobileNetV2"
    )

    st.write(
        "PCA components: 8"
    )

    st.write(
        "Quantum qubits: 8"
    )

    st.write(
        "Quantum layers: 2"
    )

    st.write(
        "Classical baseline: RBF SVM"
    )