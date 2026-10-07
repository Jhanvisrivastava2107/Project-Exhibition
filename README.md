#  Quantum-Based Plant Disease Detector

A hybrid **Classical–Quantum Machine Learning** project for detecting and classifying tomato plant diseases from leaf images. The project combines **Convolutional Neural Network (CNN) feature extraction**, **Principal Component Analysis (PCA)**, and a **Variational Quantum Circuit (VQC)** to investigate the potential of quantum machine learning for plant disease classification.

---

##  Project Overview

Plant diseases can significantly affect crop productivity and quality. Early and accurate identification of diseases can help farmers take timely preventive measures.

This project develops an intelligent image-based plant disease detection system that:

* Processes tomato leaf images.
* Extracts meaningful visual features using **MobileNetV2**.
* Reduces the high-dimensional feature space using **PCA**.
* Encodes the reduced features into a **4-qubit quantum circuit**.
* Uses a **Variational Quantum Circuit (VQC)** for classification.
* Implements a **classical SVM classifier** as a baseline.
* Compares the classical and quantum approaches.

The primary objective is to explore whether a quantum-enhanced classifier can provide a useful alternative to conventional machine learning approaches for plant disease classification.

---

##  Objectives

1. Detect diseases in tomato plant leaves using image-based classification.
2. Extract deep visual features using a pretrained CNN.
3. Reduce feature dimensionality using Principal Component Analysis.
4. Encode the reduced features into quantum states.
5. Develop a Variational Quantum Circuit for classification.
6. Develop a classical SVM model for comparison.
7. Evaluate and compare the performance of the classical and quantum approaches.

---

##  Methodology

The complete workflow follows a hybrid classical–quantum pipeline:

```text
Tomato Leaf Images
        ↓
Dataset Preparation
        ↓
CNN Feature Extraction
(MobileNetV2)
        ↓
Feature Scaling
        ↓
PCA Dimensionality Reduction
        ↓
4 Principal Components
        ↓
Angle Embedding
        ↓
4-Qubit Variational Quantum Circuit
        ↓
Disease Classification
```

A parallel classical pipeline is used for comparison:

```text
CNN Features
     ↓
Feature Scaling
     ↓
SVM Classifier
     ↓
Disease Prediction
```

---

##  Technologies Used

| Technology             | Purpose                                |
| ---------------------- | -------------------------------------- |
| **Python**             | Core programming language              |
| **TensorFlow / Keras** | CNN and MobileNetV2                    |
| **NumPy**              | Numerical computation and data storage |
| **Scikit-learn**       | PCA, SVM, preprocessing and evaluation |
| **PennyLane**          | Quantum circuit simulation and VQC     |
| **MobileNetV2**        | Deep feature extraction                |
| **SVM**                | Classical baseline classifier          |

---

##  Dataset

The project uses tomato leaf images representing multiple healthy and diseased conditions.

The dataset contains the following classes:

1. `bacterial_spot`
2. `early_blight`
3. `healthy`
4. `late_blight`
5. `leaf_mold`
6. `septoria_leaf_spot`
7. `spider_mites`
8. `target_spot`
9. `yellow_leaf_curl_virus`
10. `mosaic_virus`

The dataset is organized into training and testing splits.

---

##  CNN Feature Extraction

A pretrained **MobileNetV2** model with ImageNet weights is used as the feature extractor.

The final classification layer is removed and global average pooling is used to obtain a compact feature representation of each tomato leaf image.

```python
model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

model.trainable = False
```

This allows the project to use powerful pretrained visual representations without training a complete CNN from scratch.

---

##  PCA Dimensionality Reduction

CNN features are high-dimensional and cannot be directly mapped onto a small quantum circuit efficiently.

Therefore, the extracted features are:

1. Standardized using `StandardScaler`.
2. Reduced to **4 principal components** using PCA.

```python
pca = PCA(n_components=4)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
```

The four PCA components are then used as the four input features for the quantum circuit.

---

##  Variational Quantum Circuit

The quantum component uses a **4-qubit Variational Quantum Circuit** implemented using PennyLane.

### Quantum Pipeline

```text
4 PCA Features
      ↓
Angle Embedding
      ↓
RY Trainable Rotations
      ↓
CNOT Entanglement
      ↓
Pauli-Z Measurements
      ↓
Classification Output
```

### Quantum Circuit Components

**1. Angle Embedding**

The four PCA features are encoded into four qubits using Y-axis rotations.

```python
qml.AngleEmbedding(
    features,
    wires=range(n_qubits),
    rotation="Y"
)
```

**2. Trainable RY Rotations**

Each qubit contains a trainable rotation:

```python
qml.RY(weights[i], wires=i)
```

**3. Entanglement**

Neighbouring qubits are connected using CNOT gates:

```python
qml.CNOT(wires=[i, i + 1])
```

**4. Measurement**

The expectation value of Pauli-Z is measured from each qubit.

These measurements form the output of the quantum model.

---

##  Classical Baseline

To evaluate the usefulness of the quantum approach, a classical **Support Vector Machine (SVM)** classifier with an RBF kernel is implemented.

```python
model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)
```

The classical model uses the CNN features after standardization and predicts the disease class for each test image.

The following metrics are calculated:

* Accuracy
* Precision
* Recall
* F1-Score

This provides a baseline against which the quantum model can be studied.

---

##  Model Comparison

The project is designed to compare:

| Model           | Feature Representation | Classifier                  |
| --------------- | ---------------------- | --------------------------- |
| Classical Model | CNN Features           | SVM                         |
| Quantum Model   | 4 PCA Components       | Variational Quantum Circuit |

The comparison helps investigate the feasibility of using a compact quantum circuit for image classification after classical feature extraction and dimensionality reduction.

---

##  Project Structure

```text
Project-Exhibition/
│
├── data/
│   └── Original dataset
│
├── splits/
│   ├── train/
│   │   ├── bacterial_spot/
│   │   ├── early_blight/
│   │   ├── healthy/
│   │   ├── late_blight/
│   │   ├── leaf_mold/
│   │   ├── septoria_leaf_spot/
│   │   ├── spider_mites/
│   │   ├── target_spot/
│   │   └── yellow_leaf_curl_virus/
│   │
│   └── test/
│
├── cnn_features.py
├── cnn_test.py
├── check_dataset.py
│
├── train_classical.py
├── classical_model.py
│
├── train_quantum.py
├── vqc_test.py
│
├── X_train_pca.npy
├── X_test_pca.npy
├── y_train.npy
└── y_test.npy
```

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/Jhanvisrivastava2107/Project-Exhibition.git
cd Project-Exhibition
```

Install the required Python libraries:

```bash
pip install numpy tensorflow scikit-learn pennylane
```

---

##  Running the Project

### 1. Extract CNN Features

Run the CNN feature extraction pipeline:

```bash
python cnn_features.py
```

The extracted features are stored for subsequent processing.

### 2. Prepare PCA Data

Run:

```bash
python train_classical.py
```

This performs:

* Train-test splitting
* Feature scaling
* PCA dimensionality reduction
* Saving of PCA datasets

### 3. Train the Classical Model

Run:

```bash
python classical_model.py
```

This trains the SVM classifier and reports:

* Accuracy
* Precision
* Recall
* F1-score

### 4. Train the Quantum Model

Run:

```bash
python train_quantum.py
```

This:

* Loads the PCA features.
* Builds the 4-qubit quantum circuit.
* Optimizes the trainable quantum parameters.
* Generates predictions on the test set.
* Saves the quantum predictions and trained weights.

### 5. Test the Quantum Circuit

The circuit structure can also be inspected using:

```bash
python vqc_test.py
```

---

##  Key Features

*  Tomato plant disease classification
*  Deep feature extraction using MobileNetV2
*  PCA-based dimensionality reduction
*  4-qubit Variational Quantum Circuit
*  Quantum feature entanglement using CNOT gates
*  Classical SVM baseline
*  Performance evaluation using multiple metrics
*  Classical vs Quantum machine learning comparison

---

## Future Scope

The project can be further improved by:

* Training and evaluating deeper or fine-tuned CNN architectures.
* Experimenting with different quantum feature-encoding techniques.
* Increasing the number of quantum layers and trainable parameters.
* Testing different quantum optimizers.
* Running the circuit on real quantum hardware.
* Performing extensive hyperparameter tuning.
* Comparing VQC with additional classical classifiers.
* Developing a user-friendly web or mobile interface for real-time disease detection.
* Providing treatment or prevention recommendations based on the detected disease.

---

## Team Members

| Name                 | Registration Number |
| -------------------- | ------------------- |
| **Tejal Singh**      | 25BCE10060          |
| **Chahak Gupta**     | 25BCE10138          |
| **Jahnavi Gautam**   | 25BCE10361          |
| **Jhanvi Srivastava** | 25BCE10405          |
| **Monisha Khare**    | 25BCE11318          |

---

## Project Information

**Project Type:** Project Exhibition / Academic Project
**Domain:** Artificial Intelligence & Quantum Machine Learning
**Application:** Plant Disease Detection
**Approach:** Hybrid Classical–Quantum Machine Learning

---

## Conclusion

This project demonstrates a hybrid approach to plant disease detection by combining **deep learning-based feature extraction** with **dimensionality reduction and quantum machine learning**.

By using MobileNetV2 for extracting meaningful image features and PCA for reducing them to four dimensions, the resulting representations can be efficiently processed by a **4-qubit Variational Quantum Circuit**.

A classical SVM model is also implemented to provide a baseline for evaluating the effectiveness of the quantum approach. The project therefore serves as an exploration of how **quantum machine learning can be integrated with conventional computer vision pipelines for practical classification problems**.

---

## Acknowledgement

This project was developed as part of an academic **Project Exhibition**, with the objective of exploring the application of emerging technologies such as **Quantum Machine Learning** in real-world agricultural problems.
