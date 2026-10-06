import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# -----------------------------
# LOAD ORIGINAL CNN FEATURES
# -----------------------------

X = np.load("features.npy")
y = np.load("labels.npy")


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------
# SCALE
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# CLASSICAL CLASSIFIER
# -----------------------------

model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

print("Training classical model...")

model.fit(X_train, y_train)


# -----------------------------
# PREDICTION
# -----------------------------

predictions = model.predict(X_test)
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = np.load("features.npy")
y = np.load("labels.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="rbf", probability=True, random_state=42)

print("Training classical model...")
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    y_test, predictions,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    y_test, predictions,
    average="weighted",
    zero_division=0
)

print("\n==============================")
print("CLASSICAL MODEL RESULTS")
print("==============================")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)

# -----------------------------
# ACCURACY
# -----------------------------

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("CLASSICAL MODEL RESULTS")
print("==============================")

print("Accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")