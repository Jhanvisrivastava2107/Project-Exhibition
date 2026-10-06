import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -----------------------------
# LOAD DATA
# -----------------------------

X = np.load("features.npy")
y = np.load("labels.npy")

print("Original data shape:", X.shape)


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# -----------------------------
# SCALE FEATURES
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# PCA
# -----------------------------

pca = PCA(n_components=4)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

print("\nAfter PCA:")

print("Training shape:", X_train_pca.shape)
print("Testing shape:", X_test_pca.shape)

print("\nExplained variance:")
print(pca.explained_variance_ratio_)


# -----------------------------
# SAVE DATA
# -----------------------------

np.save("X_train_pca.npy", X_train_pca)
np.save("X_test_pca.npy", X_test_pca)

np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("\nPCA data saved successfully!")