import numpy as np
from sklearn.decomposition import PCA


# -------------------------------
# Load CNN features
# -------------------------------

X = np.load("cnn_features.npy")
y = np.load("labels.npy")


print("Original shape:", X.shape)


# -------------------------------
# PCA
# -------------------------------

pca = PCA(n_components=4)

X_pca = pca.fit_transform(X)


print("Reduced shape:", X_pca.shape)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal variance retained:")
print(pca.explained_variance_ratio_.sum())


# -------------------------------
# Save PCA features
# -------------------------------

np.save("pca_features.npy", X_pca)

print("\nSaved pca_features.npy")