#!/usr/bin/env python3
import method.io as io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Load data
x, l = io.load_training_rama('data/TP53')

# Split data
x_train, x_test, l_train, l_test = train_test_split(
    x, l, test_size=0.2, random_state=1, shuffle=True
)

# Reshape data
x_train = x_train.reshape(-1, 217 * 2)
x_test = x_test.reshape(-1, 217 * 2)

# Transform data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Set seed
np.random.seed(1)

# PCA
pca = PCA()
pca = pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

# Redo labels
y_train = []
for l in l_train:
    y_train += [l[0, 0, 1]] * 334  # 334 frame per variant
y_test = []
for l in l_test:
    y_test += [l[0, 0, 1]] * 334

# Set seed
np.random.seed(1)

# KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train[:, :100], y_train)
y_pred = knn.predict(x_test[:, :100])
print(confusion_matrix(y_test, y_pred))

# Compute centroid
x_train = x_train.reshape(-1, 334, 217 * 2)
x_test = x_test.reshape(-1, 334, 217 * 2)
x_train = np.mean(x_train, axis=1)
x_test = np.mean(x_test, axis=1)

# KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train[:, :10], l_train[:, 0, 0, 1])
y_pred = knn.predict(x_test[:, :10])
print(confusion_matrix(l_test[:, 0, 0, 1], y_pred))
