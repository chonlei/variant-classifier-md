#!/usr/bin/env python3
import method.io as io
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser('Training PCA-KNN classifier')
parser.add_argument('--seed', type=int, default=0,
                    help='Seeding of the run')
parser.add_argument('-d', '--data', type=str, choices=['tp53', 'abeta'],
                    default='abeta', help='Data for testing the method')
args = parser.parse_args()

if args.data == 'tp53':
    # Load data
    x, l = io.load_training_rama('data/TP53')

    # Split data
    x_train, x_test, l_train, l_test = train_test_split(
        x, l, test_size=0.2, random_state=args.seed, shuffle=True
    )

    # Reshape data
    x_train = x_train.reshape(-1, 217 * 2)
    x_test = x_test.reshape(-1, 217 * 2)
elif args.data == 'abeta':
    import pandas
    abeta = pandas.read_csv('data/abeta2.csv')
    l = np.zeros(abeta.shape[0])
    l[abeta.phenotype1 == 'Pathogenic'] = 1
    x = np.asarray(np.asarray(abeta)[:, 3:], dtype=float)
    test = (abeta.grouping == 'E22D') | (abeta.grouping == 'A21G')
    train = ~test
    x_train = x[train]
    y_train = l[train]
    x_test = x[test]
    y_test = l[test]

# Transform data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Set seed
np.random.seed(args.seed)

# PCA
pca = PCA()
pca = pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

# Redo labels
if args.data == 'tp53':
    y_train = []
    for l in l_train:
        y_train += [l[0, 0, 1]] * 334  # 334 frame per variant
    y_test = []
    for l in l_test:
        y_test += [l[0, 0, 1]] * 334

# Set seed
np.random.seed(args.seed)

# KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train[:, :100], y_train)
y_pred = knn.predict(x_test[:, :100])
print(confusion_matrix(y_test, y_pred))

# Compute centroid
if args.data == 'tp53':
    x_train = x_train.reshape(-1, 334, 217 * 2)
    x_test = x_test.reshape(-1, 334, 217 * 2)
    x_train = np.mean(x_train, axis=1)
    x_test = np.mean(x_test, axis=1)
elif args.data == 'abeta':
    raise NotImplementedError

# KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train[:, :10], l_train[:, 0, 0, 1])
y_pred = knn.predict(x_test[:, :10])
print(confusion_matrix(l_test[:, 0, 0, 1], y_pred))
