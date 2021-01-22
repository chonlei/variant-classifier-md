#!/usr/bin/env python3
import method.io as io
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser('Training PCA-KNN classifier')
parser.add_argument('--seed', type=int, default=0,
                    help='Seeding of the run')
parser.add_argument('-p', '--plot', action='store_true',
                    help='Making and showing some plots')
parser.add_argument('-d', '--data', type=str, choices=['tp53', 'abeta'],
                    default='abeta', help='Data for testing the method')
args = parser.parse_args()

n_pcs = 2

if args.data == 'tp53':
    # Load data
    x, l = io.load_training_rama('data/TP53')

    # Split data
    x_train, x_test, l_train, l_test = train_test_split(
        x, l, test_size=0.2, random_state=args.seed, shuffle=True
    )
    xtrs = x_train.shape  # [-1, 334, 217*2]
    xtes = x_test.shape  # [-1, 334, 217*2]

    # Reshape data
    x_train = x_train.reshape(xtrs[0] * xtrs[1], xtrs[2])
    x_test = x_test.reshape(xtes[0] * xtes[1], xtes[2])
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

# KDE for B and P
is_p = np.array(y_train, dtype=bool)
bandwidth = np.arange(0.1, 2, .1)
kde = KernelDensity(kernel='gaussian')
grid = GridSearchCV(kde, {'bandwidth': bandwidth})
grid.fit(x_train[~is_p, :n_pcs][::6])  # First N PCs, and downsamples a bit
kde_b = grid.best_estimator_
print('Done estimating KDE for B')
#kde_b.fit(x_train[~is_p, :n_pcs])
kde = KernelDensity(kernel='gaussian')
grid = GridSearchCV(kde, {'bandwidth': bandwidth})
grid.fit(x_train[is_p, :n_pcs][::10])  # First N PCs, and downsamples more
kde_p = grid.best_estimator_
print('Done estimating KDE for P')
#kde_p.fit(x_train[is_p, :n_pcs])

x_test = x_test.reshape(xtes)
print('Truth   Guess   p(B)   p(P)')
for x, l in zip(x_test, l_test[:, 0, 0, 1]):
    prob_b = np.mean(np.exp(kde_b.score_samples(x[:, :n_pcs])))
    prob_p = np.mean(np.exp(kde_p.score_samples(x[:, :n_pcs])))
    # Pathogenic or Benign
    truth = 'P' if l else 'B'
    # Unknown or Deleterious
    guess = 'U' if prob_b > prob_p and prob_b > 1e-5 else 'D'
    print(truth + ' '*7 + guess + ' '*6, prob_b, '  ', prob_p)

if args.plot and (n_pcs == 1):
    import matplotlib.pyplot as plt
    x = np.linspace(np.min(x_train[:, 0]), np.max(x_train[:, 0]), 100)
    plt.plot(x, np.exp(kde_p.score_samples(x.reshape(-1, 1))))
    plt.plot(x, np.exp(kde_b.score_samples(x.reshape(-1, 1))))
    plt.xlabel('PC1')
    plt.ylabel('Probability Density')
    plt.show()

# Compute centroid
if args.data == 'tp53':
    x_train = x_train.reshape(xtrs)
    x_test = x_test.reshape(xtes)
    if args.plot:
        from matplotlib import cm
        b = np.array(l_train[:, 0, 0, 1], dtype=bool)
        cb = [cm.Blues(x) for x in np.linspace(0.3, 1, len(x_train[b]))]
        for xi, cbi in zip(x_train[b], cb):
            plt.scatter(xi[:, 0], xi[:, 1], color=cbi)
        cp = [cm.Reds(x) for x in np.linspace(0.3, 1, len(x_train[~b]))]
        for xi, cpi in zip(x_train[~b], cp):
            plt.scatter(xi[:, 0], xi[:, 1], color=cpi)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()
    x_train_c = np.mean(x_train, axis=1)
    x_test_c = np.mean(x_test, axis=1)
    if args.plot:
        for xi, cbi in zip(x_train_c[b], cb):
            plt.scatter(xi[0], xi[1], color=cbi)
        for xi, cpi in zip(x_train_c[~b], cp):
            plt.scatter(xi[0], xi[1], color=cpi)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()
elif args.data == 'abeta':
    raise NotImplementedError

# KDE for B and P with centroids
is_p = np.array(l_train[:, 0, 0, 1], dtype=bool)
bandwidth = np.arange(0.1, 2, .1)
kde = KernelDensity(kernel='gaussian')
grid = GridSearchCV(kde, {'bandwidth': bandwidth})
grid.fit(x_train_c[~is_p, :n_pcs])  # First 5 PCs
kde_b = grid.best_estimator_
print('Done estimating KDE for B')
#kde_b.fit(x_train_c[~is_p, :n_pcs])
kde = KernelDensity(kernel='gaussian')
grid = GridSearchCV(kde, {'bandwidth': bandwidth})
grid.fit(x_train_c[is_p, :n_pcs])  # First 5 PCs
kde_p = grid.best_estimator_
print('Done estimating KDE for P')
#kde_p.fit(x_train_c[is_p, :n_pcs])

print('Truth   Guess   p(B)   p(P)')
for x, l in zip(x_test_c, l_test[:, 0, 0, 1]):
    prob_b = np.mean(np.exp(kde_b.score_samples(x[:n_pcs].reshape(1, -1))))
    prob_p = np.mean(np.exp(kde_p.score_samples(x[:n_pcs].reshape(1, -1))))
    # Pathogenic or Benign
    truth = 'P' if l else 'B'
    # Unknown or Deleterious
    guess = 'U' if prob_b > prob_p and prob_b > 1e-5 else 'D'
    print(truth + ' '*7 + guess + ' '*6, prob_b, '  ', prob_p)

if args.plot and (n_pcs == 1):
    import matplotlib.pyplot as plt
    x = np.linspace(np.min(x_train[:, 0]), np.max(x_train[:, 0]), 100)
    plt.plot(x, np.exp(kde_p.score_samples(x.reshape(-1, 1))))
    plt.plot(x, np.exp(kde_b.score_samples(x.reshape(-1, 1))))
    plt.xlabel('PC1')
    plt.ylabel('Probability Density')
    plt.show()
