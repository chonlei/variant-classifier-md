#!/usr/bin/env python3
import method.io as io
import numpy as np
import argparse
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
parser.add_argument('-m', '--method', type=str, choices=['pca', 'autoencoder'],
                    default='pca', help='Method for dimension reduction')
args = parser.parse_args()

n_pcs = 10

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

# Dimension reduction
if args.method == 'pca':
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA()
    pca = pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
elif args.method == 'autoencoder':
    # Autoencoder
    import method.autoencoder as autoencoder
    autoencoder.tf.random.set_seed(args.seed)
    encoder = autoencoder.Encoder(n_components=n_pcs)
    encoder.fit(x_train)
    x_train = encoder.transform(x_train)
    x_test = encoder.transform(x_test)

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

# Predict
if args.method == 'pca':
    x_test = x_test.reshape(xtes)
elif args.method == 'autoencoder':
    x_test = x_test.reshape(xtes[:-1] + (n_pcs,))

print('Truth   Guess   P   p(B)   p(P)')
for x, l in zip(x_test, l_test[:, 0, 0, 1]):
    prob_b = np.mean(np.exp(kde_b.score_samples(x[:, :n_pcs])))
    prob_p = np.mean(np.exp(kde_p.score_samples(x[:, :n_pcs])))
    #prob = np.max(autoencoder.tf.nn.softmax([prob_b, prob_p]).numpy())
    prob = np.max(np.array([prob_b, prob_p]) / (prob_b + prob_p))
    # Pathogenic or Benign
    truth = 'P' if l else 'B'
    # Unknown or Deleterious
    guess = 'U' if prob_b > prob_p else 'D'
    print(truth + ' '*7 + guess + ' '*6, prob, '  ', prob_b, '  ', prob_p)

if args.plot and (n_pcs == 1):
    import matplotlib.pyplot as plt
    x = np.linspace(np.min(x_train[:, 0]), np.max(x_train[:, 0]), 1000)
    plt.plot(x, np.exp(kde_p.score_samples(x.reshape(-1, 1))))
    plt.plot(x, np.exp(kde_b.score_samples(x.reshape(-1, 1))))
    plt.xlabel('PC1')
    plt.ylabel('Probability Density')
    plt.show()

# Compute centroid
if args.data == 'tp53':
    if args.method == 'pca':
        x_train = x_train.reshape(xtrs)
        x_test = x_test.reshape(xtes)
    elif args.method == 'autoencoder':
        x_train = x_train.reshape(xtrs[:-1] + (n_pcs,))
        x_test = x_test.reshape(xtes[:-1] + (n_pcs,))

    if args.plot:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        b = np.array(l_train[:, 0, 0, 1], dtype=bool)
        cb = [cm.Blues(x) for x in np.linspace(0.4, 1, len(x_train[b]))]
        cp = [cm.Reds(x) for x in np.linspace(0.4, 1, len(x_train[~b]))]
        d = np.array(l_test[:, 0, 0, 1], dtype=bool)
        cd = [cm.Purples(x) for x in np.linspace(0.7, 1, len(x_test[d]))]
        cu = [cm.Oranges(x) for x in np.linspace(0.7, 1, len(x_test[~d]))]
        _, axes = plt.subplots(n_pcs, n_pcs, figsize=(20, 20))
        for i in range(n_pcs):
            for j in range(n_pcs):
                if i == j:
                    for xi, cbi in zip(x_train[b], cb):
                        axes[i, j].hist(xi[::6, j], color=cbi, alpha=0.5)
                    for xi, cbi in zip(x_test[d], cd):
                        axes[i, j].hist(xi[::6, j], color=cbi, alpha=0.5)
                    for xi, cpi in zip(x_train[~b], cp):
                        axes[i, j].hist(xi[::6, j], color=cpi, alpha=0.5)
                    for xi, cpi in zip(x_test[~d], cu):
                        axes[i, j].hist(xi[::6, j], color=cpi, alpha=0.5)
                elif i > j:
                    for xi, cbi in zip(x_train[b], cb):
                        axes[i, j].scatter(xi[::15, j], xi[::15, i], color=cbi,
                                           alpha=0.5)
                    for xi, cbi in zip(x_test[d], cd):
                        axes[i, j].scatter(xi[::15, j], xi[::15, i], color=cbi,
                                           alpha=0.5)
                    for xi, cpi in zip(x_train[~b], cp):
                        axes[i, j].scatter(xi[::15, j], xi[::15, i], color=cpi,
                                           alpha=0.5)
                    for xi, cpi in zip(x_test[~d], cu):
                        axes[i, j].scatter(xi[::15, j], xi[::15, i], color=cpi,
                                           alpha=0.5)
                elif i < j:
                    # Top-right: no plot
                    axes[i, j].axis('off')

                # Set tick labels
                if i < n_pcs - 1:
                    # Only show x tick labels for the last row
                    axes[i, j].set_xticklabels([])
                if j > 0:
                    # Only show y tick labels for the first column
                    axes[i, j].set_yticklabels([])
            if i > 0:
                axes[i, 0].set_ylabel('dim %s' % (i + 1))
            else:
                axes[i, 0].set_ylabel('Counts')
            axes[-1, i].set_xlabel('dim %s' % (i + 1))
        plt.suptitle('Train: Red (Benign), Blue (Pathogenic) |'
                     + ' Test: Orange (B), Purple (P)', fontsize=18)
        plt.tight_layout()
        if args.method == 'pca':
            plt.savefig('out/pca-reduction', dpi=200)
        elif args.method == 'autoencoder':
            plt.savefig('out/ae-reduction', dpi=200)
        plt.close()
        #plt.show()
    x_train_c = np.mean(x_train, axis=1)
    x_test_c = np.mean(x_test, axis=1)
    if args.plot:
        for xi, cbi in zip(x_train_c[b], cb):
            plt.scatter(xi[0], xi[1], color=cbi)
        for xi, cpi in zip(x_train_c[~b], cp):
            plt.scatter(xi[0], xi[1], color=cpi)
        for xi, cbi in zip(x_test_c[d], cd):
            plt.scatter(xi[0], xi[1], color=cbi)
        for xi, cpi in zip(x_test_c[~d], cu):
            plt.scatter(xi[0], xi[1], color=cpi)
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
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

# Predict
print('Truth   Guess   P   p(B)   p(P)')
for x, l in zip(x_test_c, l_test[:, 0, 0, 1]):
    prob_b = np.mean(np.exp(kde_b.score_samples(x[:n_pcs].reshape(1, -1))))
    prob_p = np.mean(np.exp(kde_p.score_samples(x[:n_pcs].reshape(1, -1))))
    #prob = np.max(autoencoder.tf.nn.softmax([prob_b, prob_p]).numpy())
    prob = np.max(np.array([prob_b, prob_p]) / (prob_b + prob_p))
    # Pathogenic or Benign
    truth = 'P' if l else 'B'
    # Unknown or Deleterious
    guess = 'U' if prob_b > prob_p else 'D'
    print(truth + ' '*7 + guess + ' '*6, prob, '  ', prob_b, '  ', prob_p)

if args.plot and (n_pcs == 1):
    import matplotlib.pyplot as plt
    x = np.linspace(np.min(x_train[:, 0]), np.max(x_train[:, 0]), 1000)
    plt.plot(x, np.exp(kde_p.score_samples(x.reshape(-1, 1))))
    plt.plot(x, np.exp(kde_b.score_samples(x.reshape(-1, 1))))
    plt.xlabel('PC1')
    plt.ylabel('Probability Density')
    plt.show()
