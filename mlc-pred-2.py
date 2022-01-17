#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import method.io as io
import method.nn as nn
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

"""
Treat this as a multi-label classification problem, using a cost-sensitive
neural network for imbalanced classification.

Intro to MLC:
https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
Cost-sensitive
https://machinelearningmastery.com/cost-sensitive-neural-network-for-imbalanced-classification/

Imbalanced:
https://machinelearningmastery.com/what-is-imbalanced-classification/
https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/
https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5
https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
"""

np.warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser('Prediction using AE-multi-label classifier')
parser.add_argument('--seed', type=int, default=0,
                    help='Seeding of the run')
parser.add_argument('-m', '--method', type=str,
                    choices=['pca', 'ae', 'aerf'],
                    default='ae', help='Method for dimension reduction')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Printing tensorflow output to stdout')
parser.add_argument('-p', '--plot', action='store_true',
                    help='Making and showing some plots')
parser.add_argument('-x', '--cached', action='store_true',
                    help='Use cached AE and MLC')
args = parser.parse_args()

# Set seed
np.random.seed(args.seed)
nn.tf.random.set_seed(args.seed)

# Parameters (best hyperparameters from mlc-tune-2.py)
n_pcs = 6
n_neurons_ae = 1000
n_hiddens_ae = 2
l1l2_ae = None
dropout_ae = 0.1
lag_ae = 1
n_neurons = 128
n_hiddens = 3
l1l2 = None
dropout = 0.2

# Training params
epochs = 100  # NOTE: epochs, batch_size and lr are used by both AE and MLC
batch_size = 512
lr = 0.001
weights = {0:1, 1:1}


# Make save directory
savedir = 'out/mlc-pred'
if not os.path.isdir(savedir):
    os.makedirs(savedir)
saveas = str(args.seed)


# Log inputs
logger = ''
logger += 'AE hyperparameters:\n'
logger += '\nn_pcs = ' + str(n_pcs)
logger += '\nn_neurons_ae = ' + str(n_neurons_ae)
logger += '\nn_hiddens_ae = ' + str(n_hiddens_ae)
logger += '\nl1l2_ae = ' + str(l1l2_ae)
logger += '\ndropout_ae = ' + str(dropout_ae)
logger += '\nlag_ae = ' + str(lag_ae)
logger += '\n'
logger += '\nMLC hyperparameters:\n'
logger += '\nn_neurons = ' + str(n_neurons)
logger += '\nn_hiddens = ' + str(n_hiddens)
logger += '\nl1l2 = ' + str(l1l2)
logger += '\ndropout = ' + str(dropout)
logger += '\n'
logger += '\nTraining:\n'
logger += '\nepochs = ' + str(epochs)
logger += '\nbatch_size = ' + str(batch_size)
logger += '\nlr = ' + str(lr)
logger += '\nweights = {0:%s, 1:%s}' % (weights[0], weights[1])
with open('%s/%s-mlc-input-%s.txt' % (savedir, args.method, saveas), 'w') as f:
    f.write(logger)


# Load data
x_train, l_train, m_train = io.load_training_rama('data/MLH1',
                                                  postfix='_30_40ns',
                                                  extra=True)

xtrs = x_train.shape  # [-1, 334, 217*2]

# Reshape data
x_train = x_train.reshape(xtrs[0] * xtrs[1], xtrs[2])


# Transform data 1
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)


# Dimension reduction
if args.method == 'pca':
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA()
    pca = pca.fit(x_train)
    x_train = pca.transform(x_train)
elif args.method == 'ae':
    # Autoencoder
    import method.autoencoder as autoencoder
    autoencoder.tf.random.set_seed(args.seed)
    encoder_units = [n_neurons_ae] * n_hiddens_ae
    encoder = autoencoder.Encoder(n_components=n_pcs,
                                  units=encoder_units,
                                  l1l2=l1l2_ae,
                                  dropout=dropout_ae)
    if args.cached:
        # Load trained AE
        encoder.load('%s/ae-%s' % (savedir, saveas))
    else:
        # Train AE
        encoder.fit(x_train, lag=lag_ae, shape=xtrs, epochs=epochs,
                    batch_size=batch_size, verbose=args.verbose)
        # Save trained AE
        encoder.save('%s/ae-%s' % (savedir, saveas))
    x_train = encoder.transform(x_train, whiten=False)
elif args.method == 'aerf':
    # Autoencoder for e.g. 100 features; RF to pick e.g. 10 features
    import method.autoencoder as autoencoder
    n_compression = 100  # something smaller than the full MD features
    autoencoder.tf.random.set_seed(args.seed)
    encoder = autoencoder.Encoder(n_components=n_compression)
    encoder.fit(x_train, epochs=epochs, verbose=False)
    x_train = encoder.transform(x_train)
    # Save trained NN
    encoder.save('%s/aerf-%s' % (savedir, saveas))
    # NOTE, to load:
    # >>> encoder = autoencoder.Encoder(n_components=n_compression)
    # >>> encoder.load('%s/ae-%s' % (savedir, saveas))
    # Randoming AE compressed features with RF
    ms_train = []
    for m in range(len(m_train)):
        ms_train += [m] * xtrs[1]  # times number of MD frames
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    rf = RandomForestClassifier(n_estimators=50)
    rf_x_train, rf_x_test, rf_y_train, rf_y_test = train_test_split(
        x_train, ms_train, test_size=0.25, random_state=args.seed, shuffle=True
    )
    rf.fit(rf_x_train, rf_y_train)
    #sorted_idx = rf.feature_importances_.argsort()
    perm_importance = permutation_importance(rf, rf_x_test, rf_y_test)
    sorted_idx = perm_importance.importances_mean.argsort()
    rf_y_pred = rf.predict(rf_x_test)
    x_train = x_train[:, sorted_idx[:n_pcs]]


# Transform data 2
scaler2 = StandardScaler()
scaler2.fit(x_train)
x_train = scaler2.transform(x_train)

if args.plot:
    # Compare wildtype and benigns
    import seaborn as sns
    import matplotlib.pyplot as plt
    import sys
    b = np.array(l_train[:, 0, 0, 1], dtype=bool)
    w = np.array([('wildtype' in m) for m in m_train], dtype=bool)
    plotx = x_train.reshape(xtrs[:-1] + (n_pcs,))
    b = ~b & ~w

    x_train_b = plotx[b].reshape(-1, n_pcs)
    x_train_w = plotx[w].reshape(-1, n_pcs)
    x_train_p = plotx[~(b | w)].reshape(-1, n_pcs)
    np.savetxt(savedir + '/' + args.method + '-raw-wt-' + saveas + '.csv', x_train_w, delimiter=',')
    np.savetxt(savedir + '/' + args.method + '-raw-benign-' + saveas + '.csv', x_train_b, delimiter=',')
    np.savetxt(savedir + '/' + args.method + '-raw-pathogenic-' + saveas + '.csv', x_train_p, delimiter=',')

    _, axes = plt.subplots(n_pcs, n_pcs, figsize=(2 * n_pcs, 2 * n_pcs))
    for i in range(n_pcs):
        for j in range(n_pcs):
            if i == j:
                axes[i, j].hist(x_train_w[:, j], color='C2', alpha=0.8, histtype='step')
                axes[i, j].hist(x_train_b[:, j], color='C0', alpha=0.8, histtype='step')
            elif i > j:
                axes[i, j].scatter(x_train_w[::20, j], x_train_w[::20, i], color='C2', alpha=0.5)
                axes[i, j].scatter(x_train_b[::20, j], x_train_b[::20, i], color='C0', alpha=0.5)
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
    plt.suptitle('Training: Green (Wildtype), Blue (Benign)', fontsize=18)
    plt.tight_layout()
    plt.savefig(savedir + '/' + args.method + '-w-b-' + saveas + '.png', dpi=200)
    plt.savefig(savedir + '/' + args.method + '-w-b-' + saveas + '.jpg', dpi=300)
    plt.close()

    _, axes = plt.subplots(n_pcs, n_pcs, figsize=(2 * n_pcs, 2 * n_pcs))
    for i in range(n_pcs):
        for j in range(n_pcs):
            if i == j:
                axes[i, j].hist(x_train_p[:, j], color='C3', alpha=0.8, histtype='step')
                axes[i, j].hist(x_train_w[:, j], color='C2', alpha=0.8, histtype='step')
                axes[i, j].hist(x_train_b[:, j], color='C0', alpha=0.8, histtype='step')
            elif i > j:
                axes[i, j].scatter(x_train_p[::20, j], x_train_p[::20, i], color='C3', alpha=0.5)
                axes[i, j].scatter(x_train_w[::20, j], x_train_w[::20, i], color='C2', alpha=0.5)
                axes[i, j].scatter(x_train_b[::20, j], x_train_b[::20, i], color='C0', alpha=0.5)
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
    plt.suptitle('Training: Green (Wildtype), Blue (Benign), Red (Pathogenic)', fontsize=18)
    plt.tight_layout()
    plt.savefig(savedir + '/' + args.method + '-w-b-p-' + saveas + '.png', dpi=200)
    plt.savefig(savedir + '/' + args.method + '-w-b-p-' + saveas + '.jpg', dpi=300)
    plt.close()

    del(plotx, x_train_b, x_train_w, x_train_p, b, w)

    # sys.exit()

# Make y as label * #MD frames
y_train = []
for l in l_train:
    y_train += [l[0, 0]] * xtrs[1]  # times #MD frames per variant
y_train = np.asarray(y_train)


# Try SMOTE
over = SMOTE()
x_train_2, y_train_2 = over.fit_resample(x_train, y_train)
y_train_2 = np.asarray([[0, 1] if y[0] else [1, 0] for y in y_train_2])


# MLC for B and P
model = nn.build_dense_mlc_model(input_neurons=n_neurons,
                                 input_dim=n_pcs,
                                 architecture=[n_neurons] * n_hiddens,
                                 act_func="leaky_relu",
                                 l1l2=l1l2,
                                 dropout=dropout,
                                 learning_rate=lr)

# Save trained MLC
if args.cached:
    model = nn.tf.keras.models.load_model('%s/mlc-%s' % (savedir, saveas))
else:
    model.fit(
        x_train_2[:, :n_pcs],
        y_train_2,
        class_weight=weights,
        epochs=epochs,
        batch_size=batch_size,
        verbose=args.verbose,
    )
    model.save('%s/mlc-%s' % (savedir, saveas), save_format='tf')


# Fitted results
if args.method == 'pca':
    x_train = x_train.reshape(xtrs)
elif args.method in ['ae', 'aerf']:
    x_train = x_train.reshape(xtrs[:-1] + (n_pcs,))

#print('\nTruth   Guess   P   p(B)   p(P)')
pred_train = []
pred_prob_train = []
for x, l in zip(x_train, l_train[:, 0, 0, 1]):
    pred = model.predict(x[:, :n_pcs])

    prob_b = np.mean(pred[:, 0])
    prob_p = np.mean(pred[:, 1])
    prob_b_sd = np.std(pred[:, 0])
    prob_p_sd = np.std(pred[:, 1])
    #prob_b = np.percentile(pred[:, 0], 75)
    #prob_p = np.percentile(pred[:, 1], 50)
    prob = np.max(autoencoder.tf.nn.softmax([prob_b, prob_p]).numpy())
    #prob = np.max(np.array([prob_b, prob_p]) / (prob_b + prob_p))
    # Pathogenic or Benign
    truth = 'P' if l else 'B'
    # Unknown or Deleterious
    guess = 'U' if prob_b > prob_p else 'D'
    #print(truth + ' '*7 + guess + ' '*6, prob, '  ', prob_b, '  ', prob_p)

    pred_train.append(guess)
    pred_prob_train.append([prob, prob_b, prob_p, prob_b_sd, prob_p_sd])
pred_prob_train = np.array(pred_prob_train)


#
# Prediction
#
x_vus, m_vus = io.load_vus_rama('data/MLH1', postfix='_30_40ns')

xvus = x_vus.shape  # [-1, 334, 217*2]

x_vus = x_vus.reshape(xvus[0] * xvus[1], xvus[2])

x_vus = scaler.transform(x_vus)
x_vus_tmp = np.zeros(x_vus.shape)[:, :n_pcs]
for i in range(xvus[0]):
    if args.method == 'pca':
        x_vus_tmp[i*xvus[1]:(i+1)*xvus[1]] = pca.transform(x_vus[i*xvus[1]:(i+1)*xvus[1]])
    elif args.method == 'ae':
        x_vus_tmp[i*xvus[1]:(i+1)*xvus[1]] = encoder.transform(x_vus[i*xvus[1]:(i+1)*xvus[1]])
    elif args.method == 'aerf':
        x_vus[i*xvus[1]:(i+1)*xvus[1]] = encoder.transform(x_vus[i*xvus[1]:(i+1)*xvus[1]])
        x_vus_tmp[i*xvus[1]:(i+1)*xvus[1]] = x_vus[i*xvus[1]:(i+1)*xvus[1], sorted_idx[:n_pcs]]
x_vus = scaler2.transform(x_vus_tmp)

x_vus = x_vus.reshape(xvus[:-1] + (n_pcs,))
x_vus_c = np.mean(x_vus, axis=1)

if False:
    for i, x in enumerate(x_vus):
        pred = model.predict(x[:, :n_pcs])
        np.savetxt(f'tmp/ae-vus-{i}.csv', x[:, :n_pcs], delimiter=',')
        np.savetxt(f'tmp/classified-vus-{i}.csv', pred, delimiter=',')
    sys.exit()


pred_vus = []
pred_prob_vus = []
for x in x_vus:
    pred = model.predict(x[:, :n_pcs])

    prob_b = np.mean(pred[:, 0])
    prob_p = np.mean(pred[:, 1])
    prob_b_sd = np.std(pred[:, 0])
    prob_p_sd = np.std(pred[:, 1])
    #prob_b = np.percentile(pred[:, 0], 75)
    #prob_p = np.percentile(pred[:, 1], 50)
    prob = np.max(autoencoder.tf.nn.softmax([prob_b, prob_p]).numpy())
    #prob = np.max(np.array([prob_b, prob_p]) / (prob_b + prob_p))
    # Unknown or Deleterious
    guess = 'U' if prob_b > prob_p else 'D'

    pred_vus.append(guess)
    pred_prob_vus.append([prob, prob_b, prob_p, prob_b_sd, prob_p_sd])
pred_prob_vus = np.array(pred_prob_vus)


# Compute centroid
if args.method == 'pca':
    x_train = x_train.reshape(xtrs)
elif args.method in ['ae', 'aerf']:
    x_train = x_train.reshape(xtrs[:-1] + (n_pcs,))
x_train_c = np.mean(x_train, axis=1)


# Plot
if args.plot:
    import seaborn as sns
    import matplotlib.pyplot as plt
    b = np.array(l_train[:, 0, 0, 1], dtype=bool)
    d = np.array([p == 'D' for p in pred_vus], dtype=bool)

    x_train_b = x_train[~b].reshape(-1, n_pcs)
    x_train_p = x_train[b].reshape(-1, n_pcs)

    _, axes = plt.subplots(n_pcs, n_pcs, figsize=(2 * n_pcs, 2 * n_pcs))
    for i in range(n_pcs):
        for j in range(n_pcs):
            if i == j:
                axes[i, j].hist(x_train_c[b][:, j], color='C1', alpha=0.8, histtype='step')
                axes[i, j].hist(x_vus_c[d][:, j], color='C3', alpha=0.8, histtype='step')
                axes[i, j].hist(x_train_c[~b][:, j], color='C0', alpha=0.8, histtype='step')
                axes[i, j].hist(x_vus_c[~d][:, j], color='C2', alpha=0.8, histtype='step')
            elif i > j:
                axes[i, j].scatter(x_train_c[b][:, j], x_train_c[b][:, i], color='C1', alpha=0.5)
                axes[i, j].scatter(x_vus_c[d][:, j], x_vus_c[d][:, i], color='C3', alpha=0.5)
                axes[i, j].scatter(x_train_c[~b][:, j], x_train_c[~b][:, i], color='C0', alpha=0.5)
                axes[i, j].scatter(x_vus_c[~d][:, j], x_vus_c[~d][:, i], color='C2', alpha=0.5)
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
    plt.suptitle('Training: Blue (Benign), Orange (Pathogenic) |'
                 + ' Prediction: Green (Unknown), Red (Deleterious)', fontsize=18)
    plt.tight_layout()
    plt.savefig(savedir + '/' + args.method + '-mlc-prediction-' + saveas, dpi=200)
    plt.close()


#
# Output coordinates of the centroids for each mutant
#
d_train = {
    'mutants': m_train,
    'B/P': ['P' if i else 'B' for i in l_train[:, 0, 0, 1]],
    'U/D': pred_train,
    'certainty': pred_prob_train[:, 0],
    'P(U)': pred_prob_train[:, 1],
    'P(D)': pred_prob_train[:, 2],
    'SD(P(U))': pred_prob_train[:, 3],
    'SD(P(D))': pred_prob_train[:, 4],
}
for i in range(x_train_c.shape[1]):
    d_train['dim' + str(i + 1)] = x_train_c[:, i]
cols = ['mutants', 'B/P'] + ['dim' + str(i + 1)
                             for i in range(x_train_c.shape[1])] \
       + ['U/D', 'certainty', 'P(U)', 'P(D)', 'SD(P(U))', 'SD(P(D))']
df = pd.DataFrame(d_train, columns=cols)
df.to_csv('%s/%s-mlc-train-%s.csv' % (savedir, args.method, saveas),
          index=False, header=True)

d_vus = {
    'mutants': m_vus,
    'B/P': ['N/A' for i in m_vus],
    'U/D': pred_vus,
    'probability': pred_prob_vus,
    'certainty': pred_prob_vus[:, 0],
    'P(U)': pred_prob_vus[:, 1],
    'P(D)': pred_prob_vus[:, 2],
    'SD(P(U))': pred_prob_vus[:, 3],
    'SD(P(D))': pred_prob_vus[:, 4],
}
for i in range(x_vus_c.shape[1]):
    d_vus['dim' + str(i + 1)] = x_vus_c[:, i]
cols = ['mutants', 'B/P'] + ['dim' + str(i + 1)
                             for i in range(x_vus_c.shape[1])] \
       + ['U/D', 'certainty', 'P(U)', 'P(D)', 'SD(P(U))', 'SD(P(D))']
df = pd.DataFrame(d_vus, columns=cols)
df.to_csv('%s/%s-mlc-vus-%s.csv' % (savedir, args.method, saveas),
          index=False, header=True)
