#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import method.io as io
import method.nn as nn
import numpy as np
import argparse
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
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

parser = argparse.ArgumentParser('K-fold validation for AE-multi-label classifier')
parser.add_argument('--seed', type=int, default=0,
                    help='Seeding of the run')
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

# NOTE: These parameters should be/is chosen through model selection with.
#       k-fold validation.
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
epochs = 100  # NOTE: epochs and batch_size are used by both AE and MLC
batch_size = 512
weights = {0:1, 1:1}
lr = 0.001

# Make save directory
savedir = 'out/mlc-kfold'
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
with open('%s/mlc-input-%s.txt' % (savedir, saveas), 'w') as f:
    f.write(logger)


# Load data
x, l, m = io.load_training_rama('data/MLH1',
                                postfix='_30_40ns',
                                extra=True)

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(x, l, m):
    results = []

    # Define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=5,
                                 random_state=args.seed)

    print('  Training                   | Testing')
    print('  Acc    BACC   F1     AUC   | Acc    BACC   F1     AUC')
    print('-----------------------------------------------------------')

    # Enumerate data
    # 0: B (minority); 1: P (majority)
    for i_cv, (train_ix, test_ix) in enumerate(cv.split(x, l[:, 0, 0, 1])):  # TODO: Split by variant or by frame?
        # Shuffle samples
        np.random.shuffle(train_ix)
        np.random.shuffle(test_ix)

        # Split samples
        x_train, x_test = x[train_ix], x[test_ix]
        l_train, l_test = l[train_ix], l[test_ix]

        xtrs = x_train.shape  # [-1, 334, 217*2]
        xtes = x_test.shape  # [-1, 334, 217*2]

        # Reshape data
        x_train = x_train.reshape(xtrs[0] * xtrs[1], xtrs[2])
        x_test = x_test.reshape(xtes[0] * xtes[1], xtes[2])

        # Get y
        y_train = []
        for li in l_train:
            y_train += [li[0, 0]] * xtrs[1]  # times #MD frames per variant
        y_train = np.asarray(y_train)
        y_test = []
        for li in l_test:
            y_test += [li[0, 0]] * xtes[1]  # times #MD frames per variant
        y_test = np.asarray(y_test)

        # Transform data 1
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)


        # Dimension reduction
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
            encoder.load('%s/ae-%s-cv%s' % (savedir, saveas, i_cv))
        else:
            # Train AE
            encoder.fit(x_train, lag=lag_ae, shape=xtrs, epochs=epochs,
                        batch_size=batch_size, verbose=args.verbose)
            # Save trained AE
            import absl.logging
            absl.logging.set_verbosity(absl.logging.ERROR)
            encoder.save('%s/ae-%s-cv%s' % (savedir, saveas, i_cv))
        x_train = encoder.transform(x_train, whiten=False)
        x_test = encoder.transform(x_test, whiten=False)

        # Transform data 2
        scaler2 = StandardScaler()
        scaler2.fit(x_train)
        x_train = scaler2.transform(x_train)
        x_test = scaler2.transform(x_test)

        # Try SMOTE
        over = SMOTE()
        x_train_2, y_train_2 = over.fit_resample(x_train, y_train)
        # Reformat y from output of SMOTE
        y_train_2 = np.asarray([[0, 1] if y[0] else [1, 0] for y in y_train_2])

        # MLC for B and P
        model = nn.build_dense_mlc_model(
            input_neurons=n_neurons,
            input_dim=n_pcs,
            architecture=[n_neurons] * n_hiddens,
            act_func="leaky_relu",
            l1l2=l1l2,
            dropout=dropout,
            learning_rate=lr
        )
        # Save trained MLC
        if args.cached:
            model = nn.tf.keras.models.load_model('%s/mlc-%s-cv%s' % (savedir, saveas, i_cv))
        else:
            model.fit(
                x_train_2[:, :n_pcs],
                y_train_2,
                class_weight=weights,
                epochs=epochs,
                batch_size=batch_size,
                verbose=args.verbose,
            )
            import absl.logging
            absl.logging.set_verbosity(absl.logging.ERROR)
            model.save('%s/mlc-%s-cv%s' % (savedir, saveas, i_cv), save_format='tf')

        # Predict
        y_train_hat = model.predict(x_train)
        y_train_hat = y_train_hat / np.sum(y_train_hat, axis=1).reshape(-1, 1)
        y_train_hat = y_train_hat[:, 1].round()
        y_test_hat = model.predict(x_test)
        y_test_hat = y_test_hat / np.sum(y_test_hat, axis=1).reshape(-1, 1)
        y_test_hat = y_test_hat[:, 1].round()

        # Calculate scores
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
        acc_train = accuracy_score(y_train[:, 1], y_train_hat)
        acc_test = accuracy_score(y_test[:, 1], y_test_hat)
        bacc_train = balanced_accuracy_score(y_train[:, 1], y_train_hat)
        bacc_test = balanced_accuracy_score(y_test[:, 1], y_test_hat)
        f1_train = f1_score(y_train[:, 1], y_train_hat)
        f1_test = f1_score(y_test[:, 1], y_test_hat)
        auc_train = roc_auc_score(y_train[:, 1], y_train_hat)
        auc_test = roc_auc_score(y_test[:, 1], y_test_hat)

        # Store scores
        r = [acc_train, bacc_train, f1_train, auc_train,
             acc_test, bacc_test, f1_test, auc_test]
        print('> %.3f  %.3f  %.3f  %.3f | %.3f  %.3f  %.3f  %.3f' % (*r,))
        results.append(r)

        # TODO Plot ROC

    return results


results = evaluate_model(x, l, m)
results = np.array(results)

print('\n')
print('Score            : Train         | Test')
print('=================================================')
print('Accuracy         : %.3f (%.3f) | %.3f (%.3f)' % (np.mean(results[:, 0]), np.std(results[:, 0]), np.mean(results[:, 4+0]), np.std(results[:, 4+0])))
print('Balanced Accuracy: %.3f (%.3f) | %.3f (%.3f)' % (np.mean(results[:, 1]), np.std(results[:, 1]), np.mean(results[:, 4+1]), np.std(results[:, 4+1])))
print('Balanced F-score : %.3f (%.3f) | %.3f (%.3f)' % (np.mean(results[:, 2]), np.std(results[:, 2]), np.mean(results[:, 4+2]), np.std(results[:, 4+2])))
print('ROC AUC          : %.3f (%.3f) | %.3f (%.3f)' % (np.mean(results[:, 3]), np.std(results[:, 3]), np.mean(results[:, 4+3]), np.std(results[:, 4+3])))
