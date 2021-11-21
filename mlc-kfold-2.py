#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import method.io as io
import method.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns; sns.set_theme(context='paper')

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
parser.add_argument('--split', type=str, default='variants',
                    choices=['variants', 'frames'],
                    help='Splitting data method')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Printing tensorflow output to stdout')
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
if not os.path.isdir(savedir + '/cached'):
    os.makedirs(savedir + '/cached')
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

    if args.split == 'frames':
        # Reshape data and prepare data for n-lag AE
        x_4_n_lag = x[:, :-lag_ae, :]
        xshape = x_4_n_lag.shape  # [-1, 334 - n, 217*2]
        x_4_n_lag = x_4_n_lag.reshape(xshape[0] * xshape[1], xshape[2])

        # Get y (i.e. label per frame from label per variant `l`)
        y = []
        for li in l:
            y += [li[0, 0]] * xshape[1]  # times #MD frames per variant
        y = np.asarray(y)

        split_iter = cv.split(x_4_n_lag, y[:, 1])
    elif args.split == 'variants':
        split_iter = cv.split(x, l[:, 0, 0, 1])

    # Set up ROC plot
    tprs_train = []
    aucs_train = []
    tprs_test = []
    aucs_test = []
    mean_fpr = np.linspace(0, 1, 100)
    fig_train, ax_train = plt.subplots(figsize=(5, 4))
    fig_test, ax_test = plt.subplots(figsize=(5, 4))

    # Enumerate data
    # 0: B (minority); 1: P (majority)
    for i_cv, (train_ix, test_ix) in enumerate(split_iter):
        # Shuffle samples
        np.random.shuffle(train_ix)
        np.random.shuffle(test_ix)

        if args.split == 'frames':
            # Split samples
            x_train, x_test = x_4_n_lag[train_ix], x_4_n_lag[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]

            # Get lag n frames for AE
            idx1 = [int(idx // xshape[1]) for idx in train_ix]
            idx2 = (train_ix % xshape[1]) + lag_ae
            x_train_lag_n = x[idx1, idx2, :]

        elif args.split == 'variants':
            # Split samples
            x_train, x_test = x[train_ix], x[test_ix]
            l_train, l_test = l[train_ix], l[test_ix]

            xtrs = x_train.shape  # [-1, 334, 217*2]
            xtes = x_test.shape  # [-1, 334, 217*2]

            # Reshape data
            x_train = x_train.reshape(xtrs[0] * xtrs[1], xtrs[2])
            x_test = x_test.reshape(xtes[0] * xtes[1], xtes[2])

            # Get y (i.e. label per frame from label per variant `l`)
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
            encoder.load('%s/cached/ae-%s-cv%s' % (savedir, saveas, i_cv))
        else:
            # Train AE
            if args.split == 'frames':
                encoder.fit(x_train, Y=x_train_lag_n, epochs=epochs,
                            batch_size=batch_size, verbose=args.verbose)
            elif args.split == 'variants':
                encoder.fit(x_train, lag=lag_ae, shape=xtrs, epochs=epochs,
                            batch_size=batch_size, verbose=args.verbose)
            # Save trained AE
            import absl.logging
            absl.logging.set_verbosity(absl.logging.ERROR)
            encoder.save('%s/cached/ae-%s-cv%s' % (savedir, saveas, i_cv))
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
            model = nn.tf.keras.models.load_model(
                '%s/cached/mlc-%s-cv%s' % (savedir, saveas, i_cv)
            )
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
            model.save('%s/cached/mlc-%s-cv%s' % (savedir, saveas, i_cv),
                       save_format='tf')

        # Predict
        y_train_hat_ = model.predict(x_train)
        y_train_hat_ = y_train_hat_ / np.sum(y_train_hat_, axis=1).reshape(-1, 1)
        y_train_hat = y_train_hat_[:, 1].round()
        y_test_hat_ = model.predict(x_test)
        y_test_hat_ = y_test_hat_ / np.sum(y_test_hat_, axis=1).reshape(-1, 1)
        y_test_hat = y_test_hat_[:, 1].round()

        # Calculate scores
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
        acc_train = accuracy_score(y_train[:, 1], y_train_hat)
        acc_test = accuracy_score(y_test[:, 1], y_test_hat)
        bacc_train = balanced_accuracy_score(y_train[:, 1], y_train_hat)
        bacc_test = balanced_accuracy_score(y_test[:, 1], y_test_hat)
        f1_train = f1_score(y_train[:, 1], y_train_hat)
        f1_test = f1_score(y_test[:, 1], y_test_hat)
        auc_train = roc_auc_score(y_train[:, 1], y_train_hat_[:, 1])
        auc_test = roc_auc_score(y_test[:, 1], y_test_hat_[:, 1])

        # Store scores
        r = [acc_train, bacc_train, f1_train, auc_train,
             acc_test, bacc_test, f1_test, auc_test]
        print('> %.3f  %.3f  %.3f  %.3f | %.3f  %.3f  %.3f  %.3f' % (*r,))
        results.append(r)

        # Plot ROC
        viz = RocCurveDisplay.from_predictions(
            y_train[:, 1],
            y_train_hat_[:, 1],
            alpha=0.3,
            color='#7f7f7f',
            lw=1,
            name='_nolegend_',
            ax=ax_train,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs_train.append(interp_tpr)
        aucs_train.append(viz.roc_auc)
        del(viz)

        viz = RocCurveDisplay.from_predictions(
            y_test[:, 1],
            y_test_hat_[:, 1],
            alpha=0.3,
            color='#7f7f7f',
            lw=1,
            name='_nolegend_',
            ax=ax_test,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs_test.append(interp_tpr)
        aucs_test.append(viz.roc_auc)
        del(viz)

    # Tidy plot ROC (train)
    ax_train.plot([0, 1], [0, 1], linestyle="--", lw=2, color="C3", alpha=0.8)
    mean_tpr = np.mean(tprs_train, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs_train)
    ax_train.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    std_tpr = np.std(tprs_train, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax_train.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax_train.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
    ax_train.legend(loc="lower right")
    fig_train.tight_layout()
    fig_train.savefig('%s/mlc-%s-train-roc' % (savedir, saveas), dpi=200)

    train_roc_savedir = '%s/mlc-%s-train-roc' % (savedir, saveas)
    if not os.path.isdir(train_roc_savedir):
        os.makedirs(train_roc_savedir)
    np.savetxt('%s/fpr.txt' % (train_roc_savedir), mean_fpr.reshape(-1, 1))
    np.savetxt('%s/tpr.txt' % (train_roc_savedir), np.asarray(tprs_train).T)
    np.savetxt('%s/auc.txt' % (train_roc_savedir), aucs_train)
    np.savetxt('%s/tpr_mean_std.txt' % (train_roc_savedir),
               np.asarray([mean_tpr, std_tpr]).T)

    # Tidy plot ROC (test)
    ax_test.plot([0, 1], [0, 1], linestyle="--", lw=2, color="C3", alpha=0.8)
    mean_tpr = np.mean(tprs_test, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs_test)
    ax_test.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    std_tpr = np.std(tprs_test, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax_test.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax_test.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
    ax_test.legend(loc="lower right")
    fig_test.tight_layout()
    fig_test.savefig('%s/mlc-%s-test-roc' % (savedir, saveas), dpi=200)

    test_roc_savedir = '%s/mlc-%s-test-roc' % (savedir, saveas)
    if not os.path.isdir(test_roc_savedir):
        os.makedirs(test_roc_savedir)
    np.savetxt('%s/fpr.txt' % (test_roc_savedir), mean_fpr.reshape(-1, 1))
    np.savetxt('%s/tpr.txt' % (test_roc_savedir), np.asarray(tprs_test).T)
    np.savetxt('%s/auc.txt' % (test_roc_savedir), aucs_test)
    np.savetxt('%s/tpr_mean_std.txt' % (test_roc_savedir),
               np.asarray([mean_tpr, std_tpr]).T)

    return results


results = evaluate_model(x, l, m)
results = np.array(results)
np.savetxt('%s/mlc-%s-results.txt' % (savedir, saveas), results,
        header='Accuracy (train)\t'
              +'Balanced Accuracy (train)\t'
              +'Balanced F-score (train)\t'
              +'ROC AUC (train)\t'
              +'Accuracy (test)\t'
              +'Balanced Accuracy (test)\t'
              +'Balanced F-score (test)\t'
              +'ROC AUC (test)')

output_table = ''
output_table += 'Score            : Train         | Test\n'
output_table += '=================================================\n'
output_table += 'Accuracy         : %.3f (%.3f) | %.3f (%.3f)\n' % (np.mean(results[:, 0]), np.std(results[:, 0]), np.mean(results[:, 4+0]), np.std(results[:, 4+0]))
output_table += 'Balanced Accuracy: %.3f (%.3f) | %.3f (%.3f)\n' % (np.mean(results[:, 1]), np.std(results[:, 1]), np.mean(results[:, 4+1]), np.std(results[:, 4+1]))
output_table += 'Balanced F-score : %.3f (%.3f) | %.3f (%.3f)\n' % (np.mean(results[:, 2]), np.std(results[:, 2]), np.mean(results[:, 4+2]), np.std(results[:, 4+2]))
output_table += 'ROC AUC          : %.3f (%.3f) | %.3f (%.3f)\n' % (np.mean(results[:, 3]), np.std(results[:, 3]), np.mean(results[:, 4+3]), np.std(results[:, 4+3]))
print('\n')
print(output_table + '\n')

with open('%s/mlc-%s-table.txt' % (savedir, saveas), 'w') as f:
    f.write(output_table)
