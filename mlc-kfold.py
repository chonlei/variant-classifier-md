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

Intro to MLC
https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
Cost-sensitive
https://machinelearningmastery.com/cost-sensitive-neural-network-for-imbalanced-classification/
Imbalanced
https://machinelearningmastery.com/what-is-imbalanced-classification/
https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/
(XBG: https://stackoverflow.com/questions/40916939/xgboost-for-multilabel-classification)
https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5
https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
"""

np.warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser('Training AE-multi-label classifier')
parser.add_argument('--seed', type=int, default=0,
                    help='Seeding of the run')
parser.add_argument('-m', '--method', type=str,
                    choices=['pca', 'ae', 'aerf'],
                    default='ae', help='Method for dimension reduction')
parser.add_argument('-p', '--plot', action='store_true',
                    help='Making and showing some plots')
args = parser.parse_args()

# Set seed
np.random.seed(args.seed)
nn.tf.random.set_seed(args.seed)

# NOTE: Perhaps when decided to use this approach, do this as a model selection
#       problem with k-fold validation.
n_pcs = 10
n_neurons = 128
n_hiddens = 3
l1l2 = 0.05  # NOTE: l1l2 matters!
dropout = 0.01  # NOTE: dropout rate matters!

print('Parameters:')
print('n_pcs =', n_pcs)
print('n_neurons =', n_neurons)
print('n_hiddens =', n_hiddens)
print('l1l2 =', l1l2)
print('dropout =', dropout)

# Training params
epochs = 100  # NOTE: this is used by both AE and MLC
batch_size = 512
weights = {0:100, 1:1}
lr = 0.001

print('\nTraining:')
print('epochs =', epochs)
print('batch_size =', batch_size)
print('weights = {0:%s, 1:%s}' % (weights[0], weights[1]))
print('lr =', lr)
print('\n')

# Make save directory
savedir = 'out/mlc'
if not os.path.isdir(savedir):
    os.makedirs(savedir)
saveas = str(args.seed) + '-nlat' + str(n_pcs)

# Load data
x, l, m = io.load_training_rama('data/TP53')

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(x, l, m):
    results = []

    # Define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
                                 random_state=args.seed)

    print('  Training                   | Testing')
    print('  Acc    BACC   F1     AUC   | Acc    BACC   F1     AUC')
    print('-----------------------------------------------------------')

    # Enumerate data
    # 0: B (minority); 1: P (majority)
    for train_ix, test_ix in cv.split(x, l[:, 0, 0, 1]):
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
        if args.method == 'pca':
            # PCA
            from sklearn.decomposition import PCA
            pca = PCA()
            pca = pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_test = pca.transform(x_test)
        elif args.method == 'ae':
            # Autoencoder
            import method.autoencoder as autoencoder
            autoencoder.tf.random.set_seed(args.seed)
            encoder = autoencoder.Encoder(n_components=n_pcs)
            encoder.fit(x_train, epochs=epochs, verbose=False)
            x_train = encoder.transform(x_train)
            x_test = encoder.transform(x_test)
            # Save trained NN
            # >>> encoder.save('%s/ae-%s' % (savedir, saveas))
            # NOTE, to load:
            # >>> encoder = autoencoder.Encoder(n_components=n_pcs)
            # >>> encoder.load('%s/ae-%s' % (savedir, saveas))
        elif args.method == 'aerf':
            # Autoencoder for e.g. 100 features; RF to pick e.g. 10 features
            import method.autoencoder as autoencoder
            n_compression = 100  # something smaller than the full MD features
            autoencoder.tf.random.set_seed(args.seed)
            encoder = autoencoder.Encoder(n_components=n_compression)
            encoder.fit(x_train, epochs=epochs, verbose=False)
            x_train = encoder.transform(x_train)
            x_test = encoder.transform(x_test)
            # Save trained NN
            # >>> encoder.save('%s/aerf-%s' % (savedir, saveas))
            # NOTE, to load:
            # >>> encoder = autoencoder.Encoder(n_components=n_compression)
            # >>> encoder.load('%s/ae-%s' % (savedir, saveas))

            # Randoming AE compressed features with RF
            ms_train = []
            for m in range(len(m_train)):
                ms_train += [m] * xtrs[1]  # times number of MD frames

            from sklearn.ensemble import RandomForestClassifier
            from sklearn.inspection import permutation_importance
            from sklearn.metrics import classification_report, confusion_matrix
            from sklearn.metrics import accuracy_score
            rf = RandomForestClassifier(n_estimators=50)
            rf_x_train, rf_x_test, rf_y_train, rf_y_test = train_test_split(
                x_train, ms_train, test_size=0.25, random_state=args.seed,
                shuffle=True
            )
            rf.fit(rf_x_train, rf_y_train)
            #sorted_idx = rf.feature_importances_.argsort()
            perm_importance = permutation_importance(rf, rf_x_test, rf_y_test)
            sorted_idx = perm_importance.importances_mean.argsort()
            rf_y_pred = rf.predict(rf_x_test)
            #print(confusion_matrix(rf_y_test, rf_y_pred))
            #print(classification_report(rf_y_test, rf_y_pred))
            #print('RF acc. score:', accuracy_score(rf_y_test, rf_y_pred))

            x_train = x_train[:, sorted_idx[:n_pcs]]
            x_test = x_test[:, sorted_idx[:n_pcs]]

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
            act_func="relu",
            l1l2=l1l2,  # NOTE: l1l2 matters!
            dropout=dropout,  # NOTE: dropout rate matters!
            learning_rate=lr
        )
        model.fit(
            x_train_2[:, :n_pcs],
            y_train_2,
            class_weight=weights,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False,
        )

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
