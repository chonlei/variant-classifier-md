#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import method.io as io
import method.nn as nn
import numpy as np
import argparse
import kerastuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
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

Tuning:
https://www.tensorflow.org/tutorials/keras/keras_tuner
https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html
https://kegui.medium.com/how-to-do-cross-validation-in-keras-tuner-db4b2dbe079a
"""

np.warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser('AE-multi-label classifier model selection')
parser.add_argument('--seed', type=int, default=0,
                    help='Seeding of the run')
parser.add_argument('--split', type=str, default='variants',
                    choices=['variants', 'frames'],
                    help='Splitting data method')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Printing tensorflow output to stdout')
parser.add_argument('-p', '--plot', action='store_true',
                    help='Making and showing some plots')
args = parser.parse_args()

# Set seed
np.random.seed(args.seed)
nn.tf.random.set_seed(args.seed)

print('Seed:', args.seed)

# Training params
epochs = 100  # NOTE: epochs and batch_size are used by both AE and MLC
batch_size = 512
weights = {0:1, 1:1}
lr = 0.001

print('\nTraining:')
print('epochs =', epochs)
print('batch_size =', batch_size)
print('weights = {0:%s, 1:%s}' % (weights[0], weights[1]))
print('lr =', lr)
print('\n')


# Make save directory
savedir = 'out/mlc-tune'
if not os.path.isdir(savedir):
    os.makedirs(savedir)


# Load data and perform dimensionality reduction
x_train, l_train, m = io.load_training_rama('data/MLH1', postfix='_30_40ns', extra=True)
l_train = np.asarray(list(l_train))
xtrs = x_train.shape
x_train = x_train.reshape(xtrs[0] * xtrs[1], xtrs[2])

# Transform data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)

# Make y as label * #MD frames
y_train = []
for l in l_train:
    y_train += [l[0, 0]] * xtrs[1]  # times #MD frames per variant
y_train = np.asarray(y_train)


# Model selection: grid search for n_pcs
n_pcs_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]

for i_grid, n_pcs in enumerate(n_pcs_list):
    #n_pcs = 10
    #
    #print('Parameters:')
    #print('n_pcs =', n_pcs)

    print('Hyperparameter searching for n_pcs =', n_pcs)

    saveas = str(args.seed) + '-nlat' + str(n_pcs)

    # Autoencoder
    import method.autoencoder as autoencoder
    autoencoder.tf.random.set_seed(args.seed)
    encoder_units = [1000, 1000]  # [xtrs[1] * 100, n_pcs * 100]
    l1l2_ae = None
    dropout_ae = 0.1
    lag_ae = 1
    encoder = autoencoder.Encoder(n_components=n_pcs,
                                  units=encoder_units,
                                  l1l2=l1l2_ae,
                                  dropout=dropout_ae)
    try:
        # Load trained AE
        encoder.load('%s/ae-%s' % (savedir, saveas))
    except:
        # Train AE
        encoder.fit(x_train, lag=lag_ae, shape=xtrs, epochs=epochs,
                    batch_size=batch_size, verbose=args.verbose)
        # Save trained AE
        encoder.save('%s/ae-%s' % (savedir, saveas))
    x_train_2 = encoder.transform(x_train, whiten=False)

    # Transform data
    scaler2 = StandardScaler()
    scaler2.fit(x_train_2)
    x_train_2 = scaler2.transform(x_train_2)

    if args.plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import cm
        b2 = np.array(y_train[:, 1], dtype=bool)
        x_train_b2 = x_train_2[~b2].reshape(-1, n_pcs)
        x_train_p2 = x_train_2[b2].reshape(-1, n_pcs)
        skipp = 10

        _, axes = plt.subplots(n_pcs, n_pcs, figsize=(20, 20))
        for i in range(n_pcs):
            for j in range(n_pcs):
                if i == j:
                    axes[i, j].hist(x_train_p2[::, j], color='C1', alpha=0.4)
                    axes[i, j].hist(x_train_b2[::, j], color='C0', alpha=0.4)
                elif i > j:
                    axes[i, j].scatter(x_train_p2[::skipp, j], x_train_p2[::skipp, i],
                                       color='C1', alpha=0.4)
                    axes[i, j].scatter(x_train_b2[::skipp, j], x_train_b2[::skipp, i],
                                       color='C0', alpha=0.4)
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
        plt.suptitle('Train: Blue (SMOTE Benign), Red (Pathogenic)', fontsize=18)
        plt.tight_layout()
        plt.savefig(savedir + '/ae-reduction-smote-tune-' + saveas, dpi=200)
        plt.close()

    class CVTuner(kt.engine.tuner.Tuner):
        def run_trial(self, trial, x, y, class_weight={0:1, 1:1}, batch_size=512, epochs=100, callbacks=[], verbose=args.verbose):
            # Splitting training and validation data
            cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=2)
            val_bacc = []
            if args.split == 'frames':
                x_tmp = x
                split_iter = cv.split(x_tmp, y[:, 1])
                #print(x_tmp.shape, y[:, 1].shape)
            elif args.split == 'variants':
                x_tmp = x.reshape(xtrs[0], xtrs[1], n_pcs)
                split_iter = cv.split(x_tmp, l_train[:, 0, 0, 1])
                #print(x_tmp.shape, l_train[:, 0, 0, 1].shape)
            # Loop through CV
            for train_indices, test_indices in split_iter:
                # Shuffle samples
                np.random.shuffle(train_indices)
                #print(train_indices.shape, test_indices.shape)
                x_tra, x_val = x_tmp[train_indices], x_tmp[test_indices]
                if args.split == 'frames':
                    y_tra, y_val = y[train_indices], y[test_indices]
                elif args.split == 'variants':
                    l_tra, l_val = l_train[train_indices], l_train[test_indices]
                    x_shape_1 = x_tra.shape
                    x_shape_2 = x_val.shape
                    x_tra = x_tra.reshape(x_shape_1[0] * x_shape_1[1], n_pcs)
                    x_val = x_val.reshape(x_shape_2[0] * x_shape_2[1], n_pcs)
                    # Get y (i.e. label per frame from label per variant `l`)
                    y_tra = []
                    for li in l_tra:
                        y_tra += [li[0, 0]] * x_shape_1[1]  # times #MD frames per variant
                    y_tra = np.asarray(y_tra)
                    y_val = []
                    for li in l_val:
                        y_val += [li[0, 0]] * x_shape_2[1]  # times #MD frames per variant
                    y_val = np.asarray(y_val)
                # Over sampling
                over = SMOTE()
                x_tra_2, y_tra_2 = over.fit_resample(x_tra, y_tra)
                y_tra_2 = np.asarray([[0, 1] if y[0] else [1, 0] for y in y_tra_2])
                # Train
                model = self.hypermodel.build(trial.hyperparameters)
                model.fit(x_tra_2, y_tra_2, batch_size=batch_size, epochs=int(3*epochs/4), class_weight=class_weight, verbose=False)#, callbacks=callbacks, validation_data=(x_val, y_val))
                # Validate
                y_val_hat_ = model.predict(x_val)
                y_val_hat_ = y_val_hat_ / np.sum(y_val_hat_, axis=1).reshape(-1, 1)
                if args.split == 'frames':
                    y_val_hat = y_val_hat_[:, 1].round()
                    val_bacc.append(balanced_accuracy_score(y_val[:, 1], y_val_hat))
                elif args.split == 'variants':
                    l_val_hat_ = np.mean(y_val_hat_[:, 1].reshape(-1, x_shape_2[1]), axis=1)
                    l_val_hat = l_val_hat_.round()
                    #print(l_val_hat, l_val[:, 0, 0, 1])
                    val_bacc.append(balanced_accuracy_score(l_val[:, 0, 0, 1], l_val_hat))
            #print('Testing', trial.hyperparameters.values)
            #print('Validation balanced accuracy score', np.mean(val_bacc))
            # Update
            self.oracle.update_trial(trial.trial_id, {'val_balanced_acc': np.mean(val_bacc)})
            self.save_model(trial.trial_id, model)

    # Define a model
    def build_model(hp):
        n_neurons_hp = hp.Choice('n_neurons', [32, 128, 512, 1024])
        n_hiddens_hp = hp.Choice('n_hiddens', [0, 1, 2, 3])
        dropout_hp = hp.Choice('dropout', [0., 0.2, 0.4])
        model = nn.build_dense_mlc_model(input_neurons=n_neurons_hp,
                                         input_dim=n_pcs,
                                         architecture=[n_neurons_hp] * n_hiddens_hp,
                                         act_func="leaky_relu",
                                         l1l2=None,
                                         dropout=dropout_hp,
                                         learning_rate=lr)
        return model

    tuner = CVTuner(
        hypermodel=build_model,
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective("val_balanced_acc", direction="max"),
            max_trials=100,
        ),
        directory=savedir,
        project_name='tuner-' + saveas,
    )

    stop_early = EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(x_train_2[:, :n_pcs], y_train,
                 class_weight=weights,
                 epochs=epochs,
                 batch_size=batch_size,
                 #validation_split=0.3,
                 #validation_data=(x_val, y_val),
                 callbacks=[stop_early],
                 verbose=args.verbose)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the best model with full epochs with CV
    model = tuner.hypermodel.build(best_hps)
    historys = []
    cv = RepeatedStratifiedKFold(n_splits=4)
    if args.split == 'frames':
        x_tmp = x_train_2[:, :n_pcs]
        split_iter = cv.split(x_tmp, y_train[:, 1])
    elif args.split == 'variants':
        x_tmp = x_train_2[:, :n_pcs].reshape(xtrs[0], xtrs[1], n_pcs)
        split_iter = cv.split(x_tmp, l_train[:, 0, 0, 1])
    for train_indices, test_indices in split_iter:
        x_tra, x_val = x_tmp[train_indices], x_tmp[test_indices]
        if args.split == 'frames':
            y_tra, y_val = y_train[train_indices], y_train[test_indices]
        elif args.split == 'variants':
            l_tra, l_val = l_train[train_indices], l_train[test_indices]
            x_shape_1 = x_tra.shape
            x_shape_2 = x_val.shape
            x_tra = x_tra.reshape(x_shape_1[0] * x_shape_1[1], x_shape_1[2])
            x_val = x_val.reshape(x_shape_2[0] * x_shape_2[1], x_shape_2[2])
            # Get y (i.e. label per frame from label per variant `l`)
            y_tra = []
            for li in l_tra:
                y_tra += [li[0, 0]] * x_shape_1[1]  # times #MD frames per variant
            y_tra = np.asarray(y_tra)
            y_val = []
            for li in l_val:
                y_val += [li[0, 0]] * x_shape_2[1]  # times #MD frames per variant
            y_val = np.asarray(y_val)
        # Over sampling
        over = SMOTE()
        x_tra_2, y_tra_2 = over.fit_resample(x_tra, y_tra)
        y_tra_2 = np.asarray([[0, 1] if y[0] else [1, 0] for y in y_tra_2])
        model.fit(
            x_tra_2,
            y_tra_2,
            class_weight=weights,
            epochs=epochs,
            batch_size=batch_size,
            #validation_data=(x_val, y_val),
            verbose=args.verbose
        )
        # Validate
        y_tra_hat_ = model.predict(x_tra)
        y_tra_hat_ = y_tra_hat_ / np.sum(y_tra_hat_, axis=1).reshape(-1, 1)
        y_val_hat_ = model.predict(x_val)
        y_val_hat_ = y_val_hat_ / np.sum(y_val_hat_, axis=1).reshape(-1, 1)
        if args.split == 'frames':
            y_tra_hat = y_tra_hat_[:, 1].round()
            y_val_hat = y_val_hat_[:, 1].round()
            historys.append({
                'balanced_accuracy_score': balanced_accuracy_score(y_tra[:, 1], y_tra_hat),
                'roc_auc_score': roc_auc_score(y_tra[:, 1], y_tra_hat_[:, 1]),
                'val_balanced_accuracy_score': balanced_accuracy_score(y_val[:, 1], y_val_hat),
                'val_roc_auc_score': roc_auc_score(y_val[:, 1], y_val_hat_[:, 1]),
            })
        elif args.split == 'variants':
            l_tra_hat_ = np.mean(y_tra_hat_[:, 1].reshape(-1, x_shape_2[1]), axis=1)
            l_tra_hat = l_tra_hat_.round()
            l_val_hat_ = np.mean(y_val_hat_[:, 1].reshape(-1, x_shape_2[1]), axis=1)
            l_val_hat = l_val_hat_.round()
            historys.append({
                'balanced_accuracy_score': balanced_accuracy_score(l_tra[:, 0, 0, 1], l_tra_hat),
                'roc_auc_score': roc_auc_score(l_tra[:, 0, 0, 1], l_tra_hat_),
                'val_balanced_accuracy_score': balanced_accuracy_score(l_val[:, 0, 0, 1], l_val_hat),
                'val_roc_auc_score': roc_auc_score(l_val[:, 0, 0, 1], l_val_hat_),
            })

    print('Hyperparameter search completed for n_pcs =', n_pcs)
    for h in ['n_neurons', 'n_hiddens', 'dropout']:
        print(h, '=', best_hps.get(h))
    print('Metrics:')
    for m in ['balanced_accuracy_score', 'roc_auc_score', 'val_balanced_accuracy_score', 'val_roc_auc_score']:
        v = []
        for h in historys:
            v.append(h[m])
        print(m, '=', np.mean(v))
    print('\n')
