#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import method.io as io
import method.nn as nn
import numpy as np
import argparse
import kerastuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
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

Tuning:
https://www.tensorflow.org/tutorials/keras/keras_tuner
https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html
"""

np.warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser('AE-multi-label classifier model selection')
parser.add_argument('--seed', type=int, default=0,
                    help='Seeding of the run')
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
weights = {0:10, 1:1}
lr = 0.001

print('\nTraining:')
print('epochs =', epochs)
print('batch_size =', batch_size)
print('weights = {0:%s, 1:%s}' % (weights[0], weights[1]))
print('lr =', lr)
print('\n')


# Make save directory
savedir = 'out/mlc-tune-2'
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
n_pcs_list = [2, 3, 4, 5, 6, 10, 20, 50, 100]

for i_grid, n_pcs in enumerate(n_pcs_list):
    #n_pcs = 10
    #
    #print('Parameters:')
    #print('n_pcs =', n_pcs)

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

    # Over sampling
    over = SMOTE()
    x_train_2, y_train_2 = over.fit_resample(x_train_2, y_train)
    y_train_2 = np.asarray([[0, 1] if y[0] else [1, 0] for y in y_train_2])

    if args.plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import cm
        b2 = np.array(y_train_2[:, 1], dtype=bool)
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

    # Splitting training and validation data
    x_tra, x_val, y_tra, y_val = train_test_split(
        x_train_2[:, :n_pcs], y_train_2, test_size=0.3, random_state=args.seed, shuffle=True
    )
    # This gives about 0.5 pathogenic and 0.5 benign for training and validation

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

    tuner = kt.BayesianOptimization(
        build_model,
        objective=kt.Objective("val_fbeta_score", direction="max"),
        max_trials=100,
        directory=savedir,
        project_name='tuner-' + saveas,
    )

    stop_early = EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(x_tra,
                 y_tra,
                 class_weight=weights,
                 epochs=epochs,
                 batch_size=batch_size,
                 #validation_split=0.3,
                 validation_data=(x_val, y_val),
                 callbacks=[stop_early],
                 verbose=args.verbose)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the best model with full epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(x_tra,
                        y_tra,
                        class_weight=weights,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        verbose=args.verbose)

    print('Hyperparameter search completed for n_pcs =', n_pcs)
    for h in ['n_neurons', 'n_hiddens', 'dropout']:
        print(h, '=', best_hps.get(h))
    print('Metrics:')
    for m in ['accuracy', 'fbeta_score', 'val_accuracy', 'val_fbeta_score']:
        print(m, '=', history.history[m][-1])
    print('\n')
