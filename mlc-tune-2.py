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

Tuning:
https://www.tensorflow.org/tutorials/keras/keras_tuner
https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html
"""

np.warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser('AE-multi-label classifier model selection')
parser.add_argument('--seed', type=int, default=0,
                    help='Seeding of the run')
args = parser.parse_args()

# Set seed
np.random.seed(args.seed)
nn.tf.random.set_seed(args.seed)

# NOTE: Perhaps when decided to use this approach, do this as a model selection
#       problem with k-fold validation.
n_pcs = 10

print('Parameters:')
print('n_pcs =', n_pcs)

# Training params
epochs = 100  # NOTE: this is used by both AE and MLC
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
savedir = 'out/mlc'
if not os.path.isdir(savedir):
    os.makedirs(savedir)
saveas = str(args.seed) + '-nlat' + str(n_pcs)


# Load data and perform dimensionality reduction
x_train, l_train, m = io.load_training_rama('data/MLH1', postfix='_30_40ns')
l_train = np.asarray(list(l_train))
xtrs = x_train.shape
x_train = x_train.reshape(xtrs[0] * xtrs[1], xtrs[2])

# Transform data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)

# Autoencoder
import method.autoencoder as autoencoder
autoencoder.tf.random.set_seed(args.seed)
encoder_units = [1000, 1000]
l1l2_ae = 1e-5
dropout_ae = 0.3
lag_ae = 1
encoder = autoencoder.Encoder(n_components=n_pcs, units=encoder_units, l1l2=l1l2_ae, dropout=dropout_ae)
if False:
    # Load trained AE
    encoder.load('%s/ae-%s' % (savedir, saveas))
else:
    # Train AE
    encoder.fit(x_train, lag=lag_ae, shape=xtrs, epochs=epochs)
    # Save trained AE
    encoder.save('%s/ae-%s' % (savedir, saveas))
x_train = encoder.transform(x_train, whiten=False)
#x_test = encoder.transform(x_test, whiten=False)

# Transform data
scaler2 = StandardScaler()
scaler2.fit(x_train)
x_train = scaler2.transform(x_train)
#x_test = scaler2.transform(x_test)

# Make y as label * #MD frames
y_train = []
for l in l_train:
    y_train += [l[0, 0]] * xtrs[1]  # times #MD frames per variant
y_train = np.asarray(y_train)
#y_test = []
#for l in l_test:
#    y_test += [l[0, 0]] * xtes[1]  # times #MD frames per variant
#y_test = np.asarray(y_test)

# Over sampling
over = SMOTE()
x_train_2, y_train_2 = over.fit_resample(x_train, y_train)
y_train_2 = np.asarray([[0, 1] if y[0] else [1, 0] for y in y_train_2])

# Define a model
def build_model(hp):
    n_neurons_hp = hp.Choice('n_neurons', [32, 128, 512, 1024])
    n_hiddens_hp = hp.Choice('n_hiddens', [0, 1, 2, 3])
    dropout_hp = hp.Choice('dropout', [0., 0.2, 0.4])
    model = nn.build_dense_mlc_model(input_neurons=n_neurons_hp,
                                     input_dim=n_pcs,
                                     architecture=[n_neurons_hp] * n_hiddens_hp,
                                     act_func="leaky_relu",
                                     l1l2=None,  # NOTE: l1l2 matters!
                                     dropout=dropout_hp,  # NOTE: dropout rate matters!
                                     learning_rate=lr)
    return model

'''
class CVTuner(kt.engine.tuner.Tuner):
  def run_trial(self, trial, x, y, batch_size=32, epochs=1):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    val_losses = []
    for train_indices, test_indices in cv.split(x):
      # Shuffle samples
      np.random.shuffle(train_ix)
      np.random.shuffle(test_ix)
      # Split samples
      x_train, x_test = x[train_indices], x[test_indices]
      y_train, y_test = y[train_indices], y[test_indices]
      model = self.hypermodel.build(trial.hyperparameters)
      model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
      val_losses.append(model.evaluate(x_test, y_test))
    self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
    self.save_model(trial.trial_id, model)

tuner = CVTuner(
  hypermodel=build_model,
  oracle=kt.oracles.Hyperband(
    objective='val_auc',
    max_epochs=epochs))
'''

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=epochs,
    #hyperband_iterations=1,
    #factor=3,
)

stop_early = EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x_train_2[:, :n_pcs],
             y_train_2,
             class_weight=weights,
             epochs=epochs,
             batch_size=batch_size,
             validation_split=0.2,
             callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps)
