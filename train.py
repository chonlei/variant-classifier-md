#!/usr/bin/env python3
import sys
import os
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
import method.io as io
import method.transform as transform

parser = argparse.ArgumentParser('Training classifier')
parser.add_argument('--seed', type=int, default=0,
                    help='Seeding of the run')
parser.add_argument('-t', '--test', action='store_true',
                    help='Split training data into train and test sets')
parser.add_argument('--method', type=str, choices=['dnn', 'cnn'],
                    default='cnn',
                    help='Method for building the classifier')
args = parser.parse_args()

# Set seed
np.random.seed(args.seed)
if 'nn' in args.method:
    import method.nn as nn
    nn.tf.random.set_seed(args.seed)

# Make save directory
savedir = 'out/' + args.method
if not os.path.isdir(savedir):
    os.makedirs(savedir)
saveas = args.seed

# Load training data
train_densities, train_labels = io.load_training_density('data/TP53')

if args.test:
    # Split data into train and test
    from sklearn.model_selection import train_test_split
    train_densities, test_densities, train_labels, test_labels = \
        train_test_split(train_densities, train_labels, test_size=0.1,
                         random_state=args.seed, shuffle=True)

# Transform features
scaletransform = transform.StandardScalingTransform()
# Scaling per element
scaletransform.fit(np.asarray(train_densities).reshape(-1, 32 * 32))
train_densities = scaletransform.transform(
        train_densities.reshape(-1, 32 * 32)).reshape(-1, 32, 32, 1)
if args.test:
    test_densities = scaletransform.transform(
            test_densities.reshape(-1, 32 * 32)).reshape(-1, 32, 32, 1)

joblib.dump(scaletransform, '%s/scaletransform-%s.pkl' \
        % (savedir, saveas), compress=3)
# NOTE, to load:
# scaletransform = joblib.load('%s/scaletransform-%s.pkl' % (savedir, saveas))

if 'nn' in args.method:
    # Building a classifier with NN

    # Neural network epochs and batch size
    batch_size = 1
    epochs = 30

    if args.method == 'cnn':
        # Neural network architecture TODO: need to try other architecture
        architecture = [16, 32, 64]
        input_neurons = (32, 32)
        activation = 'relu'
        input_dim = 1
        l1l2_regulariser = 0.005  # Another sensitive hyperparameter...
        dropout = 0.3

        nn_model = nn.build_cnn_classification_model(
                input_neurons=input_neurons,
                architecture=architecture,
                input_dim=input_dim,
                act_func=activation,
                l1l2=l1l2_regulariser,
                dropout=dropout)
    elif args.method == 'dnn':
        # Neural network architecture (manual tuning seems working well)
        architecture = [128] * 1  # [64, 32, 16]
        input_neurons = 128
        activation = 'relu'
        input_dim = 32 * 32  # Size of the density matrix
        l1l2_regulariser = 0.001  # Another sensitive hyperparameter...

        nn_model = nn.build_dense_classification_model(
                input_neurons=input_neurons,
                architecture=architecture,
                input_dim=input_dim,
                act_func=activation,
                l1l2=l1l2_regulariser)

        # Reshape inputs and outputs for DNN
        train_densities = train_densities.reshape(-1, 32 * 32)
        train_labels = train_labels.reshape(-1, 2)
        test_densities = test_densities.reshape(-1, 32 * 32)
        test_labels = test_labels.reshape(-1, 2)

    nn_model.summary()

    if args.test:
        test_data = (test_densities, test_labels)
    else:
        test_dat = None

    print('Training the neural network for all stimuli...')
    trained_nn_model = nn.train_classification_model(
            nn_model,
            train_densities,
            train_labels,
            val=test_data,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)

    # Inspect loss function
    acc = trained_nn_model.history.history['accuracy']
    loss = trained_nn_model.history.history['loss']

    if args.test:
        val_acc = trained_nn_model.history.history['val_accuracy']
        val_loss = trained_nn_model.history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    #plt.semilogy(trained_nn_model.history.history['loss'])
    plt.plot(epochs_range, acc, label='Training Accuracy')
    if args.test:
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    if args.test:
        plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.savefig('%s/nn-%s-results' % (savedir, saveas))

    # Save trained NN
    trained_nn_model.save('%s/nn-%s.h5' % (savedir, saveas))
    # NOTE, to load:
    # >>> import tensorflow as tf
    # >>> trained_nn_model = tf.keras.models.load_model(
    # ...                    '%s/nn-%s.h5' % (savedir, saveas))
