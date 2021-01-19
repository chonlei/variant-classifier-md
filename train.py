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
        train_test_split(train_densities, train_labels, test_size=0.2,
                         random_state=args.seed, shuffle=True)

# Transform features
scaletransform = transform.SimpleStandardScalingTransform()
# TODO or scaling per element?
scaletransform.fit(np.asarray(train_densities).reshape(-1, 1))
for i in range(len(train_densities)):
    train_densities[i] = scaletransform.transform(train_densities[i])
if args.test:
    for i in range(len(test_densities)):
        test_densities[i] = scaletransform.transform(test_densities[i])
    test_data = (test_densities, test_labels)
else:
    test_data = None

joblib.dump(scaletransform, '%s/scaletransform-%s.pkl' \
        % (savedir, saveas), compress=3)
# NOTE, to load:
# scaletransform = joblib.load('%s/scaletransform-%s.pkl' % (savedir, saveas))

if 'nn' in args.method:
    # Building a classifier with NN

    # Neural network epochs and batch size
    batch_size = 1 #4
    epochs = 10 #250

    if args.method == 'cnn':
        # Neural network architecture TODO: need to try other architecture
        architecture = [16, 32, 64]
        input_neurons = (32, 32)
        activation = 'relu'
        input_dim = 1

        nn_model = nn.build_cnn_classification_model(
                input_neurons=input_neurons,
                architecture=architecture,
                input_dim=input_dim,
                act_func=activation)
    elif args.method == 'dnn':
        # Neural network architecture TODO: need to try other architecture
        architecture = [1024, 1024, 1024]
        input_neurons = 1024
        activation = 'relu'
        input_dim = 1

        nn_model = nn.build_dense_classification_model(
                input_neurons=input_neurons,
                architecture=architecture,
                input_dim=input_dim,
                act_func=activation)

    nn_model.summary()

    print('Training the neural network for all stimuli...')
    trained_nn_model = nn.train_classification_model(
            nn_model,
            train_densities,
            train_labels,
            val=test_data,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0)

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
