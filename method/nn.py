#
# This module contains functions of neural network to apply regression on
# the EFI measurements.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import tensorflow as tf


def build_dense_classification_model(input_neurons=32, input_dim=1,
        architecture=[32, 32, 32], act_func="relu"):
    """
    Builds a densely connected neural network model.

    Arguments
        input_neurons: Number of input neurons.
        input_dim: Dimension of the input vector.
        architecture: Architecture of the hidden layers (densely connected).
        act_func: Activation function. Could be 'relu', 'sigmoid', or 'tanh'.
    Returns
        A neural net (Keras) model for regression.
    """
    if act_func == "relu":
        activation = tf.nn.relu
    elif act_func == "sigmoid":
        activation = tf.nn.sigmoid
    elif act_func == "tanh":
        activation = tf.nn.tanh

    layers = [tf.keras.layers.Dense(input_neurons, input_dim=input_dim,
            activation=activation)]
    num_layers = len(architecture)
    for i in range(num_layers):
        layers.append(tf.keras.layers.Dense(architecture[i],
                activation=activation))
    layers.append(tf.keras.layers.Dense(2))

    model = tf.keras.models.Sequential(layers)
    return model


def build_cnn_classification_model(input_neurons=(32, 32), input_dim=1,
        architecture=[16, 32, 64], act_func="relu"):
    """
    Builds a covolutional neural network model.

    Arguments
        input_neurons: Number of input neurons.
        input_dim: Dimension of the input vector.
        architecture: Architecture of the hidden layers (densely connected).
        act_func: Activation function. Could be 'relu', 'sigmoid', or 'tanh'.
    Returns
        A neural net (Keras) model for regression.
    """
    if act_func == "relu":
        activation = tf.nn.relu
    elif act_func == "sigmoid":
        activation = tf.nn.sigmoid
    elif act_func == "tanh":
        activation = tf.nn.tanh

    layers = [tf.keras.layers.Conv2D(
        16,  # depth of the layer
        3,  # kernel size (n x n)
        input_shape=input_neurons + (input_dim,),
        activation=activation,
        )]
    layers.append(tf.keras.layers.MaxPooling2D())
    num_layers = len(architecture)
    for i in range(num_layers):
        layers.append(tf.keras.layers.Conv2D(
            architecture[i], 3,
            padding='same',
            activation=activation
        ))
        layers.append(tf.keras.layers.MaxPooling2D())
    layers.append(tf.keras.layers.Dropout(0.1))  # regularisation
    layers.append(tf.keras.layers.Dense(128, activation=activation))
    layers.append(tf.keras.layers.Dense(2))

    model = tf.keras.models.Sequential(layers)
    return model


def train_classification_model(model, x_train, y_train, callbacks=None,
        learning_rate=0.001, batch_size=1, epochs=20, verbose=0):
    """
    Compiles and trains a given Keras ``model`` with the given data
    (``x_train``, ``y_train``) for regression. Assumes Adam optimizer for this
    implementation. Assumes mean-squared-error loss.
      
    Arguments
        learning_rate: Learning rate for the optimizer Adam.
        batch_size: Batch size for the mini-batch operation.
        epochs: Number of epochs to train.
        verbose: Verbosity of the training process.
      
    Returns
        A copy of the trained model.
    """

    model_copy = model
    model_copy.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            #loss="mean_squared_error",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
    if callbacks != None:
        model_copy.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[callbacks],
                verbose=verbose,
            )
    else:
        model_copy.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
            )
    return model_copy
