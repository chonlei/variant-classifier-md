#
# This module contains functions of neural network to apply regression on
# the EFI measurements.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def build_dense_mlc_model(input_neurons=128, input_dim=30,
        architecture=[128, 128, 128], act_func="relu", l1l2=0.001,
        dropout=None, learning_rate=0.001):
    """
    Builds a densely connected neural network model for multi-label
    classification.

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
    elif act_func == "leaky_relu":
        activation = tf.nn.leaky_relu
    elif act_func == "sigmoid":
        activation = tf.nn.sigmoid
    elif act_func == "tanh":
        activation = tf.nn.tanh

    # Input layer
    layers = [
        # Add noise to inputs
        #tf.keras.layers.GaussianNoise(0.01, input_shape=(input_dim,)),
        # Dropout to inputs
        #tf.keras.layers.Dropout(0.2, input_shape=(input_dim,)),
        # Input dense layer
        tf.keras.layers.Dense(input_neurons,
                              input_dim=input_dim,
                              activation=activation,
                              kernel_initializer='he_uniform'),
    ]

    # Hidden layers
    num_layers = len(architecture)
    for i in range(num_layers):
        # Dropout rate 5%, meaning 1 in 20 inputs will be randomly excluded.
        if dropout:
            layers.append(tf.keras.layers.Dropout(dropout))
            # constraint: maximum norm of the weights < 3
            kernel_constraint = tf.keras.constraints.MaxNorm(3)
        else:
            kernel_constraint = None

        # Hidden dense layer
        layers.append(tf.keras.layers.Dense(
            architecture[i],
            activation=activation,
            # regularisation
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1l2),
            # constraint
            kernel_constraint=kernel_constraint,
        ))
        # Add noise to hidden layers
        #layers.append(tf.keras.layers.GaussianNoise(0.01))

    # Output layer: sigmoid to give probability-like outputs
    layers.append(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))

    model = tf.keras.models.Sequential(layers)
    model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            loss='binary_crossentropy',
            metrics=["accuracy",  # good for balanced data, and that we care equally about TN and TP.
                     #tf.keras.metrics.AUC(),  # similar to accuracy.
                     tfa.metrics.FBetaScore(num_classes=2, average='micro', beta=2.0),  # F2-score, good for if we care recalling positive observations more than precision.
                    ],
        )
    return model


def build_dense_classification_model(input_neurons=128, input_dim=32*32,
        architecture=[64, 32, 16], act_func="relu", l1l2=0.001):
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
        layers.append(tf.keras.layers.Dense(
            architecture[i],
            activation=activation,
            # regularisation
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1l2),
        ))
    layers.append(tf.keras.layers.Dense(2))

    model = tf.keras.models.Sequential(layers)
    return model


def build_cnn_classification_model(input_neurons=(16, 16), input_dim=1,
        architecture=[16, 32, 64], act_func="relu", l1l2=0.005, dropout=0.3):
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
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1l2),
        ))
        layers.append(tf.keras.layers.MaxPooling2D())
    layers.append(tf.keras.layers.Dropout(dropout))  # regularisation
    layers.append(tf.keras.layers.Dense(64, activation=activation))
    layers.append(tf.keras.layers.Dense(2))

    model = tf.keras.models.Sequential(layers)
    return model


def train_classification_model(model, x_train, y_train, callbacks=None,
        learning_rate=0.001, batch_size=1, epochs=20, verbose=0, val=None):
    """
    Compiles and trains a given Keras ``model`` with the given data
    (``x_train``, ``y_train``) for regression. Assumes Adam optimizer for this
    implementation. Assumes mean-squared-error loss.
      
    Arguments
        learning_rate: Learning rate for the optimizer Adam.
        batch_size: Batch size for the mini-batch operation.
        epochs: Number of epochs to train.
        verbose: Verbosity of the training process.
        val: Validation data (tuple).
      
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
                validation_data=val,
            )
    else:
        model_copy.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                validation_data=val,
            )
    return model_copy
