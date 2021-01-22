#
# This module contains functions of autoencoder to apply regression on
# the EFI measurements.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import tensorflow as tf


class KerasEncoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim, original_dim):
    super(KerasEncoder, self).__init__()
    self.hidden_layer = tf.keras.layers.Dense(
      units=int((intermediate_dim + original_dim) // 2),
      activation=tf.nn.relu,
      kernel_initializer='he_uniform'
    )
    self.output_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.sigmoid
    )

  def call(self, input_features):
    activation = self.hidden_layer(input_features)
    return self.output_layer(activation)


class KerasDecoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim, original_dim):
    super(KerasDecoder, self).__init__()
    self.hidden_layer = tf.keras.layers.Dense(
      units=int((intermediate_dim + original_dim) // 2),
      activation=tf.nn.relu,
      kernel_initializer='he_uniform'
    )
    self.output_layer = tf.keras.layers.Dense(
      units=original_dim,
      activation=tf.nn.sigmoid
    )

  def call(self, code):
    activation = self.hidden_layer(code)
    return self.output_layer(activation)


class Autoencoder(tf.keras.Model):
  def __init__(self, intermediate_dim, original_dim):
    super(Autoencoder, self).__init__()
    self.encoder = KerasEncoder(intermediate_dim=intermediate_dim,
                                original_dim=original_dim)
    self.decoder = KerasDecoder(intermediate_dim=intermediate_dim,
                                original_dim=original_dim)

  def call(self, input_features):
    code = self.encoder(input_features)
    reconstructed = self.decoder(code)
    return reconstructed


def _loss(model, original):
  reconstruction_error = tf.reduce_mean(
    tf.square(tf.subtract(model(original), original))
  )
  return reconstruction_error


def _train(loss, model, opt, original):
  with tf.GradientTape() as tape:
    gradients = tape.gradient(loss(model, original), model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)


class Encoder(object):
    """
    Apply transformation using neural network autoencoder.
    """
    def __init__(self, n_components):
        """
        Parameters:
            n_components: int, number of components to be encoded.
        """
        self._n_components = int(n_components)

    def fit(self, X, epochs=100, verbose=True):
        """
        Parameters:
            X: array-like, shape [n_samples, n_features]. The data to be
               fitted.
        """
        X = np.array(X, copy=True)
        n_s, n_f = X.shape
        self._autoencoder = Autoencoder(self._n_components, n_f)
        '''
        opt = tf.optimizers.Adam(learning_rate=5e-3)
        for epoch in range(epochs):
            _train(_loss, self._autoencoder, opt, X)
            if verbose:
                print('Epoch:', epoch, '| loss:', _loss(self._autoencoder, X))
        '''
        self._autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.001),
            loss="mean_squared_error",
            metrics=["accuracy"],
        )
        self._autoencoder.fit(
            X,
            X,
            epochs=epochs,
            batch_size=1,
            verbose=verbose,
            validation_data=None,
        )

    def transform(self, X):
        """
        Parameters:
            X: array-like, shape [n_samples, n_features]. The data to be
               transformed.
        """
        return self._autoencoder.encoder(X).numpy()

    def inverse_transform(self, X):
        """
        Parameters:
            X: array-like, shape [n_samples, n_features]. The data to be
               inverse-transformed.
        """
        return self._autoencoder.decoder(X).numpy()
