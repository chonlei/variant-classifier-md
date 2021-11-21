#
# This module contains functions of autoencoder to apply regression on
# the EFI measurements.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import tensorflow as tf

K = tf.keras.backend


# For a simple tutorial/example, see e.g.
# https://towardsdatascience.com/build-the-right-autoencoder-tune-and-optimize-using-pca-principles-part-i-1f01f821999b
# https://blog.keras.io/building-autoencoders-in-keras.html

class DenseTied(tf.keras.layers.Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        self.tied_to = tied_to
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = tf.keras.backend.transpose(self.tied_to.kernel)
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = tf.keras.backend.dot(inputs, self.kernel)
        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class WeightsOrthogonalityConstraint(tf.keras.constraints.Constraint):
    def __init__(self, encoding_dim, weightage = 1.0, axis = 0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis
        self.eye = K.eye(encoding_dim)

    def weights_orthogonality(self, w):
        if(self.axis==1):
            w = K.transpose(w)
        if(self.encoding_dim > 1):
            wt = K.transpose(w)
            m = K.dot(wt, w) - self.eye
            return self.weightage * K.sqrt(K.sum(K.square(m)))
        else:
            m = K.sum(w ** 2) - 1.
            return m

    def __call__(self, w):
        return self.weights_orthogonality(w)


class UncorrelatedFeaturesConstraint(tf.keras.constraints.Constraint):

    def __init__(self, encoding_dim, weightage=1.0):
        self.encoding_dim = encoding_dim
        self.eye = K.eye(encoding_dim)
        self.weightage = weightage

    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))

        x_centered = tf.stack(x_centered_list)
        covariance = K.dot(x_centered, K.transpose(x_centered)) / \
            tf.cast(x_centered.get_shape()[0], tf.float32)

        return covariance

    # Constraint penalty
    def uncorrelated_feature(self, x):
        if(self.encoding_dim <= 1):
            return 0.0
        else:
            diagonal = tf.math.multiply(self.covariance, self.eye)
            output = K.sum(K.square(self.covariance - diagonal))
            return output

    def __call__(self, x):
        self.covariance = self.get_covariance(x)
        return self.weightage * self.uncorrelated_feature(x)


class KerasEncoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim, original_dim, units=[], l1l2=None,
               dropout=None):
    super(KerasEncoder, self).__init__()
    if not units:
        units = [
          int((intermediate_dim + original_dim) // 4),
          min(int((intermediate_dim + original_dim) // 16),
              intermediate_dim),
        ]
    self.hidden_layers = []
    for unit in units:
      self.hidden_layers.append(
        tf.keras.layers.Dense(
          units=unit,
          activation=tf.nn.leaky_relu,
          #kernel_initializer='he_uniform',
          #kernel_regularizer=WeightsOrthogonalityConstraint(unit, weightage=1., axis=0),
          #activity_regularizer=UncorrelatedFeaturesConstraint(unit, weightage=1.),
          activity_regularizer=tf.keras.regularizers.l1_l2(l1l2),
          kernel_constraint=tf.keras.constraints.UnitNorm(axis=0),
          use_bias=True,
        )
      )
      if dropout:
        self.hidden_layers.append(tf.keras.layers.Dropout(dropout))
    self.output_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      #activation=tf.nn.sigmoid,
      activation=tf.nn.leaky_relu,
      activity_regularizer=tf.keras.regularizers.l1_l2(l1l2),
      #kernel_regularizer=WeightsOrthogonalityConstraint(intermediate_dim, weightage=1., axis=0),
      #activity_regularizer=UncorrelatedFeaturesConstraint(intermediate_dim, weightage=1.),
      kernel_constraint=tf.keras.constraints.UnitNorm(axis=0),
      use_bias=True,
    )

  def call(self, input_features):
    activation = input_features
    for hidden_layer in self.hidden_layers:
      activation = hidden_layer(activation)
    return self.output_layer(activation)


class KerasDecoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim, original_dim, units=[], l1l2=None,
               dropout=None):
    super(KerasDecoder, self).__init__()
    if not units:
        units = [
          min(int((intermediate_dim + original_dim) // 16),
              intermediate_dim),
          int((intermediate_dim + original_dim) // 4),
        ]
    self.hidden_layers = []
    for unit in units:
      self.hidden_layers.append(
        tf.keras.layers.Dense(
          units=unit,
          activation=tf.nn.leaky_relu,
          #kernel_initializer='he_uniform',
          #kernel_regularizer=WeightsOrthogonalityConstraint(unit, weightage=1., axis=0),
          #activity_regularizer=UncorrelatedFeaturesConstraint(unit, weightage=1.),
          activity_regularizer=tf.keras.regularizers.l1_l2(l1l2),
          kernel_constraint=tf.keras.constraints.UnitNorm(axis=0),
          use_bias=False,
        )
      )
      if dropout:
        self.hidden_layers.append(tf.keras.layers.Dropout(dropout))
    #self.output_layer = DenseTied(
    self.output_layer = tf.keras.layers.Dense(
      units=original_dim,
      #activation=tf.nn.sigmoid,
      activation=tf.nn.leaky_relu,
      #tied_to=keras_encoder.hidden_layers[0],
      activity_regularizer=tf.keras.regularizers.l1_l2(l1l2),
      kernel_constraint=tf.keras.constraints.UnitNorm(axis=0),
      use_bias=False,
    )

  def call(self, code):
    activation = code
    for hidden_layer in self.hidden_layers:
      activation = hidden_layer(activation)
    return self.output_layer(activation)


class Autoencoder(tf.keras.Model):
  def __init__(self, intermediate_dim, original_dim, units=[], l1l2=None,
               dropout=None):
    super(Autoencoder, self).__init__()
    self.encoder = KerasEncoder(intermediate_dim=intermediate_dim,
                                original_dim=original_dim,
                                units=units,
                                l1l2=l1l2,
                                dropout=dropout)
    self.decoder = KerasDecoder(intermediate_dim=intermediate_dim,
                                original_dim=original_dim,
                                units=units[::-1],
                                l1l2=l1l2,
                                dropout=dropout)

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


def _create_lag(data, shape, lag, copy=True):
    '''
    Create time step lag for the data
    '''
    data_tmp = np.array(data.reshape(shape), copy=True)
    out = data_tmp[:, :-lag, :]
    out_lag = data_tmp[:, lag:, :]
    out = out.reshape(shape[0] * (shape[1] - lag), shape[2])
    out_lag = out_lag.reshape(shape[0] * (shape[1] - lag), shape[2])
    del(data_tmp)
    return out, out_lag


def _whiten_data(data, mean, diag_eigw, eigv):
    '''
    Whiten a Tensor in the PCA basis.
    '''
    out = data - mean
    out = np.dot(np.dot(diag_eigw, eigv.T), out)
    return out


class Encoder(object):
    """
    Apply transformation using neural network autoencoder.
    """
    def __init__(self, n_components, units=[], l1l2=None, dropout=None):
        """
        Parameters:
            n_components: int, number of components to be encoded.
        Optional:
            units: list of int, sizes of hidden layers.
        """
        self._n_components = int(n_components)
        self._units = units
        self._l1l2 = l1l2
        self._dropout = dropout

    def fit(self, X, Y=None, lag=None, shape=None, epochs=100, batch_size=512,
            verbose=True):
        """
        Parameters:
            X: array-like, shape [n_samples, n_features]. The data to be
               fitted.
        Optional:
            Y: array-like, training data, same shape as X
            lag: int, specifies the lag in time steps (default: None).
            shape: tuple, specifies the shape of the data
                   (number of mutants, number of frames/time steps,
                    number of dof).
            epochs: int, training epochs.
            batch_size: int, training batch size.
            verbose: bool, print out to console.
        """
        X = np.array(X, copy=True)
        n_s, n_f = X.shape
        self._autoencoder = Autoencoder(self._n_components, n_f,
                                        units=self._units,
                                        l1l2=self._l1l2,
                                        dropout=self._dropout,)
        if (Y is None) and not lag:
            if verbose: print('Applying (normal) autoencoder')
            Y = np.array(X, copy=True)
        elif (Y is None):
            if verbose: print('Applying time %s-lagged autoencoder' % lag)
            X, Y = _create_lag(X, shape=shape, lag=lag, copy=True)
        else:
            if verbose: print('Applying autoencoder with input data Y')
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
            #loss="binary_crossentropy",
            metrics=["mean_absolute_error"],
        )
        #self._autoencoder.summary()
        self._autoencoder.fit(
            X,
            Y,
            epochs=epochs,
            batch_size=batch_size,
            #batch_size=10000,
            shuffle=True,
            verbose=verbose,
            validation_data=None,
        )

        # Prepare for whiten latent data during transform
        from scipy import linalg
        l = self._autoencoder.encoder(X).numpy()
        self._mean = np.mean(l, axis=0)
        cov = np.cov(l - self._mean, rowvar=False, bias=True)
        w, self._eigv = linalg.eig(cov)
        self._diag_eigv = np.diag(1. / ((w+.1e-5)**0.5))

    def save(self, path):
        self._autoencoder.save(path, save_format='tf')

    def load(self, path):
        self._autoencoder = tf.keras.models.load_model(path)

    def transform(self, X, whiten=False):
        """
        Parameters:
            X: array-like, shape [n_samples, n_features]. The data to be
               transformed.
            whiten: bool, set to True to whiten the transformed data.
        """
        if whiten:
            return _whiten_data(self._autoencoder.encoder(X).numpy(),
                                self._mean, self._diag_eigw, self._eigv)
        else:
            return self._autoencoder.encoder(X).numpy()

    def inverse_transform(self, X):
        """
        Parameters:
            X: array-like, shape [n_samples, n_features]. The data to be
               inverse-transformed.
        """
        return self._autoencoder.decoder(X).numpy()
