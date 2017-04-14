"""Loss functions."""

import tensorflow as tf
import semver
import numpy as np
import pdb
import keras.backend as K

def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    error = tf.abs(y_true - y_pred)
    condition = tf.less_equal(error, max_grad)
    loss = tf.where(condition, 0.5*tf.square(error), max_grad*error - 0.5*max_grad**2)
    return loss

def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    loss = huber_loss(y_true, y_pred, max_grad)
    return tf.reduce_mean(loss)