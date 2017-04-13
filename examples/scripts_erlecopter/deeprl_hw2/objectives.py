"""Loss functions."""

import tensorflow as tf
import semver

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
    # logic
    # if fabs(diff) <= delta:
    #     return 0.5 * diff * diff.transpose;
    # else:
    #     return delta*(fabs(diff) - 0.5*delta);
    
    delta = max_grad # somewhere in piazza. does this need to be a tf constant
    diff = tf.abs(y_true-y_pred)
    huber_if_diff_less_than_delta = 0.5*tf.square(diff)
    huber_if_diff_more_than_delta = delta*(diff - 0.5*delta)
    is_diff_less_than_delta = tf.less_equal(diff, delta)
    final = tf.where(is_diff_less_than_delta, huber_if_diff_less_than_delta, huber_if_diff_more_than_delta) 
    return final

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
    return tf.reduce_mean(huber_loss(y_true, y_pred, max_grad)) # todo no need to specify axis right
