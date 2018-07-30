import tensorflow as tf
import re
import numbers

import numpy as np


TOWER_NAME = 'tower'

def activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def variable_on_device(name, shape, initializer=None, dtype=tf.float32, regularizer=None, device=None, trainable=True):
    """Get variable that located on given device.

    Args:
        name (str): Variable name.
        shape (tuple): variable shape.
        initializer ([type]): initializer
        device (str, optional): Defaults to '/cpu:0'. device.
        trainable (bool, optional): Defaults to True. If trainable.

    Returns:
        Tensor: The variable tensor.
    """
    with tf.device(device):
        var = tf.get_variable(
            name, shape, dtype=dtype, initializer=initializer, regularizer=regularizer, trainable=trainable)
    return var


def l2_regularizer(scale, scope=None):
    """Returns a function that can be used to apply L2 regularization to weights.

    Small values of L2 can help prevent overfitting the training data.

    Args:
      scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
      scope: An optional scope name.

    Returns:
      A function with signature `l2(weights)` that applies L2 regularization.

    Raises:
      ValueError: If scale is negative or if scale is not a float.
    """
    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % (scale,))
    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                             scale)
        if scale == 0.:
            # logging.info('Scale of 0 disables regularizer.')
            return lambda _: None

    def l2(weights):
        """Applies l2 regularization to weights."""
        with tf.name_scope(scope, 'l2_regularizer', [weights]) as name:
            my_scale = tf.convert_to_tensor(
                scale, dtype=weights.dtype.base_dtype, name='scale')
            return tf.multiply(my_scale, tf.nn.l2_loss(weights), name=name)

    return l2


def variable_with_weight_decay(name, shape, dtype=tf.float32, stddev=0.1, wd=0.0, initializer=None,  device=None, trainable=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    if initializer is None:
        initializer = tf.truncated_normal_initializer(
            stddev=stddev, dtype=dtype)
    if wd != 0.0 and trainable:
        regularizer = l2_regularizer(wd)
    else:
        regularizer = None
    var = variable_on_device(
        name, shape, initializer, dtype, regularizer, device, trainable)
    '''
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection(tf.GraphKeys.LOSSES, weight_decay)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)
    '''
    return var


def batch_normalization_layer(input_layer, trainable):
    dimension = input_layer.get_shape().as_list()[-1]
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = variable_on_device('beta', dimension, initializer=tf.constant_initializer(0.0, tf.float32), trainable=trainable)
    gamma = variable_on_device('gamma', dimension, initializer=tf.constant_initializer(1.0, tf.float32), trainable=trainable)
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, 0.001)

    return bn_layer

def instance_normalization_layer(input_layer):
    mean, variance = tf.nn.moments(input_layer, axes=[1, 2], keep_dims=True)
    return tf.nn.batch_normalization(input_layer, mean, variance, None, None, 0.001)

def conv2d_layer(input_layer, kernel_size, out_channels, stride, wd, trainable, as_is=False):
    in_channels = input_layer.get_shape().as_list()[-1]
    initializer = None
    if as_is:
        k = np.zeros((kernel_size, kernel_size, in_channels, out_channels))
        n = int((kernel_size - 1) / 2)
        k[n+1-stride: n+stride, n+1-stride: n+stride] = np.ones((2*stride-1, 2*stride-1, in_channels, out_channels))
        initializer = tf.constant(k, dtype=tf.float32)
    
    kernel = variable_with_weight_decay(name='kernel', shape=None if as_is else (kernel_size, kernel_size, in_channels, out_channels), wd=wd, initializer=initializer, trainable=trainable)
    conv_layer = tf.nn.conv2d(input_layer, kernel, strides=(1, stride, stride, 1), padding='SAME')

    return conv_layer

def style_conv2d_layer(input_layer, kernel_size, out_channels, stride, wd, trainable):
    in_channels = input_layer.get_shape().as_list()[-1]
    if out_channels == 0:
        out_channels = in_channels * stride
    
    kernel = variable_with_weight_decay(name='kernel', shape=(kernel_size, kernel_size, in_channels, out_channels), wd=wd, initializer=None, trainable=trainable)
    conv_layer = tf.nn.conv2d(input_layer, kernel, strides=(1, stride, stride, 1), padding='SAME')
    biases = variable_on_device('biases', None, tf.constant(0.0, shape=[out_channels], dtype=tf.float32), trainable=trainable)
    conv_layer = tf.nn.bias_add(conv_layer, biases)
    conv_layer = instance_normalization_layer(conv_layer)
    conv_layer = tf.nn.relu(conv_layer)

    return conv_layer

def weighted_add(input_layers, weights, name=None):
    weighted = []
    for input_layer, weight in zip(input_layers, weights):
        weighted.append(tf.multiply(input_layer, weight))

    return tf.add_n(weighted, name=name)

def upsample_layer(input_layer):
    shape = input_layer.get_shape().as_list()

    resized = tf.image.resize_images(input_layer, (shape[1]*2, shape[2]*2))

    return resized

def downsample_layer(input_layer):
    shape = input_layer.get_shape().as_list()

    resized = tf.image.resize_images(input_layer, (int(shape[1]/2), int(shape[2]/2)))

    return resized


def upsample_conv2d_layer(input_layer, kernel_size, out_channels, wd, trainable):
    shape = input_layer.get_shape().as_list()
    if out_channels == 0:
        out_channels = int(shape[-1] / 2)

    resized = tf.image.resize_images(input_layer, (shape[1]*2, shape[2]*2), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    conv_layer = conv2d_layer(resized, kernel_size, out_channels, 1, wd, trainable)

    return conv_layer

def vgg_conv2d_layer(input_layer, kernel_size, out_channels, stride, wd, trainable, as_is=False):
    in_channels = input_layer.get_shape().as_list()[-1]
    initializer = None
    if as_is:
        k = np.zeros((kernel_size, kernel_size, in_channels, out_channels))
        n = int((kernel_size - 1) / 2)
        k[n+1-stride: n+stride, n+1-stride: n+stride] = np.ones((2*stride-1, 2*stride-1, in_channels, out_channels))
        initializer = tf.constant(k, dtype=tf.float32)
    
    kernel = variable_with_weight_decay(name='weights', shape=None if as_is else (kernel_size, kernel_size, in_channels, out_channels), wd=wd, initializer=initializer, trainable=trainable)
    conv_layer = tf.nn.conv2d(input_layer, kernel, strides=(1, stride, stride, 1), padding='SAME')

    biases = variable_on_device('biases', None, tf.constant(1.0 if as_is else 0.0, shape=[out_channels], dtype=tf.float32), trainable=trainable)
    conv_layer = tf.nn.bias_add(conv_layer, biases)

    conv_layer = tf.nn.relu(conv_layer)

    return conv_layer

def basic_block(input_layer, out_channels, stride, wd, trainable):
    in_channels = input_layer.get_shape().as_list()[-1]
    if out_channels == 0:
        out_channels = in_channels

    with tf.variable_scope('conv1') as scope:
        x = conv2d_layer(input_layer, 3, out_channels, stride, wd, trainable)
    with tf.variable_scope('bn1') as scope:
        x = batch_normalization_layer(x, trainable)
    with tf.variable_scope('relu1') as scope:
        x = tf.nn.relu(x)
    with tf.variable_scope('conv2') as scope:
        x = conv2d_layer(x, 3, out_channels, 1, wd, trainable)
    with tf.variable_scope('bn2') as scope:
        x = batch_normalization_layer(x, trainable)
    # x = tf.nn.relu(x)
    with tf.variable_scope('shortcut') as scope:
        if stride != 1 or in_channels != out_channels:
            with tf.variable_scope('conv') as scope:
                shortcut = conv2d_layer(input_layer, 1, out_channels, stride, 0, trainable)
            with tf.variable_scope('bn') as scope:
                shortcut = batch_normalization_layer(shortcut, trainable)
        else:
            shortcut = input_layer
    with tf.variable_scope('merge') as scope:
        x = tf.add(x, shortcut)
    with tf.variable_scope('relu2') as scope:
        x = tf.nn.relu(x)

    return x