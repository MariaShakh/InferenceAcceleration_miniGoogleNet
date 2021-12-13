import tensorflow as tf
from tensorflow.keras.initializers import *
import random

from ttreshape.reshape import *


def ttconv(inp, out_ch, d, window=(1, 1), strides=[1, 1], padding='SAME',
           initializers=GlorotUniform(), regularizers=None):
    '''
    Tensor Train decomposition for convolution
    Original idea of ttconv layer: https://github.com/timgaripov/TensorNet-TF
    :param inp: [batch_size, width, height, in_chan]
    :param out_ch: number of output channels
    :param d: number of TT kernels
    :param window: convolution window
    :param strides: strides, list of 2 ints - [sx, sy]
    :param padding: 'SAME' or 'VALID', string
    :param initializers: filters init function
    :param regularizers: filters regularizer function
    :return: output tensor

    in_imag      - input image
    in_ch_dims   - factorization for dimension of input channel,   i = 0, ..., d-1
    out_ch_dims  - output
    d            - number of TT kernels
    ranks        - ranks for TT kernels
    ranks[0] = ranks[d] = 1
    '''

    if padding == 'same':
        padding = 'SAME'
    else:
        padding = 'VALID'

    in_h, in_w, in_ch = inp.get_shape().as_list()[1:]
    in_ch_dims, out_ch_dims = vector_equal(factorize(in_ch), factorize(out_ch), n)
    in_imag = tf.reshape(inp, [-1, in_h, in_w, in_ch])

    ranks = [0] * (d + 1)
    for i in range(0, d):
        ranks[i] = random.randint(2, 4)
    ranks[d] = 1

    filter_shape = [window[0], window[1], 1, ranks[0]]  # first kernel
    if window[0] * window[1] * 1 * ranks[0] == 1:
        filters = tf.compat.v1.get_variable('filters', shape=filter_shape,
                                            initializer=Ones(),
                                            regularizer=regularizers)
    else:
        filters = tf.compat.v1.get_variable('filters', shape=filter_shape,
                                            initializer=initializers,
                                            regularizer=regularizers)
    kernels = []
    for i in range(d):
        kernels.append(initializers(shape=[out_ch_dims[i] * ranks[i + 1], ranks[i] * in_ch_dims[i]]))

    conv = filters
    for i in range(d):
        conv = tf.reshape(conv, [-1, ranks[i]])
        kernel = tf.transpose(kernels[i], [1, 0])
        kernel = tf.reshape(kernel, [ranks[i], -1])
        conv = tf.matmul(conv, kernel)

    conv_shape = [window[0], window[1]]
    order, in_order, out_order = [0, 1], [], []
    for i in range(d):
        conv_shape.append(in_ch_dims[i])
        in_order.append(2 + 2 * i)
        conv_shape.append(out_ch_dims[i])
        out_order.append(2 + 2 * i + 1)
    order += in_order + out_order
    conv = tf.reshape(conv, conv_shape)
    conv = tf.transpose(conv, order)
    conv = tf.reshape(conv, [window[0], window[1], in_ch, out_ch])

    return tf.nn.conv2d(in_imag, conv, [1] + strides + [1], padding=padding)
