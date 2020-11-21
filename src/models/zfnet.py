from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

#def maxout(inputs):
#    return slim.maxout(inputs, num_units=4096, scope='maxout')

#def maxout(inputs, num_units, axis=None):
#    shape = inputs.get_shape().as_list()
#    if axis is None:
#        # Assume that channel is the last dimension
#        axis = -1
#    num_channels = shape[axis]
#    if num_channels % num_units:
#        raise ValueError('number of features({}) is not a multiple of num_units({})'
#             .format(num_channels, num_units))
#    shape[axis] = num_units 
#    shape += [num_channels // num_units]
#    outputs = tf.reduce_max(tf.reshape(inputs, shape=shape), -1, keep_dims=False)
#    return output

def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    #batch_norm_params = {
    #    # Decay for the moving averages.
    #    'decay': 0.995,
    #    # epsilon to prevent 0s in variance.
    #    'epsilon': 0.001,
    #    # force in-place updates of mean and variance estimates
    #    'updates_collections': None,
    #    # Moving averages ends up in the trainable variables collection
    #    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    #}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                        weights_regularizer=slim.l2_regularizer(weight_decay),):
                        #normalizer_fn=slim.batch_norm,
                        #normalizer_params=batch_norm_params):
        with tf.variable_scope('zfnet', [images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                net = slim.conv2d(images, 64, [7, 7], stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
                #net = tf.nn.local_response_normalization(net, 5, 2.0, 0.0001, 0.75, 'rnorm1')

                net = slim.conv2d(net, 64, [1, 1], stride=1, scope='conv2a')
                net = slim.conv2d(net, 192, [3, 3], stride=1, scope='conv2')
                #net = tf.nn.local_response_normalization(net, 5, 2.0, 0.0001, 0.75, 'rnorm2')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool2')

                net = slim.conv2d(net, 192, [1, 1], stride=1, scope='conv3a')
                net = slim.conv2d(net, 384, [3, 3], stride=1, scope='conv3')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool3')

                net = slim.conv2d(net, 384, [1, 1], stride=1, scope='conv4a')
                net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv4')

                net = slim.conv2d(net, 256, [1, 1], stride=1, scope='conv5a')
                net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv5')

                net = slim.conv2d(net, 256, [1, 1], stride=1, scope='conv6a')
                net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv6')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='maxpool4')

                net = slim.flatten(net)
                #net = slim.dropout(net, keep_probability)

                net = slim.fully_connected(net, 8192, scope='fc1', activation_fn=None)
                net = slim.maxout(net, 4096, scope='maxout1')
                net = slim.fully_connected(net, 8192, scope='fc2', activation_fn=None)
                net = slim.maxout(net, 4096, scope='maxout2')

                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        normalizer_fn=tf.nn.l2_normalize, normalizer_params={'axis':-1}, scope='Bottleneck', reuse=False)
    return net, None
