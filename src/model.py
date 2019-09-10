import time

import tensorflow as tf
import numpy as np

## objectives

from util import *

## tools for building graphs
def placeholder_inputs(batch_size):
    x_pl = tf.placeholder(tf.float32, shape=(batch_size, 28*28))
    y_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return x_pl, y_pl

def weight_nat():
    weights_nat = {}
    weights_nat['W_conv1'] = weight_variable([5,5,1,32])
    weights_nat['b_conv1'] = bias_variable([32])
    weights_nat['W_conv2'] = weight_variable([5,5,32,64])
    weights_nat['b_conv2'] = bias_variable([64])
    weights_nat['W_fc1'] = weight_variable([7 * 7 * 64, 1024])
    weights_nat['b_fc1'] = bias_variable([1024])
    weights_nat['W_fc2'] = weight_variable([1024,10])
    weights_nat['b_fc2'] = bias_variable([10])


    return weights_nat

def model_nat(x_input, weights_nat):

    W_conv1 = weights_nat['W_conv1']
    b_conv1 = weights_nat['b_conv1']
    W_conv2 = weights_nat['W_conv2']
    b_conv2 = weights_nat['b_conv2']
    W_fc1 = weights_nat['W_fc1']
    b_fc1 = weights_nat['b_fc1']
    W_fc2 = weights_nat['W_fc2']
    b_fc2 = weights_nat['b_fc2']

    x_image = tf.reshape(x_input, [-1, 28, 28, 1])

    # network
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

    return pre_softmax


def weight_adv():
    output_fc1_dim = 1024 # default
    # output_fc1_dim = 2048
    weights_adv = {}
    weights_adv['W_conv1'] = weight_variable([5,5,1,32])
    weights_adv['b_conv1'] = bias_variable([32])
    weights_adv['W_conv2'] = weight_variable([5,5,32,64])
    weights_adv['b_conv2'] = bias_variable([64])
    weights_adv['W_fc1'] = weight_variable([7 * 7 * 64, 1024])
    weights_adv['b_fc1'] = bias_variable([1024])
    weights_adv['W_fc2'] = weight_variable([1024,784])
    weights_adv['b_fc2'] = bias_variable([784])

    return weights_adv


def model_adv(x_input, weights_adv):

    W_conv1 = weights_adv['W_conv1']
    b_conv1 = weights_adv['b_conv1']
    W_conv2 = weights_adv['W_conv2']
    b_conv2 = weights_adv['b_conv2']
    W_fc1 = weights_adv['W_fc1']
    b_fc1 = weights_adv['b_fc1']
    W_fc2 = weights_adv['W_fc2']
    b_fc2 = weights_adv['b_fc2']


    x_image = tf.reshape(x_input, [-1, 28, 28, 1])

    # network
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

    # h_fc2 size (-1, 784)

    output_direction =  tf.linalg.l2_normalize(h_fc2, axis = 1)

    # output_direction = h_fc2 / tf.norm(h_fc2)


    # print('h_fc2 shape', h_fc2.shape)
    # print('output_direction shape', output_direction.shape)

    return output_direction


def weight_adv_fc2():
    size_hidden1 = 1024 * 5 # default
    size_hidden2 = 1024 * 5 # default

    # output_fc1_dim = 2048
    weights_adv = {}
    weights_adv['W_fc1'] = weight_variable([784, size_hidden1])
    weights_adv['b_fc1'] = bias_variable([size_hidden1])
    weights_adv['W_fc2'] = weight_variable([size_hidden1,size_hidden2])
    weights_adv['b_fc2'] = bias_variable([size_hidden2])
    weights_adv['W_fc3'] = weight_variable([size_hidden2,784])
    weights_adv['b_fc3'] = bias_variable([784])

    return weights_adv

def model_adv_fc2(x_input, weights_adv):

    W_fc1 = weights_adv['W_fc1']
    b_fc1 = weights_adv['b_fc1']
    W_fc2 = weights_adv['W_fc2']
    b_fc2 = weights_adv['b_fc2']
    W_fc3 = weights_adv['W_fc3']
    b_fc3 = weights_adv['b_fc3']

    # network

    h_fc1 = tf.nn.relu(tf.matmul(x_input, W_fc1) + b_fc1)

    h_fc2 =  tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3

    # h_fc2 size (-1, 784)

    output_direction =  tf.linalg.l2_normalize(h_fc3, axis = 1)

    # print('h_fc2 shape', h_fc2.shape)
    # print('output_direction shape', output_direction.shape)

    return output_direction


def weight_adv_conv_2hidden():
    output_fc1_dim = 1024 * 5 # default
    output_fc2_dim = 1024 * 5

    weights_adv = {}
    weights_adv['W_conv1'] = weight_variable([5,5,1,32])
    weights_adv['b_conv1'] = bias_variable([32])
    weights_adv['W_conv2'] = weight_variable([5,5,32,64])
    weights_adv['b_conv2'] = bias_variable([64])
    weights_adv['W_fc1'] = weight_variable([7 * 7 * 64, output_fc1_dim])
    weights_adv['b_fc1'] = bias_variable([output_fc1_dim])
    weights_adv['W_fc2'] = weight_variable([output_fc1_dim, output_fc2_dim])
    weights_adv['b_fc2'] = bias_variable([output_fc2_dim])
    weights_adv['W_fc3'] = weight_variable([output_fc2_dim,784])
    weights_adv['b_fc3'] = bias_variable([784])

    return weights_adv


def model_adv_conv_2hidden(x_input, weights_adv):

    W_conv1 = weights_adv['W_conv1']
    b_conv1 = weights_adv['b_conv1']
    W_conv2 = weights_adv['W_conv2']
    b_conv2 = weights_adv['b_conv2']
    W_fc1 = weights_adv['W_fc1']
    b_fc1 = weights_adv['b_fc1']
    W_fc2 = weights_adv['W_fc2']
    b_fc2 = weights_adv['b_fc2']
    W_fc3 = weights_adv['W_fc3']
    b_fc3 = weights_adv['b_fc3']


    x_image = tf.reshape(x_input, [-1, 28, 28, 1])

    # network
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3

    # h_fc2 size (-1, 784)

    output_direction =  tf.linalg.l2_normalize(h_fc3, axis = 1)

    # output_direction = h_fc2 / tf.norm(h_fc2)


    # print('h_fc2 shape', h_fc2.shape)
    # print('output_direction shape', output_direction.shape)

    return output_direction


def weight_adv_pgd():
    # same as nat
    weights_adv = {}
    with tf.variable_scope('adv') as vs:
        weights_adv['W_conv1'] = weight_variable([5,5,1,32], name = 'W_conv1')
        weights_adv['b_conv1'] = bias_variable([32], name = 'b_conv1')
        weights_adv['W_conv2'] = weight_variable([5,5,32,64], name = 'W_conv2')
        weights_adv['b_conv2'] = bias_variable([64], name = 'b_conv2')
        weights_adv['W_fc1'] = weight_variable([7 * 7 * 64, 1024], name = 'W_fc1')
        weights_adv['b_fc1'] = bias_variable([1024], name = 'b_fc1')
        weights_adv['W_fc2'] = weight_variable([1024,10], name = 'W_fc2')
        weights_adv['b_fc2'] = bias_variable([10], name = 'b_fc2')

    return weights_adv


def model_adv_pgd(x_input, y_input ,weights_adv):
    # pgd attack network : shows gradient (= direction for best loss increasing)

    # be sure to use this with single image...

    W_conv1 = weights_adv['W_conv1']
    b_conv1 = weights_adv['b_conv1']
    W_conv2 = weights_adv['W_conv2']
    b_conv2 = weights_adv['b_conv2']
    W_fc1 = weights_adv['W_fc1']
    b_fc1 = weights_adv['b_fc1']
    W_fc2 = weights_adv['W_fc2']
    b_fc2 = weights_adv['b_fc2']

    x_image = tf.reshape(x_input, [-1, 28, 28, 1])

    # network
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    pre_softmax_adv = tf.matmul(h_fc1, W_fc2) + b_fc2

    xent_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = y_input, logits = pre_softmax_adv))

    grad = tf.gradients(xent_loss, x_input)
    grad = grad[0] # since tf.gradients give single element list
    # print(grad.shape, x_input.shape, xent_loss.shape)

    output_direction = grad / tf.norm(grad)

    return output_direction

def copy_weight_nat_to_adv_hack(sess, weights_nat, weights_adv):
    # only valid when both weights are same shape
    copy_ops = copy_ops_weight_nat_to_adv(weights_nat, weights_adv)
    sess.run(copy_ops)


def copy_ops_weight_nat_to_adv(weights_nat, weights_adv):

    copy_ops = []
    for key in weights_nat:
        op = tf.assign( weights_adv[key], tf.identity(weights_nat[key]) )
        copy_ops.append(op)

    return copy_ops



def fill_feed_dict(data_set, batch_size,images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
    }
    return images_feed, labels_feed


# utils for tensorflow

def weight_variable(shape, name = None):
    initial = tf.truncated_normal(shape, stddev=0.1)  # try 0.03, 0.05, 0.01


    if name is None:
        var = tf.Variable(initial)
    else:
        var = tf.Variable(initial, name = name)

    return var


def bias_variable(shape, init_val = 0.5, name = None):
    initial = tf.constant(init_val, shape = shape)

    if name is None:
        var = tf.Variable(initial)
    else:
        var = tf.Variable(initial, name = name)

    return var

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2( x):
    return tf.nn.max_pool(x,
                        ksize = [1,2,2,1],
                        strides=[1,2,2,1],
                        padding='SAME')
