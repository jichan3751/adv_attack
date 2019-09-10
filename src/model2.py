import time

import tensorflow as tf
import numpy as np

## objectives

from util import *

###

def layers(x_input, config):
    if config["model"] == "fcn":
        return layers_fcn(x_input, config)
    elif config["model"] == "cnn":
        return layers_cnn(x_input, config)
    else:
        assert 0

def layers_fcn(x_input, config):
    INPUT_FC_SIZE = 784
    OUTPUT_FC_SIZE = 784

    sizes = config['fc_hidden_layers']

    weights = {}
    acts = {} # activations

    for i in range(len(sizes)+1):
        input_size = INPUT_FC_SIZE if i == 0 else sizes[i-1]
        output_size = OUTPUT_FC_SIZE if i == len(sizes) else sizes[i]

        weights['W_fc%d'%(i+1)] = weight_variable([input_size, output_size])
        weights['b_fc%d'%(i+1)] = bias_variable([output_size])

        W_fc = weights['W_fc%d'%(i+1)]
        b_fc = weights['b_fc%d'%(i+1)]

        h_fc0 = x_input if i == 0 else acts['h_fc%d'%(i)]


        if i != len(sizes):
            acts['h_fc%d'%(i+1)] = tf.nn.relu(tf.matmul(h_fc0, W_fc) + b_fc)
        else:
            acts['h_fc%d'%(i+1)] = tf.matmul(h_fc0, W_fc) + b_fc

    h_fc_out = acts['h_fc%d'%(len(sizes)+1)]


    if config['norm'] == 'l2':
        output_direction =  tf.linalg.l2_normalize(h_fc_out, axis = 1)
    elif config['norm'] == 'linf':
        assert 0, "not yet implemented"
    else:
        assert 0

    return output_direction, weights


### examples 3 fc hidden ######

def layers_adv_fc3(x_input):
    def weight_adv():
        size_hidden1 = 1024 * 5 # default
        size_hidden2 = 1024 * 10 # default
        size_hidden3 = 1024 * 5 # default

        # output_fc1_dim = 2048
        weights_adv = {}
        weights_adv['W_fc1'] = weight_variable([794, size_hidden1])
        weights_adv['b_fc1'] = bias_variable([size_hidden1])
        weights_adv['W_fc2'] = weight_variable([size_hidden1,size_hidden2])
        weights_adv['b_fc2'] = bias_variable([size_hidden2])
        weights_adv['W_fc3'] = weight_variable([size_hidden2,size_hidden3])
        weights_adv['b_fc3'] = bias_variable([size_hidden3])
        weights_adv['W_fc4'] = weight_variable([size_hidden3,784])
        weights_adv['b_fc4'] = bias_variable([784])

        return weights_adv

    weights_adv = weight_adv()

    W_fc1 = weights_adv['W_fc1']
    b_fc1 = weights_adv['b_fc1']
    W_fc2 = weights_adv['W_fc2']
    b_fc2 = weights_adv['b_fc2']
    W_fc3 = weights_adv['W_fc3']
    b_fc3 = weights_adv['b_fc3']
    W_fc4 = weights_adv['W_fc4']
    b_fc4 = weights_adv['b_fc4']

    # network

    h_fc1 = tf.nn.relu(tf.matmul(x_input, W_fc1) + b_fc1)

    h_fc2 =  tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    h_fc3 =  tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    h_fc4 = tf.matmul(h_fc3, W_fc4) + b_fc4

    # h_fc2 size (-1, 784)

    output_direction =  tf.linalg.l2_normalize(h_fc4, axis = 1)

    # print('h_fc2 shape', h_fc2.shape)
    # print('output_direction shape', output_direction.shape)

    return output_direction, weights_adv

def layers_cnn(x_input, config):

    sizes = config['fc_hidden_layers']

    weights = {}
    acts = {} # activations

    # still problematic..
    x_image = tf.reshape(x_input, [-1, 28, 28, 1])

    W_conv1 = weights['W_conv1'] = weight_variable([5,5,1,32])
    b_conv1 = weights['b_conv1'] = bias_variable([32])
    W_conv2 = weights['W_conv2'] = weight_variable([5,5,32,64])
    b_conv2 = weights['b_conv2'] = bias_variable([64])

    # conv
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    INPUT_FC_SIZE = 7 * 7 * 64
    OUTPUT_FC_SIZE = 784

    for i in range(len(sizes)+1):
        input_size = INPUT_FC_SIZE if i == 0 else sizes[i-1]
        output_size = OUTPUT_FC_SIZE if i == len(sizes) else sizes[i]

        weights['W_fc%d'%(i+1)] = weight_variable([input_size, output_size])
        weights['b_fc%d'%(i+1)] = bias_variable([output_size])

        W_fc = weights['W_fc%d'%(i+1)]
        b_fc = weights['b_fc%d'%(i+1)]

        h_fc0 = h_pool2_flat if i == 0 else acts['h_fc%d'%(i)]

        if i != len(sizes):
            acts['h_fc%d'%(i+1)] = tf.nn.relu(tf.matmul(h_fc0, W_fc) + b_fc)
        else: # the last activation
            acts['h_fc%d'%(i+1)] = tf.matmul(h_fc0, W_fc) + b_fc


    h_fc_out = acts['h_fc%d'%(len(sizes)+1)]

    if config['norm'] == 'l2':
        output_direction =  tf.linalg.l2_normalize(h_fc_out, axis = 1)
    elif config['norm'] == 'linf':
        assert 0, "not yet implemented"
    else:
        assert 0

    return output_direction, weights

### examples 1 fc hidden ######

def layers_conv_fc1(x_input):
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

    weights_adv = weight_adv()

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

    return output_direction, weights_adv


############ utils for tensorflow ##############3

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
