import os
import time
from pprint import pprint

import tensorflow as tf
import numpy as np


from util import *
from model import *


BATCH_SIZE = 10

def main():
    # model_dir = 'models/natural'
    model_dir = 'models/adv_trained'

    model_file = tf.train.latest_checkpoint(model_dir)
    weights_nat = weight_nat()
    weights_adv = weight_adv()

    ts = [bias_variable([1], init_val = 28.0) for _ in range(BATCH_SIZE)]

    pprint(get_current_variables())

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(list(weights_nat.values()))

    sess = tf.Session()
    sess.run(init)

    saver.restore(sess, model_file)

    # import ipdb; ipdb.set_trace()


def fetch_checkpoint_hack(sess, weights_nat, model = 'nat'):
    # check if weigh_nat() is already defined.
    # restores Variable_1 ~ Variable_7 to weights_nat
    if model == 'nat':
        model_dir = 'models/natural'
    elif model == 'adv':
        model_dir = 'models/adv_trained'
    elif model == 'adv':
        model_dir = 'models/secret'
    else:
        assert 0

    print('load model from ', model_dir)

    model_file = tf.train.latest_checkpoint(model_dir)

    saver = tf.train.Saver(list(weights_nat.values()))
    saver.restore(sess, model_file)

    model_dir = 'models/natural/checkpoint-24900'
    # model_dir = 'models/adv_trained/checkpoint-99900'


def check_checkpoint_file():
    from tensorflow.python import pywrap_tensorflow
    import os

    model_dir = 'models/natural/checkpoint-24900'
    # model_dir = 'models/adv_trained/checkpoint-99900'

    reader = pywrap_tensorflow.NewCheckpointReader(model_dir)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # print(var_to_shape_map)

    # for key in var_to_shape_map:
    for key in sorted(var_to_shape_map.keys()):
        print("tensor_name: ", key)
        tt = reader.get_tensor(key)
        print(tt.shape) # Remove this is you want to print only variable names




def get_current_variables():
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)



if __name__ == '__main__':
    # main()
    check_checkpoint_file()