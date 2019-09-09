import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

from util import *
from model2 import *
from load_model import *

def main(RANK, eps):

    lists_fc_hidden_layers = [
            [ 1024*2, 1024*5, 1024*2, 1024*2],
            [ 1024*2, 1024*5, 1024*2],
            [ 1024*2, 1024*5],
            [ 1024*2]
    ]


    config = {
            "max_epoch": 200,
            "lr": 0.2, # base lr (default 0.1)
            "momentum": 0.9,
            "batch_size":100,
            "eps" : eps,
            "lr_multipliers":[1.0, 0.9 ,0.8, 0.7, 0.6],
            "model":"cnn", #"fcn", "cnn"
            "fc_hidden_layers": lists_fc_hidden_layers[RANK]
        }

    # override (debug)
    # config["max_epoch"] = 1
    # config["lr_multipliers"] = [1.0, 1.0]

    print('Running EXP for:')
    print(config)

    AA = Trainer(config)

    with Timer(name = 'initialize') as t:
        AA.setup()

    with Timer(name = 'run_train') as t:
        AA.run_train_pbt(rank=RANK)
        AA.save()
        # AA.run_train_single_batch()
        # AA.tmp11()
        # AA.check_initial_loss_per_train_image()


class Trainer(object):
    def __init__(self, config, seed = 0):
        self.seed = seed
        self.config = config

        """
        config = {
            "lr": 0.1 # base lr
            "lambda" : 0.1
            "max_epoch": 100
            "lr_multipliers":[1.0, 0.9, 0.5]
        }
        """

        self.model = MyModel(config=config)
        self.train_dataset = MnistTrain(config = config)
        self.test_dataset = MnistTest(config = config)

    def setup(self):
        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)

        self.model.setup()
        self.train_dataset.setup(batch_size=self.config['batch_size'], shuffle=1)
        self.test_dataset.setup(batch_size=self.config['batch_size'])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=0)

        fetch_checkpoint_hack(self.sess, self.model.weights_nat, model='adv') # load naturual model

    def run_train_pbt(self, rank=0 ,verbose = 1):
        SMALL_LR_THRESHOLD = 10 ** -5
        CKPT = 'ckpt%d/'%rank

        c = self.config

        infos_history = []
        lrm_history = []
        current_lr = c['lr']

        self.saver.save(self.sess, CKPT+"model_best.ckpt")

        stats = self.model.eval(self.sess, self.train_dataset)
        train_stats = {
            'train_loss':stats['val_loss'],
            'train_acc':stats['val_acc']
            }

        val_stats = self.model.eval(self.sess,self.test_dataset)

        info = {}
        info.update(train_stats)
        info.update(val_stats)

        print('Epoch %d train l %.3f a %.3f test l %.3f a %.3f lr %.3f'%
                (-1,info['train_loss'],info['train_acc'],info['val_loss'],info['val_acc'],-1.0) )

        for epoch in range(c['max_epoch']):

            t0 = time.time()

            weight_sets = []
            cur_infos = []

            if epoch == 0:
                lr_multipliers = [1.0, 0.9 ,0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2 , 0.1]

            else:
                lr_multipliers = c['lr_multipliers']

            # lr_multipliers = [1.0, 0.9] # debug

            for lr_i in range(len(lr_multipliers)):
                # print('epoch, lr_i',epoch, lr_i)

                # if epoch ==0 and lr_i==4:
                #     import ipdb; ipdb.set_trace()

                np.random.seed(self.seed+epoch);
                tf.random.set_random_seed(self.seed+epoch)

                t00 = time.time()
                lr = current_lr * lr_multipliers[lr_i]

                self.saver.restore(self.sess, CKPT+"model_best.ckpt")

                train_stats = self.model.train(self.sess, self.train_dataset, learning_rate = lr, verbose = 0)
                val_stats = self.model.eval(self.sess,self.test_dataset)

                info = {"lr":lr, "best":False}
                info.update(train_stats)
                info.update(val_stats)

                cur_infos.append(info)

                self.saver.save(self.sess, CKPT+"model_%d.ckpt"%(lr_i))

                t01 = time.time()
                # if verbose: print('  epoch %d train %.3f test %.3f lr %.3f took %.2fsec'%
                #     (epoch,info['train_loss'],info['val_loss'],info['lr'], t01 - t00 ))
                if verbose: print('  epoch %d train l %f a %f test l %f a %f lr %f took %.2fsec'%
                    (epoch,info['train_loss'],info['train_acc'],info['val_loss'],info['val_acc'],info['lr'], t01 - t00 ))


            b_i = self._get_best_config_index(cur_infos, metric='val_loss', min_or_max='max')
            lrm_history.append(lr_multipliers[b_i])
            cur_infos[b_i]['best'] = True
            current_lr = cur_infos[b_i]['lr']

            infos_history.append(cur_infos)
            info = cur_infos[b_i]
            self.saver.restore(self.sess, CKPT+"model_%d.ckpt"%(b_i))
            time.sleep(2)
            self.saver.save(self.sess, CKPT+"model_best.ckpt")

            t1 = time.time()

            print('Epoch %d train l %.3f a %.3f test l %.3f a %.3f lr %.3f took %.2fsec'%
                (epoch,info['train_loss'],info['train_acc'],info['val_loss'],info['val_acc'],info['lr'], t1 - t0 ))

            if info['lr'] < SMALL_LR_THRESHOLD:
                print('stop training since lr become too small')
                break

            # import ipdb; ipdb.set_trace()

        self.infos_history = infos_history
        self.lrm_history = lrm_history

    def save(self):
        # weights = self.model.get_weights(self.sess)
        # import ipdb; ipdb.set_trace()
        # self.model.save('model.file')
        # write_json('history.json', self.infos_history)
        # write_json('lrm_history.json', self.lrm_history)
        pass

    def load(self):
        # self.model.load_weights(self.sess, weights)
        # self.model.load()
        pass

    def sweep_history_train(self):

        pass

    def _get_best_config_index(self, cur_infos, metric, min_or_max='min'):
        scores = np.array([info[metric] for info in cur_infos])
        if min_or_max == 'min':
            best_config_index = np.argmin(scores)
        else:
            best_config_index = np.argmax(scores)
        return best_config_index

    ##### hacks ############


class MyModel(object):
    def __init__(self, config=None, seed = 0):
        self.seed = seed
        self.config = config

    def setup(self):

        ##### layer sturucture ######
        self.x_pl = tf.placeholder(tf.float32, shape=(None, 28*28))
        self.y_pl = tf.placeholder(tf.int32, shape=(None, ))


        self.weights_nat = weight_nat()

        self.outputs_adv , self.weights_adv = layers(x_input = self.x_pl, config = self.config)

        self.t = self.config['eps']
        self.adv_image = self.x_pl + self.t * self.outputs_adv

        self.pre_softmax_adv = model_nat(self.adv_image, self.weights_nat)

        ##### losses ######

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.y_pl, logits = self.pre_softmax_adv))
        self.obj_func = - self.loss # maximize loss

        #### optimizer ####

        self.learning_rate_pl = tf.placeholder(tf.float32, shape=())

        self.momentum = self.config['momentum']

        # print('!!!Using Gradient descent optimizer!')
        # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_pl)

        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate_pl,
            self.momentum,
            use_nesterov=True
            )
        print('!!!use use_nesterov')

        # optimizer = tf.train.AdamOptimizer()
        # print('!!!use AdamOptimizer')
        var_list = list(self.weights_adv.values())
        self.train_op = optimizer.minimize(self.obj_func, var_list = var_list) #only train adversarial network

        ##### metrics (used in eval) #####
        self.y_pred = tf.cast(tf.argmax(self.pre_softmax_adv, 1), tf.int32)
        self.correct_prediction = tf.equal(self.y_pred, self.y_pl) # used for accuracy
        self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def train(self, sess, train_dataset, learning_rate, num_epoch=1, verbose = 0):

        ###### my training loop ########

        train_dataset.intialize_batch()

        for batch_i in range(train_dataset.num_batch):
            x_batch, y_batch = train_dataset.next_batch()

            feed_dict0 = {
                self.x_pl:x_batch,
                self.y_pl:y_batch,
                self.learning_rate_pl: learning_rate}

            sess.run([self.train_op], feed_dict = feed_dict0)

            current_batch_loss = sess.run(self.loss, feed_dict = feed_dict0)
            if verbose: print('     bstep %d current_batch_loss'%batch_i, current_batch_loss)

            # if batch_i == 5:
            #     print('  stopping(debug)')
            #     break

        stats = self.eval(sess, train_dataset)

        train_stats = {
            'train_loss':stats['val_loss'],
            'train_acc':stats['val_acc']
            }

        return train_stats

    def eval(self, sess, test_dataset):
        x, y = test_dataset.entire()

        feed_dict0 = {
                self.x_pl:x,
                self.y_pl:y,
            }

        test_loss = sess.run(self.loss, feed_dict = feed_dict0)
        # correct_prediction1 = sess.run(self.correct_prediction, feed_dict = feed_dict0)

        # val_stats = {
        #     'val_loss':float(test_loss),
        #     'val_acc': sum(correct_prediction1) / test_dataset.num_data
        #     }

        accuracy = sess.run(self.accuracy, feed_dict = feed_dict0)

        val_stats = {
            'val_loss':float(test_loss),
            'val_acc': float(accuracy)
            }



        return val_stats


    def get_weights(self, sess):
        weights = {}
        for key in self.weights_adv:
            weights[key] = sess.run(self.weights_adv[key])

        return weights

    def load_weights(self,sess, weights):
        for key in self.weights_adv:
            sess.run(
                self.weights_adv[key],
                feed_dict = {
                    self.weights_adv[key]: weights[key]
                    }
                )

    def save(self, filename):

        pass

    def load(self):
        pass


    ############# hacks ################
    def hack1(self):
        pass




class Dataset(object):
    def __init__(self, config=None ,seed = 0):
        self.config = config
        self.seed = seed

    def load_data(self):
        '''override this func'''
        self.x = [[1,2],[3,4]]
        self.y = [[5],[6]]

    def setup(self, batch_size, shuffle = 0):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.load_data()

        assert len(self.x) == len(self.y)
        self.num_data = len(self.x)

        self.cur_pos = -1

        self.num_batch = self.num_data // self.batch_size # drops data if left

    def intialize_batch(self):
        if self.shuffle:
            self.read_order = np.random.permutation(self.num_data)
        else:
            self.read_order = list(range(self.num_data))

        self.cur_pos = 0


    def next_batch(self):

        if self.cur_pos == -1 or self.cur_pos + self.batch_size > self.num_data:
            self.intialize_batch()

        indices = self.read_order[ self.cur_pos : self.cur_pos + self.batch_size ]

        if len(self.x.shape)==1:
            out_x = self.x[indices]
        else:
            out_x = self.x[indices,:]

        if len(self.y.shape)==1:
            out_y = self.y[indices]
        else:
            out_y = self.y[indices,:]

        self.cur_pos += self.batch_size

        return out_x, out_y

    def entire(self):
        return self.x, self.y

class MnistTrain(Dataset):
    def load_data(self):
        data_sets = input_data.read_data_sets('MNIST_data', one_hot=False)
        self.x = data_sets.train.images
        self.y = data_sets.train.labels

class MnistTest(Dataset):
    def load_data(self):
        data_sets = input_data.read_data_sets('MNIST_data', one_hot=False)
        self.x = data_sets.test.images
        self.y = data_sets.test.labels


def get_single_datapoint_from_batch(images_feed, labels_feed, idx):
    xx = images_feed[idx,:].reshape((-1,784))
    yy = labels_feed[idx].reshape((1,))

    return xx, yy

def placeholder_inputs_direction(batch_size):
    x_pl = tf.placeholder(tf.float32, shape=(batch_size, 28*28))
    y_pl = tf.placeholder(tf.float32, shape=(batch_size, 28*28))
    return x_pl, y_pl



def L2_regularizer(trainable_weights):
    regularizer = tf.constant(0.0)
    for key in trainable_weights:
        if key[0] == "W": # only update weights not bias
            regularizer += tf.nn.l2_loss(trainable_weights[key])

    # imporve: per layer regularizer?
    # https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/

    return regularizer


if __name__ == "__main__":
    import sys
    assert len(sys.argv)>=3
    RANK = int(sys.argv[1])
    eps = float(sys.argv[2])

    generate_dirs(['./plots','./ckpt%d'%RANK])
    start_time = time.time()
    main(RANK, eps)
    duration = (time.time() - start_time)

    print("---Program Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))



