import time
import argparse

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

from util import *
from model2 import *
from load_model import *

def parse_args_get_config():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",type=str,default='train')
    parser.add_argument("--eps",type=float,default=0.3)
    parser.add_argument("--load-weight",type=str,default='nat')
    parser.add_argument("--fc-id",type=int,default=0)
    parser.add_argument("--norm",type=str,default='linf')
    parser.add_argument("--skip-finished",action='store_true')
    args = parser.parse_args()

    config = {
            ### experiment related ##
            "max_epoch": 200,
            "lr": 0.1, # base lr (default 0.1)
            "momentum": 0.9,
            "batch_size": 100,
            "eps" : 0.3,
            "norm": "linf", # l2, linf
            "lmbd": 100, # lambda for barrier regularizer
            "load_weight": 'nat', # nat, adv, sec
            "model":"cnn", #"fcn", "cnn"
            "fc_id": 0, # id for model
            "desc":'', # optional descriptor

            ### not in desciptor ##
            "lr_multipliers": [1.0, 0.9 ,0.8, 0.7, 0.6],
            # optional
            "lr_multipliers_init": [
                1.0, 0.9 ,0.8, 0.7, 0.6, \
                0.5, 0.4,0.3,0.2,0.1] ,

            ## runtime related ##
            "mode": 'train',
            "skip_finished": False

        }

    config.update(vars(args))

    # override (debug)
    config["max_epoch"] = 5
    config["lr_multipliers_init"] = [1.0, 0.9, 0.8]

    ##### processing ########

    lists_fc_hidden_layers = [
            [ 1024*2],
            [ 1024*2, 1024*5],
            [ 1024*2, 1024*5, 1024*2],
            [ 1024*2, 1024*5, 1024*2, 1024*2]
    ]
    config["fc_hidden_layers"] = lists_fc_hidden_layers[config['fc_id']]

    return config


def main():

    config = parse_args_get_config()

    print('Running EXP for:')
    print(config)

    AA = Trainer(config)

    with Timer(name = 'initialize') as t:
        AA.setup()

    with Timer(name = 'run_train') as t:
        AA.run_train_pbt()
        AA.save()

    #     # not right now
    #     # AA.run_train_single_batch()
    #     # AA.tmp11()
    #     # AA.check_initial_loss_per_train_image()

    with Timer(name = 'avg_loss') as t:
        # img_indices = [0,1,2]
        # img_indices = range(2000, 2100)
        # AA.restore_and_get_avg_loss(dataset='train' ,img_indices=img_indices)

        img_indices = range(0, 10000)
        AA.restore_and_get_avg_loss(dataset='test' ,img_indices=img_indices)

        img_indices = range(0, 55000)
        AA.restore_and_get_avg_loss(dataset='train' ,img_indices=img_indices)

    # with Timer(name = 'check_norm') as t:
    #     AA.restore_and_check_norm()

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

        fetch_checkpoint_hack(self.sess, self.model.weights_nat, model=self.config['load_weight']) # load naturual model

    def run_train_pbt(self, verbose = 1):

        if self.config['skip_finished'] and self.check_result_file_exist():
            print("!!!! skipping experiment since result file exist")
            return


        SMALL_LR_THRESHOLD = 0.0001
        checkpoint_dir = './ckpt/' + self.fname_prefix()

        c = self.config

        infos_history = []
        current_lr = c['lr']
        self.saver.save(self.sess, checkpoint_dir+"model_best.ckpt")

        train_stats = self.model.eval(self.sess, self.train_dataset, prefix = 'train_')

        val_stats = self.model.eval(self.sess,self.test_dataset, prefix = 'val_')

        info = {"epoch":-1, "lr":-1}
        info.update(train_stats)
        info.update(val_stats)

        print(f"{self.format_stats(info)} ")

        for epoch in range(c['max_epoch']):

            t0 = time.time()

            weight_sets = []
            cur_infos = []

            if epoch == 0 and 'lr_multipliers_init' in self.config:
                lr_multipliers = self.config['lr_multipliers_init']
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

                self.saver.restore(self.sess, checkpoint_dir+"model_best.ckpt")

                train_stats = self.model.train(self.sess, self.train_dataset, learning_rate = lr, verbose = 0)
                val_stats = self.model.eval(self.sess,self.test_dataset, prefix = 'val_')

                info = {"epoch":epoch, "lr":lr, "lrm":lr_multipliers[lr_i],"best":False}
                info.update(train_stats)
                info.update(val_stats)

                cur_infos.append(info)

                self.saver.save(self.sess, checkpoint_dir+"model_%d.ckpt"%(lr_i))

                t01 = time.time()
                if verbose: print(f"  {self.format_stats(info)} took {(t01 - t00):.3f}sec")

            b_i = self._get_best_config_index(cur_infos, metric='val_loss', min_or_max='max')
            cur_infos[b_i]['best'] = True
            current_lr = cur_infos[b_i]['lr']

            infos_history.append(cur_infos)
            info = cur_infos[b_i]
            self.saver.restore(self.sess, checkpoint_dir+"model_%d.ckpt"%(b_i))
            time.sleep(2)
            self.saver.save(self.sess, checkpoint_dir+"model_best.ckpt")

            t1 = time.time()

            print(f"{self.format_stats(info)} took {(t1 - t0):.3f}sec")

            if info['lr'] < SMALL_LR_THRESHOLD:
                print('stop training since lr become too small')
                break

            # import ipdb; ipdb.set_trace()

        results_path = './results/' + self.fname_prefix()
        self.saver.save(self.sess, results_path+"model_best.ckpt")

        self.infos_history = infos_history
        import pickle
        pickle.dump( self.infos_history , open( results_path+"infos_history.p", "wb" ))

        self.remove_checkpoint_files()

    def check_result_file_exist(self):
        import os
        results_path = './results/' + self.fname_prefix()
        fname = results_path+"infos_history.p"
        print("checking ", fname)

        return os.path.isfile( results_path+"infos_history.p")

    def remove_checkpoint_files(self):
        checkpoint_dir = './ckpt/' + self.fname_prefix()

        if 'lr_multipliers_init' in self.config:
            num_run_per_epoch = len(self.config['lr_multipliers_init'])
        else:
            num_run_per_epoch = len(self.config['lr_multipliers'])

        def rm_file(fname):
            import os
            if os.path.isfile(fname):
                os.remove(fname)

        for i in range(num_run_per_epoch):
            ckpt_name = checkpoint_dir+"model_%d.ckpt"%(i)
            rm_file(ckpt_name + ".index")
            rm_file(ckpt_name + ".meta")
            rm_file(ckpt_name + ".data-00000-of-00001")

        ckpt_name = checkpoint_dir+"model_best.ckpt"
        rm_file(ckpt_name + ".index")
        rm_file(ckpt_name + ".meta")
        rm_file(ckpt_name + ".data-00000-of-00001")




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
        # for reproducing
        pass

    def _get_best_config_index(self, cur_infos, metric, min_or_max='min'):
        scores = np.array([info[metric] for info in cur_infos])
        if min_or_max == 'min':
            best_config_index = np.argmin(scores)
        else:
            best_config_index = np.argmax(scores)
        return best_config_index

    def fname_prefix(self):
        str1 = "norm{norm}_w{load_weight}_eps{eps:.2f}_lmbd{lmbd:.3f}_model{model}_fc{fc_id}_lr{lr:.3f}_me{max_epoch}_{desc}_".format(**self.config)
        return str1

    def format_stats(self, info):
        str0 = "epoch {epoch} trl {train_loss:.5f} trlc {train_loss_clipped:.5f}, trrl {train_regularizer:.5f} vl {val_loss:.5f} vlc {val_loss_clipped:.5f} vlrl {train_regularizer:.5f} lr {lr:.4f}".format(**info)
        return str0

    ##### hacks ############

    def restore_and_get_avg_loss(self, dataset ,img_indices):
        results_path = './results/' + self.fname_prefix()

        self.saver.restore(self.sess, results_path+"model_best.ckpt")

        if dataset=='train':
            val_stats = self.model.eval_by_index(self.sess, self.train_dataset, img_indices)
        elif dataset=='test':
            val_stats = self.model.eval_by_index(self.sess, self.test_dataset, img_indices)
        else:
            assert 0

        print(dataset+" avg_loss:{val_loss}, avg_acc:{val_acc}".format(**val_stats))

    def restore_and_check_norm(self):
        results_path = './results/' + self.fname_prefix()
        self.saver.restore(self.sess, results_path+"model_best.ckpt")

        indices = range(0, 55000)
        x, y = self.train_dataset.load_index(indices)

        # import ipdb; ipdb.set_trace()

        feed_dict0 = {
                self.model.x_pl:x,
                self.model.y_pl:y,
            }

        print(self.config['norm'],'norm', self.sess.run(self.model.adv_norm, feed_dict = feed_dict0))



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
        self.regularizer = self.config['lmbd']* tf.reduce_mean(barrier_regularizer(self.adv_image))

        self.obj_func = - self.loss  +  self.regularizer  # maximize loss, minimize regularizer term
        # self.obj_func = - (self.loss) # maximize loss

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

        if self.config['norm'] == 'l2':
            self.adv_norm = tf.norm( self.adv_image - self.x_pl)
        elif self.config['norm'] == 'linf':
            self.adv_norm = tf.norm( self.adv_image - self.x_pl, ord = np.inf)
        else:
            assert 0

        # clipped loss
        self.adv_image_clipped = tf.clip_by_value(self.adv_image, 0, 1) # added
        self.pre_softmax_adv_clipped = model_nat(self.adv_image_clipped, self.weights_nat)
        self.loss_clipped = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.y_pl, logits = self.pre_softmax_adv_clipped))

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

        train_stats = self.eval(sess, train_dataset, prefix='train_')


        return train_stats

    def eval(self, sess, test_dataset, prefix = None):
        x, y = test_dataset.entire()

        feed_dict0 = {
                self.x_pl:x,
                self.y_pl:y,
            }

        test_loss = sess.run(self.obj_func, feed_dict = feed_dict0)
        # correct_prediction1 = sess.run(self.correct_prediction, feed_dict = feed_dict0)

        # val_stats = {
        #     'val_loss':float(test_loss),
        #     'val_acc': sum(correct_prediction1) / test_dataset.num_data
        #     }

        accuracy = sess.run(self.accuracy, feed_dict = feed_dict0)

        norm = sess.run(self.adv_norm, feed_dict = feed_dict0)

        loss_clipped = sess.run(self.loss_clipped, feed_dict = feed_dict0)

        regularizer = sess.run(self.regularizer, feed_dict = feed_dict0)


        stats = {
            'loss':float(test_loss),
            'acc': float(accuracy),
            'norm': float(norm),
            'loss_clipped': float(loss_clipped),
            'regularizer': float(regularizer),
            }

        if prefix is not None:
            stats = dict_keys_add_prefix(stats, prefix)

        return stats

    def eval_by_index(self, sess, test_dataset, indices):
        x, y = test_dataset.load_index(indices)

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

    def load_index(self, indices):
        if len(self.x.shape)==1:
            out_x = self.x[indices]
        else:
            out_x = self.x[indices,:]

        if len(self.y.shape)==1:
            out_y = self.y[indices]
        else:
            out_y = self.y[indices,:]

        return out_x, out_y

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

def barrier_regularizer(image):
    c = 1000 # scale x

    a = 0.0 # lower limit
    b = 1.0 # upper limit

    y = ( tf.exp(c * (image-b) ) + tf.exp(c * (-(image-a)) ) )

    return y

def dict_keys_add_prefix(dict0,prefix):
    keys = list(dict0.keys())
    for key in keys:
        new_key = prefix+key
        dict0[new_key] = dict0[key]
        del dict0[key]

    return dict0



if __name__ == "__main__":

    generate_dirs(['./plots','./ckpt','./results'])
    start_time = time.time()
    main()
    duration = (time.time() - start_time)

    print("---Program Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))



