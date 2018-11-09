#coding:utf-8

import gym
import numpy
import math
import tensorflow as tf


env = gym.Make('CartPole-v0')


class PolyGradient:
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay = 0.95):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_intializer())

    def build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        layer = tf.layers.dense(
                inputs = self.tf_obs,
                units=10, 
                activation = tf.nn.tanh,
                name='fc1'
        )
        all_act = tf.layers.dense(
                inputs = layer,
                units = self.n_actions,
                activation = None,
                name='fc2'
        )

        # 行動選択確率(softmax)
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.refuce_neam(neg_log_prob * self.tf_vt)
