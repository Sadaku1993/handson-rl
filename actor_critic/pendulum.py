#coding:utf-8

"""Pendulum-v0




"""

import gym
import numpy as np
import tensorflow as tf

np.random.seed(2)
tf.set_random_seed(2)

class Actor(object):
    def __init__(
            self,
            sess,
            n_features,
            action_bound,
            lr = 0.0001
    ):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.float32, None, name="action")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")

        l1 = tf.layers.dense(
                inputs = self.s,
                units = 20,
                activation = tf.nn.relu,
                kernel_initializer = tf.random_normal_initializer(0., 1.),
                bias_initializer = tf.constant_initializer(0.1),
                name='l1'
        )

        mu = tf.layers.dense(
                inputs = l1,
                units = 1,
                activation = tf.nn.tanh,
                kernel_initializer = tf.random_normal_initializer(0., 1.),
                bias_initializer = tf.constant_initializer(0.1),
                name='mu'
        )

        sigma = tf.layers.dense(
                inputs = l1,
                units = 1,
                activation = tf.nn.softplus,
                kernel_initializer = tf.random_normal_initializer(0., 1.),
                bias_initializer = tf.constant_initializer(0.1),
                name='sigma'
        )

        global_step = tf.Variable(0, trainable=False)

        self.mu = tf.squeeze(mu**2)
        self.sigma = tf.squeeze(sigma+0.1)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        self.action = tf.clip_by_value(
                self.normal_dist.sample(1),
                action_bound[0],
                action_bound[1]
        )

        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)
            self.exp_v = log_prob * self.td_error
            self.exp_v += 0.01*self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)


    def learn(self, s, a, td):
        s = s[np.nexaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: tf}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s: s})

    def show(self, s, a):
        s = s[np.newaxis, :]
        
        mu, sigma = self.sess.run([self.mu, self.sigma], {self.s: s})
        print("mu:" , mu, " sigma:", sigma)

        sample = self.normal_dist.sample(1)
        log_prob = self.normal_dist.log_prob(self.a)
        entropy = 0.01*self.normal_dist.entropy()
        s, l, e = self.sess.run([sample, log_prob, entropy], {self.s: s, self.a: a})
        print("sample:", s, " log_prob:", l, " entropy:", entropy)

def main():
    env = gym.make('Pendulum-v0')

    n_features = env.observation_space.shape[0]
    n_actions = env.action_space
    action_bound = env.action_space.high

    print("n_features:", n_features)
    print("n_actions:", n_actions)
    print("action_bound:", action_bound[0])

    sess = tf.Session()

    actor = Actor(sess, n_features, lr=0.001, action_bound=[-action_bound, action_bound])

    sess.run(tf.global_variables_initializer())

    max_episode = 2000
    max_step = 200

    for episode in range(max_episode):
        state = env.reset()
        for step in range(max_step):
            action = actor.choose_action(state)
            actor.show(state, action)
            next_state, reward, done, _ = env.step(action)
            state = next_state


if __name__ == '__main__':
    main()
