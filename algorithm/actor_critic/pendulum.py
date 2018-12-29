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
            lr = 0.001
    ):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [None, n_features], "state")
        self.a = tf.placeholder(tf.float32, None, name="action")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")

        l1 = tf.layers.dense(
                inputs = self.s,
                units = 30,
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

        self.mu = tf.squeeze(mu*2)
        self.sigma = tf.squeeze(sigma+0.1)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)
        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])

        # self.mu = tf.squeeze(mu*2)
        # self.sigma = tf.squeeze(sigma+0.1)
        # self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        # self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])

        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)
            self.exp_v = log_prob * self.td_error
            self.exp_v += 0.01*self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s: s})

class Critic(object):
    def __init__(
            self,
            sess,
            n_features,
            lr = 0.01,
            gamma = 0.9
    ):
        self.sess = sess

        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [1, n_features], "state")
            self.v_ = tf.placeholder(tf.float32, [1, 1], name="v_next")
            self.r = tf.placeholder(tf.float32, name="r")

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                    inputs = self.s,
                    units = 30,
                    activation = tf.nn.relu,
                    kernel_initializer = tf.random_normal_initializer(0., 1.),
                    bias_initializer = tf.constant_initializer(0.1),
                    name='l1'
            )

            self.v = tf.layers.dense(
                    inputs = l1,
                    units = 1,
                    activation = None,
                    kernel_initializer = tf.random_normal_initializer(0., 1.),
                    bias_initializer = tf.constant_initializer(0.1),
                    name='v'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + gamma * self.v_ - self.v)
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, feed_dict={self.s: s_})
        
        td_error, _ = self.sess.run(
                [self.td_error, self.train_op],
                feed_dict = {self.s: s, self.v_: v_, self.r: r}
        )

        return td_error

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
    critic = Critic(sess, n_features, lr=0.01, gamma=0.9)

    sess.run(tf.global_variables_initializer())

    max_episode = 300
    max_step = 200
    RENDER = False

    goal_average_reward = -100
    num_consecutice_iterations = 20
    total_reward_vec = np.zeros(num_consecutice_iterations)

    for episode in range(max_episode):
        state = env.reset()
        step = 0
        episode_reward = []
        while True:
            if RENDER: env.render()
            action = actor.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            reward /= 10

            td_error = critic.learn(state, reward, next_state)
            actor.learn(state, action, td_error)

            state = next_state

            episode_reward.append(reward)
            step += 1

            if step > max_step:
                episode_reward_sum = sum(episode_reward)
                print('%d Episode finished reward %d / mean %f' %
                      (episode, episode_reward_sum, total_reward_vec.mean()))
                total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward_sum))
                break
            
        if (total_reward_vec.mean() >= goal_average_reward and 
                episode >= num_consecutice_iterations):
            print('Episode %d train agent successfuly!' % episode)
            break

if __name__ == '__main__':
    main()
