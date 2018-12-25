#coding:utf-8

"""
CartPole-v0

    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(2)
tf.set_random_seed(2)

class Actor(object):
    def __init__(
            self,
            sess,
            n_features,
            n_actions,
            learning_rate
    ):
        self.sess = sess

        self.s = tf.placeholder(
                tf.float32, 
                shape = [1, n_features], 
                name = "state"
        )
        self.a = tf.placeholder(
                tf.int32, 
                shape = None, 
                name = "action"
        )
        self.td_error = tf.placeholder(
                tf.float32, 
                shape = None, 
                name = "td_error"
        )

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                    inputs = self.s,
                    units = 20,
                    activation = tf.nn.relu,
                    kernel_initializer = tf.random_normal_initializer(0., 1.),
                    bias_initializer = tf.constant_initializer(0.1),
                    name = 'l1'
            )

            self.action_prob = tf.layers.dense(
                    inputs = l1,
                    units = n_actions,
                    activation = tf.nn.softmax,
                    kernel_initializer = tf.random_normal_initializer(0., 1.),
                    bias_initializer = tf.constant_initializer(0.1),
                    name = "acts_prob"
            )

            self.action = tf.multinomial(
                    tf.log(self.action_prob), 
                    num_samples=1
            )

        with tf.variable_scope('loss'):
            log_prob = tf.reduce_sum(tf.log(self.action_prob) * tf.one_hot(self.a, n_actions), axis=1)
            self.exp_v = log_prob * self.td_error

            # log_prob = -tf.reduce_sum(tf.log(self.action_prob) * tf.one_hot(self.a, n_actions), axis=1)
            # loss = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(-self.exp_v)

    def choose_action(self, observation):
        action = self.sess.run(
                self.action,
                feed_dict={self.s:observation}
        )
        return action

    def learn(self, state, action, td_error):
        self.sess.run(
                self.train_op, 
                feed_dict={self.s: state, self.a: action, self.td_error: td_error}
        )

class Critic(object):
    def __init__(
            self,
            sess,
            n_features,
            learning_rate,
            gamma
    ):
        self.sess = sess

        self.s = tf.placeholder(
                tf.float32,
                shape = [1, n_features],
                name = "state"
        )

        self.v_ = tf.placeholder(
                tf.float32,
                shape = [1, 1],
                name = "v_next"
        )

        self.r = tf.placeholder(
                tf.float32,
                shape = None,
                name = "reward"
        )

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                    inputs = self.s,
                    units = 20,
                    activation = tf.nn.relu,
                    kernel_initializer = tf.random_normal_initializer(0., 1.),
                    bias_initializer = tf.constant_initializer(0.1),
                    name='l1'
            )

            self.v = tf.layers.dense(
                    inputs = l1,
                    units = 1,
                    activation = None,
                    kernel_initializer = tf.random_normal_initializer(0. ,1.),
                    bias_initializer = tf.constant_initializer(0.1),
                    name="V"
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error) 
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def learn(self, s, r, s_):
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error
        
def main():
    env = gym.make('CartPole-v0')

    print("action space:", env.action_space)
    print("observation spaace:", env.observation_space)

    RENDER = False
    
    n_features = env.observation_space.shape[0]
    n_hidden   = 4
    n_actions  = 2
    
    max_episodes = 3000
    max_steps = 200

    xlabel = []
    ylabel = []

    sess = tf.Session()

    actor = Actor(
            sess,
            n_features = n_features,
            n_actions  = n_actions,
            learning_rate = 0.001
    )

    critic = Critic(
            sess,
            n_features = n_features,
            learning_rate = 0.01,
            gamma = 0.99
    )

    sess.run(tf.global_variables_initializer())

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            if RENDER: env.render()

            action = actor.choose_action(state.reshape(1, n_features))
            next_state, reward, done, _ = env.step(action[0][0])


            if done:
                reward = -20

            episode_reward += reward

            td_error = critic.learn(
                    state.reshape(1, n_features), 
                    reward, 
                    next_state.reshape(1, n_features)
            )

            actor.learn(
                    state.reshape(1, n_features),
                    action[0][0],
                    reward
            )

            if done:
                print("episode:%3d reward:%3d" % (episode, episode_reward))
                xlabel.append(episode)
                ylabel.append(episode_reward)
                break

            next_state = state

    plt.plot(xlabel, ylabel, 'b-')
    plt.xlabel('episode steps')
    plt.ylabel('episode rewards')
    plt.show()

if __name__ == '__main__':
    main()
