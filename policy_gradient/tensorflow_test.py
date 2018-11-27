#coding:utf-8

import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import math
import numpy as np
import random

ep_observation = []
ep_action = []
ep_reward = []

def store_transition(state, action, reward):
    ep_observation.append(state)
    ep_action.append(action)
    ep_reward.append(reward)

env = gym.make('CartPole-v0')

# 1. Specify the network architecture
n_inputs = env.observation_space.shape[0]
n_hidden = 4
n_outputs = 2
initializer = tf.variance_scaling_initializer()

# 2. Build the neural network
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
                         kernel_initializer = initializer)
action_prob = tf.layers.dense(hidden, n_outputs, activation=tf.nn.softmax, kernel_initializer = initializer)

# 3. Select a random action based on the estimated probabilities
choose_action = tf.multinomial(tf.log(action_prob), num_samples=1)

# 4. Set loss function
# action_inputs = tf.placeholer(tf.int32, shape=[None])
# reward_inputs = tf.placeholder(tf.float32, shape=[None])
# softmax_cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits()
# loss = tf.reduce_mean(softmax_cross_entropy_loss * reward_inputs)
# optimizer = tf.Adamoptimizer(loss)

init = tf.global_variables_initializer()

max_episodes = 200
max_steps = 200
rewards = []

with tf.Session() as sess:
    sess.run(init)

    for episode in range(max_episodes):
        observation = env.reset()
        episord_reward = 0

        for step in range(max_steps):
            env.render()

            action = sess.run(choose_action,
                    feed_dict={X:observation.reshape(1, n_inputs)})

            observation, reward, done, _ = env.step(action[0][0])
            episord_reward += reward

            store_transition(observation, action, reward)

            if done:
                rewards.append(episord_reward)
                break

env.close()

plt.plot(rewards, "b-")
plt.show()
