#coding:utf-8

import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import math

env = gym.make('CartPole-v0')

# 1. Specify the network architecture
n_inputs = env.observation_space.shape[0]
n_hidden = 4
n_outputs = 1
initializer = tf.variance_scaling_initializer()

# 2. Build the neural network
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
                         kernel_initializer = initializer)
outputs = tf.layers.dense(hidden, n_outputs, activation=tf.nn.elu,
                          kernel_initializer = initializer)

# 3. Select a random action based on the estimated probabilities
p_left_and_right = tf.concat(axis=1, values=[outputs, 1-outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

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
            # env.render()
            action_val = action.eval(feed_dict={X:observation.reshape(1, n_inputs)})
            observation, reward, done, _ = env.step(action_val[0][0])
            episord_reward += reward

            if done:
                rewards.append(episord_reward)
                break

env.close()

plt.plot(rewards, "b-")
plt.show()
