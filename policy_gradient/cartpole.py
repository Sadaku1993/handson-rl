#coding:utf-8

import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow as tf

from policy_gradient import PolicyGradient

RENDER = False

env = gym.make('CartPole-v0')
# env.seed(1)
# np.random.seed(1)

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

max_episode = 3000
max_step = 200

for episode in range(max_episode):

    observation = env.reset()

    for step in range(max_step):
        if RENDER: env.render()


