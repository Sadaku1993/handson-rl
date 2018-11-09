#coding:utf-8

import gym
import numpy as np
import matplotlib.pyplot as ply

"""
MountainCar-v0
    
    Observation:
        Type: Box(2)
        Num     Observation     Min     Max
        0       Position        -1.2    0.6
        1       Speed           -0.07   0.07

    Actions:
        Type: Discrete(3)
        Num     Action
        0       Push car to the left
        1       Not Push
        2       Push car to the right
"""

env = gym.make('MountainCar-v0')

num_episodes = 3000
max_step = 200
num_digitized = 50

def bins(state_min, state_max, num):
    return np.linspace(state_min, state_max, num+1)[1:-1]

def digitize_state(observation):
    car_position, car_speed = observation
    digitized = [
            np.digitize(car_position, bins=bins(-1.2, 0.6, num_digitized)),
            np.digitize(car_speed, bins=bins(-0.07, 0.07, num_digitized))
    ]
    return sum([x*(num_digitized**i) for i, x, in enumerate(digitized)])

def get_action(state, episode):
    epsilon = 0.5 * (1/(episode+1))
    if epsilon<0.1:
        epsilon = 0.1
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[state])
    else:
        next_action = np.random.choice([0, 2])
    return next_action

def update_Qtable(q_table, state, action, reward, next_action):
    gamma = 0.99
    alpha = 0.5
    next_Max_Q = max(q_table[next_state][0], q_table[next_state][1], q_table[next_state][2])
    q_table[state, action] = (1-alpha) * q_table[state, action] + alpha * (reward + gamma * next_Max_Q)
    return q_table

num_consecutice_iterations = 100
q_table = np.random.uniform(
        low=-1, high=1, size=(num_digitized**2, env.action_space.n))
total_reward_vec = np.zeros(num_consecutice_iterations)

for episode in range(num_episodes):
    observation = env.reset()
    state = digitize_state(observation)
    action = np.argmax(q_table[state])
    episode_reward = 0

    for t in range(max_step):
        # env.render()
        observation, reward, done, info = env.step(action)

        if observation[0] > 0.1:
            reward = 2**observation[0] + 0.5

        episode_reward += reward

        next_state = digitize_state(observation)
        q_table = update_Qtable(q_table, state, action, reward, next_state)
        action = get_action(next_state, episode)

        state = next_state

        if done:
            print('%d Episode finished after %f time steps state %f / mean %f' %
                  (episode, t + 1, observation[0], total_reward_vec.mean()))
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  #報酬を記録
            break

for _ in range(10):
    observation = env.reset()
    state = digitize_state(observation)
    action = np.argmax(q_table[state])
    for i in range(max_step):
        env.render()
        observation, reward, done, info = env.step(action)
        next_state = digitize_state(observation)
        action = np.argmax(q_table[next_state])

        if done:
            break

# print(env.action_space)
# print(env.observation_space)
# 
# observation = env.reset()
# 
# for _ in range(10000):
#     env.render()
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
# 
#     print(action, observation , reward)
#     
#     if done:
#         break
