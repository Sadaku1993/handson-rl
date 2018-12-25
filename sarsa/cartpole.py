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
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CartPole-v0')

num_episodes = 2000
max_step = 200
num_dizitized = 6

RENDER = False

# 状態を離散化
def bins(state_min, state_max, num):
    return np.linspace(state_min, state_max, num+1)[1:-1]

def digitize_state(observation):
    cart_position, cart_velocity, pole_angle, pole_velocity = observation
    digitized = [
        np.digitize(cart_position, bins=bins(-2.4, 2.4, num_dizitized)),
        np.digitize(cart_velocity, bins=bins(-3.0, 3.0, num_dizitized)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_dizitized)),
        np.digitize(pole_velocity, bins=bins(-2.0, 2.0, num_dizitized))
    ]
    return sum([x*(num_dizitized**i) for i, x in enumerate(digitized)])


def get_action(q_table, next_state, episode):
    epsilon = 0.5 * (1/(episode+1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])
    return next_action

def update_Qtable(q_table, state, action, reward, next_state, next_action):
    gamma = 0.99
    alpha = 0.5
    q_table[state, action] = (1-alpha) * q_table[state, action] + \
            alpha * (reward + gamma + q_table[next_state, next_action])
    return q_table

goal_average_reward = 195
num_consecutice_iteratinos = 100
q_table = np.random.uniform(
        low=-1, high=1, size=(num_dizitized**4, env.action_space.n))
total_reward_vec = np.zeros(num_consecutice_iteratinos)

for episode in range(num_episodes):
    observation = env.reset()
    state = digitize_state(observation)
    action = np.argmax(q_table[state])
    episode_reward = 0

    for t in range(max_step):
        if RENDER:
            env.render()
        observation, reward, done, info = env.step(action)

        if done:
            if t<195:
                reward -= 200
            else:
                reward = 1
        else:
            reward = 1

        episode_reward += reward
        
        next_state = digitize_state(observation)
        next_action = get_action(q_table, next_state, episode)
        
        q_table = update_Qtable(q_table, state, action, reward, next_state, next_action)

        state = next_state
        action = next_action

        if done:
            print('%d Episode finished after %f time steps / mean %f' %
                  (episode, t + 1, total_reward_vec.mean()))
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  #報酬を記録
            break
 
    if (total_reward_vec.mean() >= goal_average_reward):  # 直近の100エピソードが規定報酬以上であれば成功
        print('Episode %d train agent successfuly!' % episode)
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
        state = next_state

        if done:
            break
