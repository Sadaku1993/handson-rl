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

class Critic(object):
    def __init__(self):
        self.value = np.random.uniform(
                low=-1, high=1, size=num_dizitized**4)
        self.gamma = 0.5
        self.alpha = 0.1

    def update(self, state, reward, next_state):
        next_value = self.value[next_state]
        self.value[state] = (1-self.alpha) * self.value[state] + self.alpha * (reward + self.gamma * next_value)

    def td_error(self, state, reward, next_state):
        next_value = self.value[next_state]
        return reward + self.gamma * next_value - self.value[state]

    def value(self, state):
        return self.value[state]

class Actor(object):
    def __init__(self):
        self.action = np.random.uniform(
                low=0, high=1, size=(num_dizitized**4, env.action_space.n))
        self.param = 0.5
        
    def update(self, state, action):
        self.action[state, action] = (self.param + self.action[state, action]) / float(self.param + np.sum(self.action[state]))

    def get_action(self, state, episode):
        epsilon = 0.5 * (1/(episode+1))
        if epsilon <= np.random.uniform(0, 1):
            next_action = np.argmax(self.action[state])
        else:
            next_action = np.random.choice([0, 1])
        return next_action

goal_average_reward = 195
num_consecutice_iterations = 100

critic = Critic()
actor = Actor()

total_reward_vec = np.zeros(num_consecutice_iterations)

for episode in range(num_episodes):
    observation = env.reset()
    state = digitize_state(observation)
    episode_reward = 0

    for t in range(max_step):
        action = actor.get_action(state, episode)
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

        td_error = critic.td_error(state, reward, next_state)

        if td_error>0:
            actor.update(state, action)

        critic.update(state, reward, next_state)

        state = next_state

        if done:
            print('%d Episode finished after %f time steps / mean %f' %
                  (episode, t + 1, total_reward_vec.mean()))
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  #報酬を記録
            break
 
    if (total_reward_vec.mean() >= goal_average_reward):  # 直近の100エピソードが規定報酬以上であれば成功
        print('Episode %d train agent successfuly!' % episode)
        break
