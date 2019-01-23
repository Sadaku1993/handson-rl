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

# 状態を離散化
def bins(state_min, state_max, num):
    return np.linspace(state_min, state_max, num+1)[1:-1]

def digitize_state(observation, num_digitized):
    cart_position, cart_velocity, pole_angle, pole_velocity = observation
    digitized = [
        np.digitize(cart_position, bins=bins(-2.4, 2.4, num_digitized)),
        np.digitize(cart_velocity, bins=bins(-3.0, 3.0, num_digitized)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_digitized)),
        np.digitize(pole_velocity, bins=bins(-2.0, 2.0, num_digitized))
    ]
    return sum([x*(num_digitized**i) for i, x in enumerate(digitized)])

class QLearning:
    def __init__(
            self,
            n_inputs,
            n_outputs,
            num_digitized,
            gamma,
            alpha
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.num_digitized = num_digitized
        self.gamma = gamma
        self.alpha = alpha

        self.q_table = np.random.uniform(
                low = -1,
                high = 1,
                size = (self.num_digitized**4, self.n_outputs)
        )
        
    def choose_action(
            self, 
            next_state, 
            episode, 
    ):
        epsilon = 0.5 * (1/(episode+1))
        if epsilon <= np.random.uniform(0, 1):
            next_action = np.argmax(self.q_table[next_state])
        else:
            next_action = np.random.choice([0, 1])
        return next_action

    def update_Qtable(
            self,
            state,
            action,
            reward,
            next_state,
            next_action
    ):
        next_Max_Q = max(self.q_table[next_state][0], self.q_table[next_state][1])
        self.q_table[state, action] = (1-self.alpha) * self.q_table[state, action] \
                                      + self.alpha * (reward + self.gamma * next_Max_Q)

def main():

    env = gym.make('CartPole-v0')

    n_inputs = env.observation_space.shape[0]
    n_outputs = env.action_space.n
    num_digitized = 6

    max_episode = 2000
    max_step = 200

    RENDER = False

    agent = QLearning(
            n_inputs = n_inputs,
            n_outputs = n_outputs,
            num_digitized = num_digitized,
            gamma = 0.95,
            alpha = 0.5
    )

    goal_average_reward = 195
    num_consecutice_iteratinos = 100
    total_reward_vec = np.zeros(num_consecutice_iteratinos)

    for episode in range(max_episode):
        observation = env.reset()
        state = digitize_state(observation, num_digitized)
        action = np.argmax(agent.q_table[state])
        episode_reward = 0

        for step in range(max_step):
            if RENDER:
                env.render()
            observation, reward, done, info = env.step(action)

            if done:
                if step<195:
                    reward -= 200
                else:
                    reward = 1
            else:
                reward = 1

            episode_reward += reward

            next_state = digitize_state(observation, num_digitized)
            next_action = agent.choose_action(next_state, episode)

            agent.update_Qtable(
                    state, 
                    action, 
                    reward, 
                    next_state, 
                    next_action
            )

            state = next_state
            action = next_action

            if done:
                print('%d Episode finished after %f time steps / mean %f' %
                      (episode, step + 1, total_reward_vec.mean()))
                total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  
                break
     
        if (total_reward_vec.mean() >= goal_average_reward):
            print('Episode %d train agent successfuly!' % episode)
            break

if __name__ == '__main__':
    main()
