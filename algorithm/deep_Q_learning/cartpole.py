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
import tensorflow as tf
import numpy as np

np.random.seed(2)
tf.set_random_seed(2)

class DQN(object):
    def __init__(
            self,
            sess,
            n_features, 
            n_actions,
            lr = 0.00001,
            epsilon = 0.9,
            replace_target_iter = 300,
            gamma = 0.99,
            memory_size = 5000,
            batch_size = 32
    ):
        self.sess = sess
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.memory_counter = 0

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        print(t_params)
        print(e_params)

        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess.run(tf.global_variables_initializer())


    def build_net(self):

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        n_l1 = 32
        n_l2 = 32
        w_initializer = tf.random_normal_initializer(0., 1.)
        b_initializer = tf.constant_initializer(0.1)

        # ----------------- build eval net ----------------------
        with tf.variable_scope('eval_net'):
             c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
             with tf.variable_scope('l1'):
                 w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                 b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                 l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

             with tf.variable_scope('l2'):
                 w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                 b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                 self.q_eval = tf.matmul(l1, w2) + b2
    
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            print(c_names)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if self.epsilon<np.random.uniform(0., 1.):
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, action, reward, next_state))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def learn(self):

        if self.memory_counter < self.memory_size:
            return

        # target modelをアップデートするかどうか
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('replage target model')

        # sample batch memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_eval, q_next = self.sess.run(
                [self.q_eval, self.q_next],
                feed_dict = {
                    self.s   : batch_memory[:, :self.n_features],
                    self.s_  : batch_memory[:, -self.n_features:],
                })

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target = q_eval.copy()
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        
        # train step
        _, self.cost = self.sess.run([self.train_op, self.loss],
                 feed_dict = {
                    self.s : batch_memory[:, :self.n_features],
                    self.q_target  : q_target
                    })

        self.learn_step_counter += 1

    def epsilon_decay(self, episode):
        # self.epsilon = 1. / (1.0 + episode*0.01)
        self.epsilon = 0.001 + 0.9 / (1.0+episode*0.05)
        if self.epsilon<0.1:
            self.epsilon = 0.1

def main():

    env = gym.make('CartPole-v0')

    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n

    sess = tf.Session()

    dqn = DQN(
            sess,
            n_features = n_features,
            n_actions = n_actions
    )

    max_episode = 2000
    max_step = 200
    RENDER = False

    goal_average_reward = 195
    num_consecutice_iterations = 10
    total_reward_vec = np.zeros(num_consecutice_iterations)

    for episode in range(max_episode):
        state = env.reset()
        episode_reward = 0

        for step in range(max_step):
            if RENDER: env.render()
            action = dqn.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            if done:
                if step<195:
                    reward = -1
                else:
                    reward = 1
            else:
                reward = 0

            # store stansition
            dqn.store_transition(state, action, reward, next_state)

            # learn
            # if episode> 10:
            dqn.learn()

            episode_reward+=1
            state = next_state

            if done:
                print('%d Episode finished after %f time steps / mean %f ' %
                      (episode, step + 1, total_reward_vec.mean()))
                total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))   
                dqn.epsilon_decay(episode)
                
                break

        if goal_average_reward < total_reward_vec.mean():
            print("Finish")
            break

if __name__ == '__main__':
    main()
