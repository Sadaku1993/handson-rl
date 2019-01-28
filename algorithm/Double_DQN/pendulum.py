import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)

class DoubleDQN:
    def __init__(
            self,
            sess,
            n_actions,
            n_features,
            learning_rate = 0.005,
            reward_decay = 0.9,
            replace_target_iter = 200,
            memory_size = 1000,
            batch_size = 32,
            epsilon=1.0
    ):
        self.sess = sess
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.replace_target_iter = replace_target_iter
        self.reward_decay = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.memory = np.zeros((self.memory_size, n_features*2+2))
        self.memory_counter = 0

        self.learn_step_counter = 0

        self.build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess.run(tf.global_variables_initializer())
    
    def build_layers(self, s, c_names, n_l1, w_initializer, b_initializer):
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
            l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable('b2', [1, self.n_actions], initializer=w_initializer, collections=c_names)
            output = tf.matmul(l1, w2) + b2
        return output

    def build_net(self):
        n_l1 = 20
        w_initializer = tf.random_normal_initializer(0., 1.)
        b_initializer = tf.constant_initializer(0.1)

        # --------------- build eval net -----------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
        with tf.variable_scope('eval_net'):
            eval_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_eval = self.build_layers(self.s, eval_names, n_l1, w_initializer, b_initializer)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # -------------- build target net ---------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            target_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = self.build_layers(self.s_, target_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if self.epsilon < np.random.uniform():
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('target_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_eval4next, q_next = self.sess.run(
                [self.q_eval, self.q_next],
                feed_dict = {self.s  : batch_memory[:, -self.n_features:],
                             self.s_ : batch_memory[:, -self.n_features:]
                })
        q_eval = self.sess.run(
                self.q_eval,
                feed_dict = {self.s : batch_memory[:, :self.n_features]
                })
        
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        max_act4next = np.argmax(q_eval4next, axis=1)
        selected_q_next = q_next[batch_index, max_act4next]

        q_target[batch_index, eval_act_index] = reward + self.reward_decay * selected_q_next

        _, self.cost = self.sess.run(
                [self.train_op, self.loss],
                feed_dict={self.s: batch_memory[:, :self.n_features],
                           self.q_target: q_target
                })


        self.learn_step_counter += 1

    def epsilon_decay(self, episode):
        self.epsilon = 1. / (1.0 + episode*0.01)
        if self.epsilon<0.1:
            self.epsilon = 0.1

def main():

    env = gym.make('Pendulum-v0')
    
    n_actions = 11
    n_features = env.observation_space.shape[0]
    learning_rate = 0.005
    reward_decay = 0.9
    memory_size = 3000

    sess = tf.Session()

    model = DoubleDQN(
        sess = sess,
        n_actions = n_actions,
        n_features = n_features,
        learning_rate = learning_rate,
        reward_decay = reward_decay,
        memory_size = memory_size
    )

    RENDER = False
    max_steps = 200000
    
    total_steps = 0
    state = env.reset()
    
    while True:
        if RENDER: env.render()
        
        action = model.choose_action(state)
        f_action = (action-(n_actions-1)/2.)/((n_actions-1)/4.)
        
        next_state, reward, done, _ = env.step(np.array([f_action]))
        reward /= 10.

        model.store_transition(state, action, reward, next_state)

        if memory_size < total_steps:
            model.learn()

        if max_steps < total_steps - memory_size:
            break

        if max_steps*0.9 < total_steps:
            RENDER = True
    
        state = next_state
        total_steps += 1

if __name__ == '__main__':
    main()
