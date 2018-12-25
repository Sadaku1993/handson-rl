import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

class PolicyGradient:
    def __init__(
            self, 
            n_inputs, 
            n_outputs, 
            n_hidden,
            learning_rate=0.01,
            gamma = 0.95,
            output_graph = False
    ):
        self.sess = tf.Session()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.ep_observation, self.ep_action, self.ep_reward = [], [], []

        self.build_model()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def build_model(self):

        with tf.name_scope('input'):
            self.tf_obs = tf.placeholder(
                    tf.float32, 
                    shape=[None, self.n_inputs], 
                    name="observations"
            )
            self.tf_acts = tf.placeholder(
                    tf.int32, 
                    shape=[None, ], 
                    name="actions_num"
            )
            self.tf_value = tf.placeholder(
                    tf.float32,
                    shape=[None, ],
                    name="actions_value"
            )
        
        layer1 = tf.layers.dense(
                inputs=self.tf_obs, 
                units=self.n_hidden, 
                activation=None,
                kernel_initializer = tf.random_normal_initializer(),
                bias_initializer = tf.constant_initializer()
        )
        action_prob = tf.layers.dense(
                inputs = layer1, 
                units=self.n_outputs, 
                activation=tf.nn.softmax, 
                kernel_initializer = tf.variance_scaling_initializer(),
                bias_initializer = tf.constant_initializer()
        )

        self.action = tf.multinomial(tf.log(action_prob), num_samples=1)

        with tf.name_scope('loss'):
            self.neg_log_prob = -tf.reduce_sum(tf.log(action_prob) * tf.one_hot(self.tf_acts, self.n_outputs), axis=1)
            loss = tf.reduce_mean(self.neg_log_prob * self.tf_value)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def choose_action(self, observation):
        action = self.sess.run(
                self.action, 
                feed_dict={self.tf_obs:observation}
        )
        return action

    def learn(self):
        discount_ep_rs_norm = self._disconut_and_norm_rewards()

        self.sess.run(self.train_op, feed_dict={
            self.tf_obs   : np.vstack(self.ep_observation),
            self.tf_acts  : np.array(self.ep_action),
            self.tf_value : discount_ep_rs_norm
        })

        self.ep_observation, self.ep_action, self.ep_reward = [], [], []

        return discount_ep_rs_norm

    def store_transition(self, observation, action, reward):
        self.ep_observation.append(observation)
        self.ep_action.append(action)
        self.ep_reward.append(reward)

    def _disconut_and_norm_rewards(self):
        discount_ep_rs = np.zeros_like(self.ep_reward)
        running_add = 0
        for t in reversed(range(0, len(self.ep_reward))):
            running_add = running_add * self.gamma + self.ep_reward[t]
            discount_ep_rs[t] = running_add

        discount_ep_rs -= np.mean(discount_ep_rs)
        discount_ep_rs /= np.std(discount_ep_rs)
        return discount_ep_rs

def main():
    env = gym.make('CartPole-v0')

    print("action space:", env.action_space)
    print("observation spaace:", env.observation_space)
    
    n_inputs = env.observation_space.shape[0]
    n_hidden = 4
    n_outputs = 2
    
    max_episodes = 300
    max_steps = 200

    xlabel = []
    ylabel = []

    agent = PolicyGradient(
            n_inputs = n_inputs,
            n_outputs = n_outputs,
            n_hidden = n_hidden)

    for episode in range(max_episodes):
        observation = env.reset()
        
        for step in range(max_steps):
            # env.render()
            action = agent.choose_action(observation.reshape(1, n_inputs))
            observation_, reward, done, _ = env.step(action[0][0])
            agent.store_transition(observation, action[0][0], reward)

            if done:
                episode_reward = sum(agent.ep_reward)
                print("episode:%3d reward:%3d" % (episode, episode_reward))
                agent.learn()
                xlabel.append(episode)
                ylabel.append(episode_reward)
                break
            observation = observation_

    plt.plot(xlabel, ylabel, 'b-')
    plt.xlabel('episode steps')
    plt.ylabel('episode rewards')
    plt.show()

if __name__ == '__main__':
    main()
