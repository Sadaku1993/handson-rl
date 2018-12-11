import tensorflow as tf
import numpy as np
import gym

class PolicyGradient:
    def __init__(
            self, 
            n_inputs, 
            n_outputs, 
            n_hidden,
    ):
        self.sess = tf.Session()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden

        self.build_model()

        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_inputs])
        
        layer1 = tf.layers.dense(
                inputs=self.X, 
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

    def choose_action(self, observation):
        action = self.sess.run(self.action, feed_dict={self.X:observation})
        return action

def main():
    
    env = gym.make('CartPole-v0')
    
    n_inputs = env.observation_space.shape[0]
    n_hidden = 4
    n_outputs = 2
    
    max_episodes = 200
    max_steps = 200

    agent = PolicyGradient(
            n_inputs = n_inputs,
            n_outputs = n_outputs,
            n_hidden = n_hidden)

    for episode in range(max_episodes):
        observation = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            env.render()
            action = agent.choose_action(observation.reshape(1, n_inputs))
            observatoin, reward, done, _ = env.step(action[0][0])
            episode_reward += reward
            if done:
                print("episode:%3d reward:%3d" % (episode, episode_reward))
                break

if __name__ == '__main__':
    main()
