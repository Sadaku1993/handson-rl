import numpy as np
import tensorflow as tf

outputs  = tf.placeholder(tf.float32, [None, 2])
actions  = tf.placeholder(tf.float32, [None, 2])
rewards  = tf.placeholder(tf.float32, [None])

softmax = tf.nn.softmax(outputs)
cross_entropy = -tf.reduce_sum(tf.log(softmax) * actions, axis=1)
loss = cross_entropy * rewards

softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=outputs,
                                                                labels=actions)

output = np.array([[0.7, 0.3],
                   [0.7, 0.3],
                   [1, 2],
                   [1, 2]])
action = np.array([[0, 1],
                   [1, 0],
                   [1, 0],
                   [1, 0]])

reward = np.array([1,
                   1,
                   5,
                   1])

with tf.Session() as sess:
    print("loss")
    loss_, cross_entropy_,softmax_cross_entropy_ = sess.run(
            [loss, cross_entropy, softmax_cross_entropy],
            feed_dict={outputs:output,
                       actions:action,
                       rewards:reward}
    )
print(cross_entropy_)
print(softmax_cross_entropy_)
print(loss_)

