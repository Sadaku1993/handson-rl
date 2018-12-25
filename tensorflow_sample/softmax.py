import numpy as np
import tensorflow as tf

logits = tf.placeholder(tf.float32, [None, 2])
labels = tf.placeholder(tf.float32, [None, 2])
sparse_labels = tf.placeholder(tf.int32, [None])

# Use softmax function
softmax = tf.nn.softmax(logits)
output1 = -tf.reduce_sum(tf.log(softmax) * labels, axis=1)

# Use softmax_cross_entropy_with_logits function
output2 = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, 
        logits=logits)

# Use sparse_softmax_cross_entropy_with_logits function
output3 = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = logits,
        labels = sparse_labels)

# Use one_hot function
one_hot = tf.one_hot(sparse_labels, 2)
output4 = -tf.reduce_sum(tf.log(softmax) * one_hot, axis=1)

logit = np.array([[0.7, 0.3],
                  [0.2, 0.8]])

label = np.array([[1, 0],
                  [0, 1]])

sparse_label = np.array([0,
                         1])

with tf.Session() as sess:
    y1, y2, y3, y4 = sess.run(
            [output1, output2, output3, output4],
            feed_dict = {logits: logit, 
                         labels: label,
                         sparse_labels: sparse_label
            }
    )

    one_hot = sess.run(one_hot, feed_dict={sparse_labels: sparse_label})

    print("y1:", y1)
    print("y2:", y2)
    print("y3:", y3)
    print("y4:", y4)
