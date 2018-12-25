import tensorflow as tf
import numpy as np

sparse_logits = tf.placeholder(tf.float32, shape=[None, 2])
sparse_labels = tf.placeholder(tf.int32, shape=[None])

sparse_softmax_corss_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = sparse_logits,
    labels = sparse_labels)

one_hot = tf.one_hot(sparse_labels, 2)
softmax = tf.nn.softmax(sparse_logits)
one_hot_softmax_cross_entropy =  - tf.reduce_sum(tf.log(softmax) * one_hot, axis=1)

logits = np.array([[0.7, 0.3],
                   [0.7, 0.3],
                   [1, 2],
                   [1, 2]])

labels = np.array([0,
                   1,
                   0,
                   1])

with tf.Session() as sess:
    result = sess.run(
            sparse_softmax_corss_entropy, 
                feed_dict = {sparse_logits:logits,
                             sparse_labels:labels}
    )

    one_hot_result = sess.run(
                   one_hot_softmax_cross_entropy,
                   feed_dict = {sparse_logits:logits,
                                sparse_labels:labels}
    )

    print(result)
    print(one_hot_result)
