import numpy as np
import tensorflow as tf
import tensorflow.contrib as conb
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

n_sample = int(mnist.train.num_examples)
batch_size = 128
n_input = 784
n_hidden = 200
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)


X = tf.placeholder(tf.float32, shape=[None, n_input])

w1 = tf.get_variable("w1", shape=[n_input, n_hidden], initializer=conb.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([n_hidden], dtype=tf.float32))
w2 = tf.Variable(tf.zeros([n_hidden, n_input], dtype=tf.float32))
b2 = tf.Variable(tf.zeros([n_input], dtype=tf.float32))

hidden = tf.nn.softplus(tf.add(tf.matmul(X, w1), b1))
reconstruction = tf.add(tf.matmul(hidden, w2), b2)

cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction, X), 2.0))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(30):
        avg_cost = 0.
        total_batch = int(n_sample / batch_size)
        for step in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X:batch_xs})
            avg_cost += c / n_sample * batch_size

        print("Epcch : ", epoch+1, "avg cost : ",avg_cost)

    sample_size = 10
    samples = sess.run(reconstruction, feed_dict={X:X_test[:sample_size]})
    fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

    for i in range(sample_size):
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        ax[1][i].imshow(np.reshape(samples[i], (28,28)))

    plt.show()