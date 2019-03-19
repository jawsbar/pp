import tensorflow as tf
import numpy as np
import pandas as pd
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

def load_file(is_test):
    if is_test:
        data_df = pd.read_csv("bike/test.csv")
        data = data_df.values[:, 1:]
        labels = data_df["datetime"].values
    else:
        data_df = pd.read_csv("bike/train.csv")
        data = data_df.values[:, 1:-3]
        labels = data_df["count"].values

    return data, labels

x_train, y_train = load_file(0)

y_train = np.expand_dims(y_train, 1)
train_len = len(x_train)

x_test, testIds = load_file(1)
all_x = np.vstack((x_train, x_test))

x_min_max_all = MinMaxScaler(all_x)
x_train = x_min_max_all[:train_len]
x_test = x_min_max_all[train_len:]

learning_rate = 0.1

# Network Parameters
n_input = x_train.shape[1]
n_classes = 1  # regression

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, n_input])
Y = tf.placeholder(tf.float32, [None, 1])  # 0 ~ 6

W = tf.Variable(tf.random_normal([n_input, n_classes]), name='weight')
b = tf.Variable(tf.random_normal([n_classes]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

training_epochs = 100
batch_size = 32
display_step = 10
step_size = (int)(x_train.shape[0]/batch_size)+1
print(step_size)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over step_size
        for step in range(step_size):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (y_train.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = x_train[offset:(offset + batch_size), :]
            batch_labels = y_train[offset:(offset + batch_size), :]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_data,
                                                          Y: batch_labels})
            avg_cost += c / step_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%02d' % (epoch + 1), "cost={:.4f}".format(avg_cost))


    outputs = sess.run(hypothesis, feed_dict={X: x_test})
    outputs = np.where(outputs < 0, 0, outputs)
    submission = ['datetime,count']

    for id, p in zip(testIds, outputs):
        submission.append('{0},{1}'.format(id, int(p)))

    submission = '\n'.join(submission)

    with open('submission.csv', 'w') as outfile:
        outfile.write(submission)
