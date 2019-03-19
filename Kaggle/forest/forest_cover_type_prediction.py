import tensorflow as tf
import numpy as np
import pandas as pd

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

def load_file(is_test):
    if is_test:
        data_df = pd.read_csv('test.csv.zip', compression='zip')

        data = data_df.values[:, 1:] #id 제외한 데이터들
        labels = data_df["Id"].values #id로만된 데이터들
    else:
        data_df = pd.read_csv('train.csv.zip', compression='zip')

        data = data_df.values[:, 1:-1] #id랑 covertype제외한 데이터들
        labels = data_df["Cover_Type"].values

    return labels, data

y_train, x_train = load_file(0)
y_train -= 1
y_train = np.expand_dims(y_train, 1)
train_len = len(x_train)

testIds, x_test = load_file(1)

x_all = np.vstack((x_train, x_test))

x_min_max_all = MinMaxScaler(x_all)
x_train = x_min_max_all[:train_len]
x_test = x_min_max_all[train_len:]



n_input = x_train.shape[1]

X = tf.placeholder(tf.float32, shape=[None, n_input])
Y = tf.placeholder(tf.int32, shape=[None, 1])
Y_one_hot = tf.one_hot(Y, 7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 7])

W1 = tf.Variable(tf.truncated_normal([n_input, 32], stddev=0.1), name="weight1")
b1 = tf.Variable(tf.truncated_normal([32], stddev=0.1), name="bias1")
L1 = tf.nn.relu(tf.matmul(X, W1)+b1)

W2 = tf.Variable(tf.truncated_normal([32, 7], stddev=0.1), name="weight2")
b2 = tf.Variable(tf.truncated_normal([7], stddev=0.1), name="bias2")
logits = tf.matmul(L1, W2)+b2
hypothesis = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_predict = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

training_epoch = 15
batch_size = 32
display_step = 1
step_size = 1000

saver = tf.train.Saver()
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, "/forest/model.ckpt")
    for epoch in range(training_epoch):
        avg_cost = 0.
        avg_accuracy = 0.

        for step in range(step_size):
            offset = (step*batch_size) % (y_train.shape[0] - batch_size)

            batch_data = x_train[offset:(offset+batch_size),:]
            batch_labels = y_train[offset:(offset+batch_size),:]

            _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={X:batch_data, Y:batch_labels})

            avg_cost += c / step_size
            avg_accuracy += a / step_size

        if epoch % display_step == 0:
            print("epoch : ", epoch, "cost : ", avg_cost, "acc : ", avg_accuracy)
    save_path = saver.save(sess, "/forest/model.ckpt")
    print("Model saved in file: %s" % save_path)

    outputs = sess.run(prediction, feed_dict={X: x_test})
    outputs += 1  # +1 to make 1-7
    submission = ['Id,Cover_Type']

    for id, p in zip(testIds, outputs):
        submission.append('{0},{1}'.format(id, int(p)))

    submission = '\n'.join(submission)

    with open('submission.csv', 'w') as outfile:
        outfile.write(submission)