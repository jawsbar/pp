import models
import pickle
import os
import numpy as np
import tensorflow as tf

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def one_hot_vec(label):
    vec = np.zeros(10)
    vec[label] = 1
    return vec

def load_data():
    x_all = []
    y_all = []
    for i in range (5):
        d = unpickle("cifar-10-batches-py/data_batch_" + str(i+1))
        x_ = d['data']
        y_ = d['labels']
        x_all.append(x_)
        y_all.append(y_)

    d = unpickle('cifar-10-batches-py/test_batch')
    x_all.append(d['data'])
    y_all.append(d['labels'])

    x = np.concatenate(x_all) / np.float32(255)
    y = np.concatenate(y_all)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3))
    
    pixel_mean = np.mean(x[0:50000],axis=0)
    x -= pixel_mean

    y = list(map(one_hot_vec, y))
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return (X_train, Y_train, X_test, Y_test)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_integer('batch_size', 25, 'Batch size')
ckpt_path = os.path.join('ckpt/', 'model.ckpt')
X_train, Y_train, X_test, Y_test = load_data()
trainable = False
batch_size = 200 #128, 256
learning_rate = 0.001

X = tf.placeholder("float", [batch_size, 32, 32, 3])
Y = tf.placeholder("float", [batch_size, 10])


# ResNet Models
#net = models.resnet(X, 20)
net = models.resnet(X, 32)
# net = models.resnet(X, 44)
# net = models.resnet(X, 56)

cross_entropy = -tf.reduce_sum(Y*tf.log(net))
opt = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)
    accsum = 0.
    if trainable:
        for j in range (10):
            for i in range (0, 50000, batch_size):
                feed_dict={
                    X: X_train[i:i + batch_size],
                    Y: Y_train[i:i + batch_size]
                    }
                _, loss = sess.run([opt, cross_entropy], feed_dict=feed_dict)
                if i % 10000 == 0:
                    print("training on image #%d" % i)
                    print("LOSS :", loss)
                    saver.save(sess, ckpt_path)
    else:
        for i in range (0, 10000, batch_size):
            if i + batch_size < 10000:
                acc = sess.run(accuracy,feed_dict={
                    X: X_test[i:i+batch_size],
                    Y: Y_test[i:i+batch_size]
                })
                tf.summary.scalar("accuracy", accuracy)
                accsum += acc
        print("acc : ", accsum/50)


