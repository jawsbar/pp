import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
noise_d = 128
image_d = 784
hidden_d = 256
batch_size = 100

def generator(X):
    hidden = tf.add(tf.matmul(X, g_w1), g_b1)
    hidden = tf.nn.relu(hidden)
    out = tf.add(tf.matmul(hidden, g_w2), g_b2)
    out = tf.nn.sigmoid(out)
    return out

def discriminator(X):
    hidden = tf.add(tf.matmul(X, d_w1), d_b1)
    hidden = tf.nn.relu(hidden)
    out = tf.add(tf.matmul(hidden, d_w2), d_b2)
    out = tf.nn.sigmoid(out)
    return out


g_X = tf.placeholder(tf.float32, shape=[None, noise_d])
d_X = tf.placeholder(tf.float32, shape=[None, image_d])

g_w1 = tf.Variable(tf.random_normal([noise_d, hidden_d], stddev=0.01))
g_b1 = tf.Variable(tf.zeros([hidden_d]))
g_w2 = tf.Variable(tf.random_normal([hidden_d, image_d], stddev=0.01))
g_b2 = tf.Variable(tf.zeros([image_d]))

d_w1 = tf.Variable(tf.random_normal([image_d, hidden_d], stddev=0.01))
d_b1 = tf.Variable(tf.zeros([hidden_d]))
d_w2 = tf.Variable(tf.random_normal([hidden_d, 1], stddev=0.01))
d_b2 = tf.Variable(tf.zeros([1]))


#노이즈를 이용한 이미지 생성
G = generator(g_X)

#노이즈를 이용해 생성된 이미지와 진짜 이미지의 값
D_real = discriminator(d_X)
D_fake = discriminator(G)

#변수들의 묶음
gen_vars = [g_w1, g_w2, g_b1, g_b2]
disc_vars = [d_w1, d_w2, d_b1, d_b2]

gen_loss = -tf.reduce_mean(tf.log(D_fake))
disc_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))


# 각각의 optimize 과정에서 variable들을 구분해서 학습을 시켜준다.
# 예를들어 optimizer_gen 을 통해 학습을 시킬 때, gen_loss를 minimize 해주는데 D_fake값을 구하기 위해
# discriminator를 통과하게 되고, 원래는 그곳에 있는 변수들까지 학습이 되지만
# generator안에 있는 변수들만(var_list 안에 있는 변수들만) 학습을 시킨다.
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(gen_loss, var_list=gen_vars)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(disc_loss, var_list=disc_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_batch = int(mnist.train.num_examples/batch_size)
    for epoch in range(100):
        for step in range(total_batch):
            batch_x, _ = mnist.train.next_batch(batch_size)
            #노이즈의 생성
            z = np.random.uniform(-1., 1., size=[batch_size, noise_d])
            _, _,loss_g, loss_d = sess.run([optimizer_gen, optimizer_disc, gen_loss, disc_loss], feed_dict={g_X: z, d_X:batch_x})


        print("epoch : ",epoch+1, "G LOSS :", loss_g, "D LOSS :", loss_d)

    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        z = np.random.uniform(-1., 1., size=[4, noise_d])
        g = sess.run([G], feed_dict={g_X: z})
        g = np.reshape(g, newshape=(4, 28, 28, 1))
        g = -1 * (g - 1)
        for j in range(4):
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()
    plt.waitforbuttonpress()