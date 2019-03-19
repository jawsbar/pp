# VGG16

import os
import tensorflow as tf
from load_cifar import Dataset

# Hyper parameters
batch_size = 200 #256
n_batches = int(50000/batch_size)
momentum = 0.9
weight_decay = 0.0005
learning_rate = 0.0001#0.02
img_s = 32 #224
n_labels = 10
is_use_ckpt = True
ckpt_path = os.path.join('ckpt/', 'model.ckpt')
checkpoint_path = os.path.join(os.getcwd(), ckpt_path)
trainable = False

def conv_layer(tensor, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer(mean=0,stddev=0.01))
    tf.add_to_collection(weights, "all_weights")
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    output = tf.nn.conv2d(tensor, weights, strides=[1,1,1,1], padding='SAME')
    return tf.nn.relu(output+biases)

def fc_layer(vector, batch_size, n_in, n_out, activation_fn=tf.nn.relu):
    weights = tf.get_variable("weights", [n_in, n_out], initializer=tf.random_normal_initializer(mean=0,stddev=0.01))
    biases = tf.get_variable("biases", [batch_size, n_out], initializer=tf.constant_initializer(0.0))
    output = tf.add(tf.matmul(vector, weights),biases)
    if activation_fn is not None:
        output = activation_fn(output)
    return output

def max_pool_layer(tensor, image_s):
    image_s = int(image_s/2)
    tensor = tf.nn.max_pool(tensor, [1,2,2,1], [1,2,2,1], "SAME")
    return tensor, image_s

g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32, [None, img_s, img_s, 3], name="image")
    y = tf.placeholder(tf.float32, [None, n_labels])
    dropout_prob = tf.placeholder(tf.float32)


    with tf.variable_scope("conv1"):
        conv1 = conv_layer(X, [3,3,3,64], [img_s, img_s, 64])
    with tf.variable_scope("conv2"):
        conv2 = conv_layer(conv1, [3, 3, 64, 64], [img_s, img_s, 64])
        pool1, img_s = max_pool_layer(conv2, img_s)


    with tf.variable_scope("conv3"):
        conv3 = conv_layer(pool1, [3,3,64,128], [img_s, img_s, 128])
    with tf.variable_scope("conv4"):
        conv4 = conv_layer(conv3, [3, 3, 128, 128], [img_s, img_s, 128])
        pool2, img_s = max_pool_layer(conv4, img_s)

    with tf.variable_scope("conv5"):
        conv5 = conv_layer(pool2, [3, 3, 128, 256], [img_s, img_s, 256])
    with tf.variable_scope("conv6"):
        conv6 = conv_layer(conv5, [3, 3, 256, 256], [img_s, img_s, 256])
    with tf.variable_scope("conv7"):
        conv7 = conv_layer(conv6, [3, 3, 256, 256], [img_s, img_s, 256])
        pool3, img_s = max_pool_layer(conv7, img_s)

    with tf.variable_scope("conv8"):
        conv8 = conv_layer(pool3, [3, 3, 256, 512], [img_s, img_s, 512])
    with tf.variable_scope("conv9"):
        conv9 = conv_layer(conv8, [3, 3, 512, 512], [img_s, img_s, 512])
    with tf.variable_scope("conv10"):
        conv10 = conv_layer(conv9, [3, 3, 512, 512], [img_s, img_s, 512])
        pool4, img_s = max_pool_layer(conv10, img_s)

    with tf.variable_scope("conv11"):
        conv11 = conv_layer(pool4, [3, 3, 512, 512], [img_s, img_s, 512])
    with tf.variable_scope("conv12"):
        conv12 = conv_layer(conv11, [3, 3, 512, 512], [img_s, img_s, 512])
    with tf.variable_scope("conv13"):
        conv13 = conv_layer(conv12, [3, 3, 512, 512], [img_s, img_s, 512])
        pool5, img_s = max_pool_layer(conv13, img_s)

    with tf.variable_scope("fc1"):
        n_in = img_s * img_s * 512
        n_out = 4096
        pool5_1d = tf.reshape(pool5, [batch_size, n_in])
        fc1 = fc_layer(pool5_1d, batch_size, n_in, n_out)
        fc1_drop = tf.nn.dropout(fc1, dropout_prob)

    with tf.variable_scope("fc2"):
        n_in = 4096
        n_out = 4096
        fc2 = fc_layer(fc1_drop, batch_size, n_in, n_out)
        fc2_drop = tf.nn.dropout(fc2, dropout_prob)

    with tf.variable_scope("fc3"):
        n_in = 4096
        n_out = 4096
        logits = fc_layer(fc2_drop, batch_size, n_in, n_labels, activation_fn=None)



    #weight decay
    with tf.variable_scope('weights_norm'):
        weights_norm = tf.reduce_sum(
            input_tensor=weight_decay * tf.stack(
                [tf.nn.l2_loss(i) for i in tf.get_collection('all_weights')]
            ),
            name='weights_norm'
        )
    tf.add_to_collection('losses', weights_norm)

    with tf.variable_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

    tf.add_to_collection('losses', cross_entropy)

    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    # 원래는 모멘텀의 사용
    #train = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(total_loss)
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
    prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))



with tf.Session(graph=g) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    cifar = Dataset(batch_size=batch_size)

    if is_use_ckpt:
        saver.restore(sess, checkpoint_path)

    if trainable:
        X_tr, y_tr = cifar.setup_train_batches()
        #X_tr, y_tr = cifar.load_n_images()
        for epoch in range(30):
            acc = 0.
            print("epoch : ", epoch+1)
            for step in range(n_batches):
                b_step = batch_size * step
                #_, loss, a_tr = sess.run([train, total_loss, accuracy], feed_dict={X: X_tr, y:y_tr, dropout_prob:0.5})
                _, loss, a_tr = sess.run([train, total_loss, accuracy], feed_dict={X: X_tr[b_step:b_step+batch_size],
                                                                                   y: y_tr[b_step:b_step+batch_size],
                                                                                   dropout_prob:0.5})
                acc += a_tr
                if step%100==0:
                    print("Loss : ", loss, "Acc : ", acc/(step+1)) #train acc의 경우 98~99
                    saver.save(sess, ckpt_path)

    else:
        X_val, y_val = cifar.load_test()
        avg_acc = 0.
        for i in range(50):
            b_step = 200 * i
            validation_acc = sess.run(accuracy, feed_dict={X: X_val[b_step:b_step+batch_size], y: y_val[b_step:b_step+batch_size], dropout_prob:1.0})
            avg_acc += validation_acc
        print("val acc : ", (avg_acc/50))


