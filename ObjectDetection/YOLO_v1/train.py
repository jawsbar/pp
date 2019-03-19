import tensorflow as tf
import numpy as np
import os
from utils import VOC #from objectdetection.YOLO.utils import VOC #
import set #import objectdetection.YOLO.set as set #
import model #import objectdetection.YOLO.model as model
import time
model = model.Model()
utils = VOC('train')
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, os.getcwd() + '/model.ckpt')
    #saver.restore(sess, os.getcwd() + '/YOLO_small.ckpt')
    print('load from past checkpoint')

    for i in range(set.epoch):
        last_time = time.time()
        total_loss = 0.
        np.random.shuffle(utils.gt_labels)
        for x in range(0, len(utils.gt_labels) - set.batch_size, set.batch_size):
            images = np.zeros((set.batch_size, set.image_size, set.image_size, 3))
            labels = np.zeros((set.batch_size, set.cell_size, set.cell_size, (set.num_class + set.box_per_cell * 5)))

            for n in range(set.batch_size):
                imname = utils.gt_labels[x + n]['imname']
                flipped = utils.gt_labels[x + n]['flipped']
                images[n, :, :, :] = utils.image_read(imname, flipped)
                labels[n, :, :, :] = utils.gt_labels[x + n]['label']

            loss, _ = sess.run([model.total_loss, model.optimizer],
                               feed_dict={model.images: images, model.labels: labels})
            total_loss += loss

            if (x + 1) % set.checkpoint == 0:
                print('checkpoint reached : ' + str(x + 1))


        print('epoch : ' + str(i + 1) + ', loss : ' + str(
            loss / (len(utils.gt_labels) - set.batch_size / (set.batch_size * 1.0))) + ', s / epoch : ' + str(
            time.time() - last_time))
        saver.save(sess, os.getcwd() + '/model.ckpt')

#saver.restore(sess, os.getcwd() + '\YOLO_small.ckpt')

'''
try:
    saver.restore(sess, os.getcwd() + '\model.ckpt')
    print('load from past checkpoint')
except Exception as e:
    print(e)
    try:
        print('load yolo1')
        saver.restore(sess, os.getcwd() + '\YOLO_small.ckpt')
    except Exception as e:
        print(e)
        print('exit, atleast need a pretrained model')
        exit(0)
'''
















