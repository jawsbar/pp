import tensorflow as tf
import set
import numpy as np

slim = tf.contrib.slim


class Model:

    def __init__(self, training=True):
        self.classes = set.classes_name
        self.num_classes = len(set.classes_name)
        self.image_size = set.image_size
        self.cell_size = set.cell_size
        self.boxes_per_cell = set.box_per_cell
        self.output_size = (self.cell_size * self.cell_size) * (self.num_classes + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_classes
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell



        self.no_object_scale = set.no_object_scale
        self.coord_scale = set.coordinate_scale

        self.offset = np.transpose(
            np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                       (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(tf.float32, [None, set.image_size, set.image_size, 3])

        if set.model_type == 'yolo':
            self.logits = self.build_network(self.images, num_outputs=self.output_size, alpha=set.alpha_relu,
                                             training=training)
        elif set.model_type == 'tiny':
            self.logits = self.build_network_tiny(self.images, num_outputs=self.output_size, alpha=set.alpha_relu,
                                                  training=training)

        if training:
            self.batch = tf.Variable(0)
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, (5 * self.boxes_per_cell + self.num_classes)])
            self.loss_layer(self.logits, self.labels)

            self.total_loss = tf.contrib.losses.get_total_loss()
            self.learning_rate = tf.train.exponential_decay(set.learning_rate, self.batch * set.batch_size,
                                                       set.decay_step, set.decay_rate, True)
            self.learning_rate = 0.0001
            self.optimizer = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=self.learning_rate).minimize(self.total_loss, global_step=self.batch)
            #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

    def build_network(self, images, num_outputs, alpha, keep_prob=set.dropout, training=True, scope='yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=leaky_relu(alpha),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
                net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(net, keep_prob=keep_prob, is_training=training, scope='dropout_35')
                net = slim.fully_connected(net, num_outputs, activation_fn=None, scope='fc_36')
        return net

    def build_network_tiny(self, images, num_outputs, alpha, keep_prob=set.dropout, training=True, scope='yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=leaky_relu(alpha),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
                net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(net, keep_prob=keep_prob, is_training=training, scope='dropout_35')
                net = slim.fully_connected(net, num_outputs, activation_fn=None, scope='fc_36')
        return net

    # interpret 과정을 옮겨서 loss에 해야 할지 말아야 할지

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)



    # 아직 수정해야될 사항이 많음. 학습시간 : 4~5h  / gpu : gtx1080
    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(predicts[:, :self.boundary1],
                                         [set.batch_size, self.cell_size, self.cell_size, self.num_classes])
            predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2],
                                        [set.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape(predicts[:, self.boundary2:],
                                       [set.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            response1 = tf.reshape(labels[:, :, :, 4], [set.batch_size, self.cell_size, self.cell_size, 1])
            response2 = tf.reshape(labels[:, :, :, 9], [set.batch_size, self.cell_size, self.cell_size, 1])
            response = tf.concat([response1, response2], axis=3) # 7 x 7 x 2

            boxes1 = tf.reshape(labels[:, :, :, 0:4], [set.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes2 = tf.reshape(labels[:, :, :, 5:9], [set.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes =  tf.concat([boxes1, boxes2], axis=3)

            classes = labels[:, :, :, 10:]


            #boxes = [batch, 7, 7, 2, 4]
            boxes_x = tf.reshape(boxes[:, :, :, :, 0], [set.batch_size, self.cell_size, self.cell_size, 2])
            boxes_y = tf.reshape(boxes[:, :, :, :, 1], [set.batch_size, self.cell_size, self.cell_size, 2])
            boxes_w = tf.reshape(boxes[:, :, :, :, 2], [set.batch_size, self.cell_size, self.cell_size, 2])
            boxes_h = tf.reshape(boxes[:, :, :, :, 3], [set.batch_size, self.cell_size, self.cell_size, 2])


            # 난수 방지용
            # 0인 값들에 루트씌우면 NAN 값이 나올 수 있음. 임시방편으로 난수방지용 0.0001
            predict_box_x = tf.reshape(predict_boxes[:, :, :, :, 0], [set.batch_size, self.cell_size, self.cell_size, 2])
            predict_box_y = tf.reshape(predict_boxes[:, :, :, :, 1], [set.batch_size, self.cell_size, self.cell_size, 2])
            predict_box_w = tf.reshape(predict_boxes[:, :, :, :, 2], [set.batch_size, self.cell_size, self.cell_size, 2]) + 0.0001
            predict_box_h = tf.reshape(predict_boxes[:, :, :, :, 3], [set.batch_size, self.cell_size, self.cell_size, 2]) + 0.0001
            predict_box_w = tf.sqrt(tf.abs(predict_box_w))
            predict_box_h = tf.sqrt(tf.abs(predict_box_h))

            # 논문에서의 loss function을 따름.
            sub_x = predict_box_x - boxes_x
            coord_loss_x = tf.reduce_mean(tf.reduce_sum(tf.multiply(response, tf.square(sub_x)), axis=[1, 2, 3]),
                                          name='coord_loss') * self.coord_scale
            sub_y = predict_box_y - boxes_y
            coord_loss_y = tf.reduce_mean(tf.reduce_sum(tf.multiply(response, tf.square(sub_y)), axis=[1, 2, 3]),
                                          name='coord_loss') * self.coord_scale
            sub_w = predict_box_w - tf.sqrt(boxes_w)
            coord_loss_w = tf.reduce_mean(tf.reduce_sum(tf.multiply(response, tf.square(sub_w)), axis=[1, 2, 3]),
                                          name='coord_loss') * self.coord_scale
            sub_h = predict_box_h - tf.sqrt(boxes_h)
            coord_loss_h = tf.reduce_mean(tf.reduce_sum(tf.multiply(response, tf.square(sub_h)), axis=[1, 2, 3]),
                                          name='coord_loss') * self.coord_scale

            # 7 x 7 x 20 의 값에 bbox가 두개면 0.5가 두개 들어있음. 하나면 1.0 하나
            class_list = tf.reshape(response1, [set.batch_size, self.cell_size, self.cell_size])
            class_delta = (predict_classes - classes) # 7 x 7 x 20 짜리임
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(class_list ,tf.reduce_sum(tf.square(class_delta), axis=3)), axis=[1, 2]),
                                        name='class_loss')

            # 7 x 7 x 2 의 1만 들어있는 값들
            object_delta = response * (predict_scales - response)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                                         name='object_loss')

            # noobject mask의 값만 전환시키기
            noobject_mask = tf.ones([set.batch_size, self.cell_size, self.cell_size, 2]) - response
            noobject_delta = noobject_mask * (predict_scales - response)
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                                           name='noobject_loss') * self.no_object_scale

            tf.contrib.losses.add_loss(coord_loss_x)
            tf.contrib.losses.add_loss(coord_loss_y)
            tf.contrib.losses.add_loss(coord_loss_w)
            tf.contrib.losses.add_loss(coord_loss_h)
            tf.contrib.losses.add_loss(object_loss)
            tf.contrib.losses.add_loss(noobject_loss)
            tf.contrib.losses.add_loss(class_loss)



def leaky_relu(alpha):
    def op(inputs):
        return tf.maximum(alpha * inputs, inputs)

    return op