import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import itertools
from data_parsing import voc_utils
from data_parsing import voc_train

class YOLO_TF:
    fromfile = None
    tofile_img = 'test/output.jpg'
    tofile_txt = 'test/output.txt'
    imshow = True
    filewrite_img = False
    writter = None
    video = False
    filewrite_txt = False
    disp_console = True
    weights_file = 'network/weights/YOLO_small.ckpt'

    # algorihtm variable
    alpha = 0.1
    threshold = 0.2
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    w_img = 640 # 이건 왜
    h_img = 480 # ?

    # training variaible
    training = True
    keep_prob = tf.placeholder(tf.float32)
    lambdacoord = 5.0
    lambdanoobj = 0.5
    label = None
    label = None
    index_in_epoch = 0
    epochs_completed = 0

    def __init__(self, argvs=[]):
        self.argv_parser(argvs)
        self.build_networks()
        if self.training:
            self.build_training()
            self.train()
        print("detection")
        print(self.fromfile)
        if self.fromfile is not None:
            if self.video:
                print("video")
                self.detect_from_file_video(self.fromfile)
            else:
                print("image")
                self.detect_from_file(self.fromfile)

    def argv_parser(self, argvs):
        for i in range(1, len(argvs), 2):
            print(argvs[i])
            if argvs[i] == '-train': self.training = True;
            if argvs[i] == '-fromfile': self.fromfile = argvs[i + 1]
            if argvs[i] == '-tofile_img': self.tofile_img = argvs[i + 1]; self.filewrite_img = True
            if argvs[i] == '-tofile_vid': self.tofile_img = argvs[i + 1]; self.filewrite_img = True
            if argvs[i] == '-tofile_txt': self.tofile_txt = argvs[i + 1]; self.filewrite_txt = True
            if argvs[i] == '-imshow':
                if argvs[i + 1] == '1':
                    self.imshow = True
                else:
                    self.imshow = False
            if argvs[i] == '-disp_console':
                if argvs[i + 1] == '1':
                    self.disp_console = True
                else:
                    self.disp_console = False
            if argvs[i] == '-video':
                self.video = True
                self.fromfile = argvs[i + 1]
                self.filewrite_img = True

    def build_networks(self):
        if self.disp_console: print("Building YOLO_small graph...")
        self.x = tf.placeholder('float32', [None, 448, 448, 3])
        self.conv_1 = self.conv_layer(1, self.x, 64, 7, 2)
        self.pool_2 = self.pooling_layer(2, self.conv_1, 2, 2)
        self.conv_3 = self.conv_layer(3, self.pool_2, 192, 3, 1)
        self.pool_4 = self.pooling_layer(4, self.conv_3, 2, 2)
        self.conv_5 = self.conv_layer(5, self.pool_4, 128, 1, 1)
        self.conv_6 = self.conv_layer(6, self.conv_5, 256, 3, 1)
        self.conv_7 = self.conv_layer(7, self.conv_6, 256, 1, 1)
        self.conv_8 = self.conv_layer(8, self.conv_7, 512, 3, 1)
        self.pool_9 = self.pooling_layer(9, self.conv_8, 2, 2)
        self.conv_10 = self.conv_layer(10, self.pool_9, 256, 1, 1)
        self.conv_11 = self.conv_layer(11, self.conv_10, 512, 3, 1)
        self.conv_12 = self.conv_layer(12, self.conv_11, 256, 1, 1)
        self.conv_13 = self.conv_layer(13, self.conv_12, 512, 3, 1)
        self.conv_14 = self.conv_layer(14, self.conv_13, 256, 1, 1)
        self.conv_15 = self.conv_layer(15, self.conv_14, 512, 3, 1)
        self.conv_16 = self.conv_layer(16, self.conv_15, 256, 1, 1)
        self.conv_17 = self.conv_layer(17, self.conv_16, 512, 3, 1)
        self.conv_18 = self.conv_layer(18, self.conv_17, 512, 1, 1)
        self.conv_19 = self.conv_layer(19, self.conv_18, 1024, 3, 1)
        self.pool_20 = self.pooling_layer(20, self.conv_19, 2, 2)
        self.conv_21 = self.conv_layer(21, self.pool_20, 512, 1, 1)
        self.conv_22 = self.conv_layer(22, self.conv_21, 1024, 3, 1)
        self.conv_23 = self.conv_layer(23, self.conv_22, 512, 1, 1)
        self.conv_24 = self.conv_layer(24, self.conv_23, 1024, 3, 1)
        self.conv_25 = self.conv_layer(25, self.conv_24, 1024, 3, 1, trainable=self.training)
        self.conv_26 = self.conv_layer(26, self.conv_25, 1024, 3, 2, trainable=self.training)
        self.conv_27 = self.conv_layer(27, self.conv_26, 1024, 3, 1, trainable=self.training)
        self.conv_28 = self.conv_layer(28, self.conv_27, 1024, 3, 1, trainable=self.training)
        self.fc_29 = self.fc_layer(29, self.conv_28, 512, flat=True, linear=False, trainable=self.training)
        self.fc_30 = self.fc_layer(30, self.fc_29, 4096, flat=False, linear=False, trainable=self.training)
        self.drop_31 = self.dropout(31, self.fc_30)
        self.fc_32 = self.fc_layer(32, self.drop_31, 1470, flat=False, linear=True, trainable=self.training)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        #self.saver.restore(self.sess, self.weights_file)
        if self.disp_console: print("Loading complete!" + '\n')

    def conv_layer(self, idx, inputs, filters, size, stride, trainable=False):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size, size, int(channels), filters], stddev=0.1), trainable=trainable)
        biases = tf.Variable(tf.constant(0.1, shape=[filters]), trainable=trainable)

        pad_size = size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        inputs_pad = tf.pad(inputs, pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',
                            name=str(idx) + '_conv')
        conv_biased = tf.add(conv, biases, name=str(idx) + '_conv_biased')
        if self.disp_console: print
        '    Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (
            idx, size, size, stride, filters, int(channels))
        return tf.maximum(tf.multiply(self.alpha, conv_biased), conv_biased, name=str(idx) + '_leaky_relu')

    def pooling_layer(self, idx, inputs, size, stride):
        if self.disp_console: print('    Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (
            idx, size, size, stride))
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME',
                              name=str(idx) + '_pool')

    def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False, trainable=False):
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
            inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs
        # weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1), trainable=trainable)
        weight = tf.Variable(tf.zeros([dim, hiddens]), trainable=trainable)
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]), trainable=trainable)
        if self.disp_console: print('    Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (
            idx, hiddens, int(dim), int(flat), 1 - int(linear)))
        if linear: return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx) + '_fc')
        ip = tf.add(tf.matmul(inputs_processed, weight), biases)
        return tf.maximum(tf.multiply(self.alpha, ip), ip, name=str(idx) + '_fc')

    def dropout(self, idx, inputs):
        if self.disp_console: print('    Layer  %d : Type = DropOut' % (idx))
        return tf.nn.dropout(inputs, keep_prob=self.keep_prob)

    def detect_from_cvmat(self, img):
        s = time.time()
        self.h_img, self.w_img, _ = img.shape
        img_resized = cv2.resize(img, (448, 448))
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray(img_RGB)
        inputs = np.zeros((1, 448, 448, 3), dtype='float32')
        inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
        in_dict = {self.x: inputs, self.keep_prob: 1.0}
        net_output = self.sess.run(self.fc_32, feed_dict=in_dict)
        self.result = self.interpret_output(net_output[0])
        strtime = str(time.time() - s)
        if self.disp_console: print('Elapsed time : '+strtime+'secs'+'\n')

    def detect_from_file_video(self, filename):
        if self.disp_console: print(        'Detect from ' + filename)
        input = cv2.VideoCapture(filename)
        while not input.isOpened():
            input = cv2.VideoCapture(filename)
            cv2.waitKey(1000)
            print(            "Wait for the header")
        if self.filewrite_img:
            print(            input.get(cv2.CAP_PROP_FOURCC))
            self.writter = cv2.VideoWriter(self.tofile_img,
                                           int(input.get(cv2.CAP_PROP_FOURCC)),
                                           int(input.get(cv2.CAP_PROP_FPS)),
                                           (int(input.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                            int(input.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            while not self.writter.isOpened():
                self.writter = cv2.VideoWriter(self.tofile_img, -1, input.get(cv2.CAP_PROP_FPS),
                                               [input.get(cv2.CAP_PROP_FRAME_WIDTH),
                                                input.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                                True])
                cv2.waitKey(1000)
                print(                "Wait for the header Writter")
        pos_frame = input.get(cv2.CAP_PROP_POS_FRAMES)
        while True:
            flag, frame = input.read()
            if flag:
                # The frame is ready and already captured
                cv2.imshow('video', frame)
                pos_frame = input.get(cv2.CAP_PROP_POS_FRAMES)
                print(             str(pos_frame) + " frames")
                self.detect_from_cvmat(frame)
                self.show_results(frame, self.result)
            else:
                # The next frame is not ready, so we try to read it again
                input.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                print(                "frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)
            if cv2.waitKey(10) == 27:
                break
            if input.get(cv2.CAP_PROP_POS_FRAMES) == input.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break
        input.release()
        self.writter.release()
        cv2.destroyAllWindows()

    def detect_from_file(self, filename):
        if self.disp_console: print('Detect from '+filename)
        img = cv2.imread(filename)
        self.detect_from_cvmat(img)
        self.show_results(img, self.result)

    def interpret_output(self, output):
        probs = np.zeros((7, 7, 2, 20))
        class_probs = np.reshape(output[0:980], (7, 7, 20))
        scales = np.reshape(output[980:1078], (7, 7, 2))
        boxes = np.reshape(output[1078:], (7, 7, 2, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)] * 14), (2, 7, 7)), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / 7.0
        boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
        boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

        boxes[:, :, :, 0] *= self.w_img
        boxes[:, :, :, 1] *= self.h_img
        boxes[:, :, :, 2] *= self.w_img
        boxes[:, :, :, 3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0: continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1],
                           boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def show_results(self, img, results):
        img_cp = img.copy()
        if self.filewrite_txt:
            ftxt = open(self.tofile_txt, 'w')
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3]) // 2
            h = int(results[i][4]) // 2
            if self.disp_console: print(            '    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(
                y) + ',' + str(int(results[i][3])) + ',' + str(int(results[i][4])) + '], Confidence = ' + str(
                results[i][5]))
            if self.filewrite_img or self.imshow:
                cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
                cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if self.filewrite_txt:
                ftxt.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h) + ',' + str(
                    results[i][5]) + '\n')
        if self.filewrite_img:
            if self.disp_console: print('    image file writed : ' + self.tofile_img)
            if self.video:
                self.writter.write(img_cp)
            else:
                cv2.imwrite(self.tofile_img, img_cp)
        if self.imshow:
            cv2.imshow('YOLO_small detection', img_cp)
            cv2.waitKey(1)
        if self.filewrite_txt:
            if self.disp_console: print(            '    txt file writed : ' + self.tofile_txt)
            ftxt.close()

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2],
                                                                         box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3],
                                                                         box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def build_training(self):  # TODO add training function!
        # the label of image
        self.x_ = tf.placeholder(tf.float32, [None, 7, 7, 2])  # the first dimension (None) will index the images
        self.y_ = tf.placeholder(tf.float32, [None, 7, 7, 2])
        self.w_ = tf.placeholder(tf.float32, [None, 7, 7, 2])
        self.h_ = tf.placeholder(tf.float32, [None, 7, 7, 2])
        self.C_ = tf.placeholder(tf.float32, [None, 7, 7, 2])
        self.p_ = tf.placeholder(tf.float32, [None, 7, 7, 20])
        self.obj = tf.placeholder(tf.float32, [None, 7, 7, 2])
        self.objI = tf.placeholder(tf.float32, [None, 7, 7])
        self.noobj = tf.placeholder(tf.float32, [None, 7, 7, 2])

        # output network
        output = self.fc_32
        nb_image = tf.shape(self.x_)[0]
        class_probs = tf.reshape(output[0:nb_image, 0:980], (nb_image, 7, 7, 20))
        scales = tf.reshape(output[0:nb_image, 980:1078], (nb_image, 7, 7, 2))
        boxes = tf.reshape(output[0:nb_image, 1078:], (nb_image, 7, 7, 2, 4))

        boxes0 = boxes[:, :, :, :, 0]
        boxes1 = boxes[:, :, :, :, 1]
        boxes2 = boxes[:, :, :, :, 2]
        boxes3 = boxes[:, :, :, :, 3]

        # loss funtion
        self.subX = tf.subtract(boxes0, self.x_) # x_ = 7 x 7 x 2 짜리 center bbox 좌표값이 들어있다. 0~6 float형으로 들어있음.
        self.subY = tf.subtract(boxes1, self.y_) # y_ = 7 x 7 x 2 짜리 center bbox 좌표값이 들어있다. 0~6 float형으로 들어있음.
        self.subW = tf.subtract(tf.sqrt(tf.abs(boxes2)), tf.sqrt(self.w_)) # w_ = 7 x 7 x 2 짜리 center bbox width 값인데 이미지 사이즈가 클수록 값이 작음. cell size가 곱해져있음.
        self.subH = tf.subtract(tf.sqrt(tf.abs(boxes3)), tf.sqrt(self.h_)) # h_ = 7 x 7 x 2 짜리 center bbox height 값 상동, 그러나 cell size가 곱해져있지 않음.
        self.subC = tf.subtract(scales, self.C_) # C_ = 최대 1이 두개 들어가있음 7 x 7 x 2
        self.subP = tf.subtract(class_probs, self.p_) # p_ = 0.5 or 1.0 값이 grid cell에 bbox가 두개일경우 7x7x20 안에 0.5가 20개중에 두개 들어가있음
        self.lossX = tf.multiply(self.lambdacoord,
                            tf.reduce_sum(tf.multiply(self.obj, tf.multiply(self.subX, self.subX)), axis=[1, 2, 3])) # X의 차이 제곱 lambdacoord ## obj의 값은 1 or 0 있는지 없는지 확인하기 위한 도구.
        self.lossY = tf.multiply(self.lambdacoord,
                            tf.reduce_sum(tf.multiply(self.obj, tf.multiply(self.subY, self.subY)), axis=[1, 2, 3])) # Y의 차이 제곱
        self.lossW = tf.multiply(self.lambdacoord,
                            tf.reduce_sum(tf.multiply(self.obj, tf.multiply(self.subW, self.subW)), axis=[1, 2, 3])) # w 루트 차이 제곱
        self.lossH = tf.multiply(self.lambdacoord,
                            tf.reduce_sum(tf.multiply(self.obj, tf.multiply(self.subH, self.subH)), axis=[1, 2, 3])) # h 루트 차이 제곱
        self.lossCObj = tf.reduce_sum(tf.multiply(self.obj, tf.multiply(self.subC, self.subC)), axis=[1, 2, 3]) # obj가 있을 경우
        self.lossCNobj = tf.multiply(self.lambdanoobj,
                                tf.reduce_sum(tf.multiply(self.noobj, tf.multiply(self.subC, self.subC)), axis=[1, 2, 3])) # obj가 없는 경우의 loss
        self.lossP = tf.reduce_sum(tf.multiply(self.objI, tf.reduce_sum(tf.multiply(self.subP, self.subP), axis=3)), axis=[1, 2]) # bbox가 있는지 없는지 체크하는 용도의 7 x 7 값은 0 or 1
        self.loss = tf.add_n(
            (self.lossX, self.lossY, self.lossW, self.lossH, self.lossCObj, self.lossCNobj, self.lossP))
        self.loss = tf.reduce_mean(self.loss)

        # variable for the training
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        decay = 0.0005
        end_learning_rate = 0.01
        self.epoch = tf.placeholder(tf.int32)

        # Different case of learning rate
        def lr1():
            return tf.train.polynomial_decay(starter_learning_rate, global_step, decay,
                                             end_learning_rate=end_learning_rate,
                                             power=1.0)

        def lr2():
            return tf.constant(0.01)

        def lr3():
            return tf.constant(0.001)

        def lr4():
            return tf.constant(0.0001)

        lr = tf.case({tf.less_equal(self.epoch, 1): lr1,
                      tf.logical_and(tf.greater(self.epoch, 76), tf.less_equal(self.epoch, 106)): lr2,
                      tf.logical_and(tf.greater(self.epoch, 106), tf.less_equal(self.epoch, 136)): lr3,
                      tf.greater(self.epoch, 136): lr4}, lr4, exclusive=True)

        self.train_step = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(self.loss,
                                                                                              global_step=global_step)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())




    # 제일 중요함
    def build_label(self, img_filenames, epoch):
        X_global = []; Y_global = []; W_global = []; H_global = []; C_global = []
        P_global = []; obj_global = []; objI_global = []; noobj_global = []; Image = []
        for img_filename in img_filenames:
            prelabel = voc_train.get_training_data(img_filename)
            x = np.zeros([7, 7, 2])
            y = np.zeros([7, 7, 2])
            w = np.zeros([7, 7, 2])
            h = np.zeros([7, 7, 2])
            C = np.zeros([7, 7, 2])
            p = np.zeros([7, 7, 20])
            obj = np.zeros([7, 7, 2])
            objI = np.zeros([7, 7])
            noobj = np.ones([7, 7, 2])
            img = voc_utils.load_img(img_filename)
            for i, j in itertools.product(range(0, 7), range(0, 7)):
                if prelabel[i][j] is not None:
                    index = 0
                    while (len(prelabel[i][j]) > index and index < 2):
                        x[i][j][index] = (float(prelabel[i][j][index][0]) / len(img)) * 7 - i # 7x7에서의 center bbox 좌표값x 0~1
                        y[i][j][index] = (float(prelabel[i][j][index][1]) / len(img[0])) * 7 - j # 7x7에서의 center bbox 좌표값y 0~1
                        w[i][j][index] = np.sqrt(prelabel[i][j][index][2]) / len(img) * 7 # 아마도 0~1까지의 값인데 이미지 사이즈가 작을 수록 값이 커지고, 크면 클수록 값이 작아짐.
                        h[i][j][index] = np.sqrt(prelabel[i][j][index][3]) / len(img[0]) # 0~1
                        C[i][j][index] = 1.0
                        p[i][j][self.classes.index(prelabel[i][j][index][4])] = 1.0 / float(len(prelabel[i][j])) # 1.0 / prelabel안에 들어있는 bbox 수 1 or 2  ## 따라서 값은 0.5 or 1.0
                        obj[i][j][index] = 1.0
                        objI[i][j] = 1.0
                        noobj[i][j][index] = 0.0
                        index = index + 1
            X_global.append(x)
            Y_global.append(y)
            W_global.append(w)
            H_global.append(h)
            C_global.append(C)
            P_global.append(p)
            obj_global.append(obj)
            objI_global.append(objI)
            noobj_global.append(noobj)

            # resize the image
            img_resized = cv2.resize(img, (448, 448))
            img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_resized_np = np.asarray(img_RGB)
            inputs = np.zeros((1, 448, 448, 3), dtype='float32')
            inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
            Image.append(inputs[0])
        X_global = np.array(X_global)
        Y_global = np.array(Y_global)
        W_global = np.array(W_global)
        H_global = np.array(H_global)
        C_global = np.array(C_global)
        P_global = np.array(P_global)
        obj_global = np.array(obj_global)
        objI_global = np.array(objI_global)
        noobj_global = np.array(noobj_global)
        Image = np.array(Image)
        return {self.x: Image, self.x_: X_global, self.y_: Y_global, self.w_: W_global, self.h_: H_global,
                self.C_: C_global,
                self.p_: P_global, self.obj: obj_global, self.objI: objI_global, self.noobj: noobj_global,
                self.keep_prob: 0.5, self.epoch: epoch}

    def next_batch(self, batch_size, num_examples):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            self.label = self.label[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= num_examples
        end = self.index_in_epoch
        return self.label[start:end]

    def training_step(self, i, update_test_data, update_train_data):

        # TODO need to create the loop for the training and test
        for nbatch in range(0, int(len(self.label))):
            dict = self.build_label(self.next_batch(1, num_examples=len(self.label)), i)
            self.sess.run(self.train_step, dict)

        train_l = []
        test_l = []

        if update_train_data:
            l = self.sess.run(self.loss, feed_dict=self.build_label(self.label, i))
            train_l.append(l)

        if update_test_data:
            l = self.sess.run(self.loss, feed_dict=self.build_label(self.label_test, i))
            print("\r", i, "loss : ", l)
            test_l.append(l)

        return (train_l, test_l)

    def train(self):
        train_l = []
        test_l = []
        self.label = voc_utils.imgs_from_category_as_list("bird", "train")
        self.label_test = voc_utils.imgs_from_category_as_list("bird", "val")
        training_iter = 137
        epoch_size = 5
        for i in range(training_iter):
            test = False
            if i % epoch_size == 0:
                test = True
            l, tl = self.training_step(i, test, test)
            train_l += l
            test_l += tl
        print("train loss")
        print(train_l)
        print("test loss")
        print(test_l)

if __name__ == '__main__':
        yolo = YOLO_TF(sys.argv)
        cv2.waitKey(1000)




















