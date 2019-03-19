import tensorflow as tf
import numpy as np
import collections
import os

class StyleTransfer:

    def __init__(self, content_layer_ids, style_layer_ids, init_image, content_image, save_dir,
                  style_image, session, net, num_iter, loss_ratio, content_loss_norm_type):

        self.net = net
        self.sess = session

        self.CONTENT_LAYERS = collections.OrderedDict(sorted(content_layer_ids.items()))
        self.STYLE_LAYERS = collections.OrderedDict(sorted(style_layer_ids.items()))

        self.save_dir = save_dir

        self.p0 = np.float32(self.net.preprocess(content_image))
        self.a0 = np.float32(self.net.preprocess(style_image))
        self.x0 = np.float32(self.net.preprocess(init_image))

        self.content_loss_norm_type = content_loss_norm_type
        self.num_iter = num_iter
        self.loss_ratio = loss_ratio

        self._build_graph()

    def _build_graph(self):


        self.x = tf.Variable(self.x0, trainable=True, dtype=tf.float32)

        self.p = tf.placeholder(tf.float32, shape=self.p0.shape, name='content')
        self.a = tf.placeholder(tf.float32, shape=self.a0.shape, name='style')

        content_layers = self.net.feed_forward(self.p, scope='content')
        self.Ps = {}
        for id in self.CONTENT_LAYERS:
            self.Ps[id] = content_layers[id]

        style_layers = self.net.feed_forward(self.a, scope='style')
        self.As = {}
        for id in self.STYLE_LAYERS:
            self.As[id] = StyleTransfer._gram_matrix(style_layers[id])

        self.Fs = self.net.feed_forward(self.x, scope='mixed')

        L_content = 0
        L_style = 0

        for id in self.Fs:
            if id in self.CONTENT_LAYERS:

                F = self.Fs[id]
                P = self.Ps[id]

                _, h, w, d = F.get_shape()
                N = h.value * w.value
                M = d.value

                w = self.CONTENT_LAYERS[id]

                if self.content_loss_norm_type == 1:
                    L_content += w * tf.reduce_sum(tf.pow((F-P), 2)) / 2
                elif self.content_loss_norm_type == 2:
                    L_content += w * tf.reduce_sum(tf.pow((F - P), 2)) / (N*M)
                elif self.content_loss_norm_type == 3:
                    L_content +=  w * (1. / (2. * np.sqrt(M) * np.sqrt(N))) * tf.reduce_sum(tf.pow((F - P), 2))

            elif id in self.STYLE_LAYERS:

                F = self.Fs[id]

                _, h, w, d = F.get_shape()
                N = h.value * w.value
                M = d.value

                w = self.STYLE_LAYERS[id]

                G = StyleTransfer._gram_matrix(F)
                A = self.As[id]

                L_style += w * (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G - A), 2))

        alpha = self.loss_ratio
        beta = 1

        self.L_content = L_content
        self.L_style = L_style
        self.L_total = alpha * L_content + beta * L_style

        #all_vars = tf.trainable_variables(scope='mixed')
        self.saver = tf.train.Saver()

    def update(self):

        global _iter
        _iter = 0

        def callback(tl, cl, sl):
            global _iter
            print('iter : %4d, ' % _iter, 'L_total : %g, L_content : %g, L_style : %g' % (tl, cl, sl))
            _iter += 1

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.L_total, method='L-BFGS-B', options={'maxiter': self.num_iter})

        with self.sess as session:
            init_op = tf.global_variables_initializer()
            session.run(init_op)
            if os.path.isfile(self.save_dir+ '/model.meta'):
                self.saver.restore(session, tf.train.latest_checkpoint(self.save_dir))
                print('Finished loading trained model')

            optimizer.minimize(session, feed_dict={self.a:self.a0, self.p:self.p0},
                               fetches=[self.L_total, self.L_content, self.L_style], loss_callback=callback)
            self.saver.save(session, self.save_dir + '/model')
            final_image = session.run(self.x)

        final_image = np.clip(self.net.undo_pre_process(final_image), 0.0, 255.0)

        return final_image

    def result_test(self):

        with self.sess as session:
            init_op = tf.global_variables_initializer()
            session.run(init_op)
            self.saver.restore(session, tf.train.latest_checkpoint(self.save_dir))
            #final_image = session.run(self.x)
            final_image = session.run(self.x)
        final_image = np.clip(self.net.undo_pre_process(final_image), 0.0, 255.0)
        return final_image

    @staticmethod
    def _gram_matrix(tensor):
        shape = tensor.get_shape()

        num_channels = int(shape[3])

        matrix = tf.reshape(tensor, shape=[-1, num_channels])

        gram = tf.matmul(tf.transpose(matrix), matrix)

        return gram