import os
import tarfile
import pickle
import tensorflow as tf
import numpy as np
from scipy.misc import imresize


# image preprocess
def rotate_reshape(images, output_shape):
    new_images = []
    for img in images:
        img = np.reshape(img, output_shape, order="F")
        img = np.rot90(img, k=3)
        new_images.append(img)
    return new_images

def rescale(images, new_size):
    return list(map(lambda img:imresize(img, new_size), images))

def substract_mean_rgb(images):
    return images - np.round(np.mean(images))


# cifar10 dataset extract / preprocess
def _extract(path, filename):
    fullpath = os.path.join(path, filename)
    if not tarfile.is_tarfile(fullpath):
        raise Exception("'{}' is not a tarfile".format(fullpath))
    with tarfile.open(name=fullpath) as file:
        file.extractall(path=path)

def _unprocessed_batch(dir_path, filename):
    with open(os.path.join(dir_path, filename), 'rb') as file:
        data = pickle.load(file, encoding='bytes')
    images = data[list(data.keys())[2]]
    labels = data[list(data.keys())[1]]
    return [images, labels]

def _load_meta(path="./cifar-10-batches-py/batches.meta"):
    with open(path, 'rb') as file:
        meta = pickle.load(file, encoding='bytes')
    labels_bytes = meta[list(meta.keys())[1]]
    labels = [l.decode("utf-8") for l in labels_bytes]
    return labels

def _one_hot_encode(vector, n_unique):
    one_hot_matrix = np.zeros((np.shape(vector)[0], n_unique))
    for i, y_i in enumerate(vector):
        one_hot_matrix[i, y_i] = 1
    return one_hot_matrix

#preprocess 과정에서 224 x 224로 resize할 경우 메모리 부족이라. 32 x 32로 수정
def _preprocess(images_1d, labels_1d, n_labels=10, dshape=(32, 32, 3), reshape=[32, 32, 3]):
    labels = _one_hot_encode(labels_1d, n_labels)
    images_raw = rotate_reshape(images_1d, dshape)
    images_rescaled = rescale(images_raw, reshape)
    images = substract_mean_rgb(images_rescaled)
    return images, labels


class Dataset:
    def __init__(self, batch_size, data_path="./cifar-10-batches-py"):
        self.n_data = 50000
        self.n_labels = 10
        if self.n_data % batch_size == 0:
            self._batch_size = batch_size
            self._n_batches = int(self.n_data/batch_size)
        else:
            raise Exception("데이터포인트를 배치사이즈로 나눌 수 없다.")

        self._batch_counter = 0
        self.data = {"train":[], "test":[]}
        self.data_path = data_path
        self._setup_tr_complete = False

    def setup_train_batches(self):
        if self._setup_tr_complete:
            raise Exception("setup_train_batches를 이미 불렀다.")
        train_files = ["data_batch_1", "data_batch_2", "data_batch_3",
                       "data_batch_4", "data_batch_5"]
        images_1d, labels_1d = [], []

        for file in train_files:
            images_temp, labels_temp = _unprocessed_batch(self.data_path, file)
            images_1d.append(images_temp)
            labels_1d.append(labels_temp)

        images_1d = np.array(images_1d)
        labels_1d = np.array(labels_1d)
        images_1d = np.reshape(images_1d, (50000, 3072))
        labels_1d = np.reshape(labels_1d, (50000,))

        images, labels = _preprocess(images_1d, labels_1d)

        '''
        for i in range(self._n_batches):
            image_batch = images[i * self._batch_size:(i+1) * self._batch_size]
            label_batch = labels[i * self._batch_size:(i+1) * self._batch_size]
            self.data["train"].append(image_batch, label_batch)
        '''
        self._setup_tr_complete = True
        return images, labels

    def load_n_images(self, n=0, shape=(32, 32, 3)):
        with tf.device('/cpu:0'):
            files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
            random_file = files[np.random.choice(len(files))]
            imgs_1d, labels_1d = _unprocessed_batch(self.data_path, random_file)
            if n != 0:
                rn = np.random.choice(len(labels_1d))
                if (rn+n) > len(labels_1d):
                    rn -= n
                images, labels = _preprocess(imgs_1d[rn:rn+n], labels_1d[rn:rn+n], reshape=shape)
            else:
                images, labels = _preprocess(imgs_1d, labels_1d, reshape=shape)
        return images, labels

    def load_train_batch(self):
        if not self._setup_tr_complete:
            raise Exception("setup_train_batches를 먼저 실행해야됨")
        images, labels = self.data["train"][self._batch_counter]
        self._batch_counter += 1
        if self._batch_counter == self._n_batches:
            self._batch_counter = 0
        return images, labels

    def load_test(self):
        file_name = "test_batch"
        images_1d, labels_1d = _unprocessed_batch(self.data_path, file_name)
        images, labels = _preprocess(images_1d, labels_1d)
        return images, labels


