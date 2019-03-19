import os
import math
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import set#import objectdetection.YOLO.set as set

class VOC:
    def __init__(self, phase):
        self.data_path = os.path.join('VOCdevkit', 'VOC2012')
        self.image_size = set.image_size
        self.cell_size = set.cell_size
        self.classes = set.classes_name
        self.class_to_ind = set.classes_dict
        self.flipped = set.flipped
        self.phase = phase
        self.gt_labels = None
        self.prepare()

    def image_read(self, imname, flipped = False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        if self.flipped:
            print("flipped example 추가")
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = self.image_size - 1 - gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        cache_file = 'pascal_' + self.phase + '_labels.pkl'
        if os.path.isfile(cache_file):
            print('Loading gt_labels from : ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from : ', self.data_path)

        self.image_index = os.listdir(os.path.join('VOCdevkit', 'VOC2012', 'JPEGImages'))
        self.image_index = [i.replace('.jpg', '') for i in self.image_index]

        import random
        random.shuffle(self.image_index)

        if self.phase == 'train':
            val = int(len(self.image_index) * (1 - set.test_percentage))
            self.image_index = self.image_index[:val]
        else:
            val = int(len(self.image_index) * set.test_percentage)
            self.image_index = self.image_index[:val]

        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imname':imname, 'label':label, 'flipped':False})
        print('Saving gt_labels to : ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels



    # 지금 이 코드는 box per cell에 관해서 처리가 안되어있고, 약식으로 대충 써놓은 코드.
    # 어떻게 고치지...
    def load_pascal_annotation(self, index):
        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]

        label = np.zeros((self.cell_size, self.cell_size, 30))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            # xml 파일에서 바운딩박스부분을(boundingbox of ground truth) 고정된 image size(448)에 맞게 비율 조정
            '''
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            
            # boxes = [bbox의 x축 중간 좌표값, bbox의 y축 중간 좌표값, bbox x 크기, bbox y 크기]
            # cell로 나눠지지 않은 448에 맞춘 x, y center값들
            '''
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]

            width = float(xmax - xmin)
            height = float(ymax - ymin)
            center_x = float(xmin - xmax) / 2
            center_y = float(ymin - ymax) / 2

            x_ind = int(math.floor(center_x / width * (7 - 1)))
            y_ind = int(math.floor(center_y / height * (7 - 1)))

            width_b = math.sqrt(width) / im.shape[0] * 7.
            height_b = math.sqrt(height) / im.shape[1]
            center_x_b = center_x / im.shape[0] * 7 - float(y_ind)
            center_y_b = center_y / im.shape[1] * 7 - float(x_ind)

            boxes = [center_x_b, center_y_b, width_b, height_b]

            # bbox의 중심좌표값을 (x_ind, y_ind)  cell grid(7 x 7)의 좌표값으로
            # 맞춰서 변환 0~7까지 int
            '''
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            '''
            if label[y_ind, x_ind, 4] == 1:
                if label[y_ind, x_ind, 9] == 1:
                    continue
                label[y_ind, x_ind, 9] = 1
                label[y_ind, x_ind, 5:9] = boxes
                label[y_ind, x_ind, 10 + cls_ind] = 1.
                label[y_ind, x_ind, 10:] *= 0.5
                continue
            # 0의 경우 labeling 여부
            # 1~4번까지 boxes의 정보
            # 5번~24번까지 20개의 class index 중 일치하는 class에 1
            # ex) 'aeroplane'일 경우 index번호가 0번이므로 5 + 0 = 5번에 1을 넣음으로써 classification
            # 따라서 7x7의 grid 각 cell 마다 class정보와 bbox의 좌표값등 정보가 들어있음.
            # label : 7 x 7 x 25
            label[y_ind, x_ind, 4] = 1
            label[y_ind, x_ind, 0:4] = boxes
            label[y_ind, x_ind, 10 + cls_ind] = 1.
        # label값과 object 개수 리턴
        return label, len(objs)