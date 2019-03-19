classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes_no = [i for i in range(len(classes_name))]
classes_dict = dict(zip(classes_name, classes_no))
num_class = len(classes_name)

model_type = 'yolo'

image_size = 448
cell_size = 7
box_per_cell = 2
alpha_relu = 0.1
no_object_scale = 0.5
coordinate_scale = 5.0
flipped = False

decay_step = 30000
decay_rate = 0.1
learning_rate = 0.0001
dropout = 0.5
batch_size = 3
epoch = 1000
checkpoint = 1000

# For main
threshold = 0.2
IOU_threshold = 0.5
test_percentage = 0.99

# output 값에 따른 방법
# 1 for read a picture
# 2 to read from testing dataset
# 3 to read from video
output = 1
# output = 2일 경우
picture_name = 'group.jpg'