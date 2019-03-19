import cv2
import numpy as np

n = 4
fused_image = [0] * n
image = np.zeros(shape=[96, 96, 3])
image += 1.1111

for i in range(n):
    fused_image[i] = []
    for j in range(n):
        k = i * n + j
        image[k] = (image[k] + 1) * 127.5
        fused_image[i].append(image[k])
    fused_image[i] = np.hstack(fused_image[i])
fused_image = np.vstack(fused_image)
print(fused_image.shape)
