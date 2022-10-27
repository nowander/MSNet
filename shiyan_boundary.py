import cv2

dst = '/home/zy/PycharmProjects/zy/newdata/train/NJUNLPR/bound/'
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os

gtTest = os.listdir('/home/zy/PycharmProjects/zy/newdata/train/NJUNLPR/GT')
gtTest = [os.path.join('/home/zy/PycharmProjects/zy/newdata/train/NJUNLPR/GT', gtimg) for gtimg in gtTest]

for img in gtTest:
    # data = Image.open(img)
    # arr = np.asarray(data)
    arr = cv2.imread(img, 0)
    sobelx = cv2.Laplacian(arr, cv2.CV_64FC3, ksize=3)
    sobely = cv2.Laplacian(arr, cv2.CV_64FC3, ksize=3)
    gm = cv2.sqrt(sobelx ** 2, sobely ** 2)
    print(gm.shape)
    name = img.split('/')[-1]
    cv2.imwrite(dst + name, gm)
