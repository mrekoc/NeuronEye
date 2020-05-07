"""Thia file produces filtered images with median
filter. Got from pythonimagesearch"""


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

filenames = ['Data/' + format(file) for file in os.listdir('Data/')]

for i in filenames:
    img = cv.imread(i)

    kernel = np.ones((5,5), np.float32)/25
    dst = cv.filter2D(img, -1, kernel)

    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
    plt.xticks([]), plt.yticks([])
    plt.show()
