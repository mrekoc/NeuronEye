import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
img = cv.imread('Data/ting.jpg')

fgbg = cv.createBackgroundSubtractorMOG2()

fgmask = fgbg.apply(img)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(fgmask),plt.title('Subtrackted')
plt.xticks([]), plt.yticks([])
plt.show()'''

cap = cv.VideoCapture('Data/people-walking.mp4')
fgbg = cv. createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    cv.imshow('original', frame)
    cv.imshow('fg', fgmask)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()