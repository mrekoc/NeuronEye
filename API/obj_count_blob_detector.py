import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt

sigma = 0.33
im = cv.imread("Data/iron-nut-250x250.jpeg")
imcopy = im.copy()
#im = cv.resize(im, (800,800))
im = cv.medianBlur(im, 9)
v = np.median(im)
kernel = np.ones((3,3),np.uint8)
gray_im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
gaus = cv.adaptiveThreshold(gray_im, 127, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
retval2, otsu = cv.threshold(gaus, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
erode = cv.erode(otsu,kernel,iterations = 1)
dilate = cv.dilate(erode,kernel,iterations = 1)
opening = cv.morphologyEx(dilate, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
closing = cv.erode(closing,kernel,iterations = 2)

params = cv.SimpleBlobDetector_Params()

params.filterByColor = True
params.blobColor = 255

params.filterByArea = True
params.minArea = 500

params.filterByCircularity = True
params.minCircularity = 0.6

ver = (cv.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv.SimpleBlobDetector(params)
else:
    detector = cv.SimpleBlobDetector_create(params)

keypoints = detector.detect(closing)

im_with_keypoints = cv.drawKeypoints (imcopy, keypoints, np.array([]), (255,0,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print("Total number of objects:")
print(len(keypoints))


plt.figure(figsize=(20, 8))

plt.subplot(1,3,1)
plt.imshow(im)
#plt.subplot(1,6,2)
#plt.imshow(imcopy)
#plt.subplot(1,6,3)
#plt.imshow(gray_im, cmap = 'gray')
#plt.subplot(1,3,2)
#plt.imshow(otsu, cmap = 'gray')
plt.subplot(1,3,2)
plt.imshow(closing, cmap = 'gray')
plt.subplot(1,3,3)
plt.imshow(im_with_keypoints)
    
plt.show()

