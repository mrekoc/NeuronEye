import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt

sigma = 0.33
im = cv.imread("Data/ting12.png")
imcopy = im.copy()
#im = cv.resize(im, (800,800))
im = cv.medianBlur(im, 7)
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


lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edges = cv.Canny(closing, lower, upper)
# Detect points that form a line
size = len(gray_im[0])/16
circles = cv.HoughCircles(closing, cv.HOUGH_GRADIENT, 1.1, size, param1=50, param2=30, minRadius=0,maxRadius=70)
num = len(circles[0])
print(len(circles[0]))
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        cv.circle(imcopy, (x, y), r, (0, 255, 0), 4)
        cv.rectangle(imcopy, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


cv.putText(imcopy, "Total number of objects:%3s" % (num),
		(imcopy.shape[1] - 1200, imcopy.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX,
		2.3, (0, 0, 0), 6)

plt.figure(figsize=(20, 8))

plt.subplot(1,3,1)
plt.imshow(im)
plt.subplot(1,3,2)
plt.imshow(closing)
#plt.subplot(1,6,3)
#plt.imshow(gray_im)
#plt.subplot(1,6,4)
#plt.imshow(edges)
#plt.subplot(1,6,5)
#plt.imshow(otsu)
plt.subplot(1,3,3)
plt.imshow(imcopy)
    
plt.show()

