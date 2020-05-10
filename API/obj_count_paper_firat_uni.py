"""inspired by paper. Firat uni"""

import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt

sigma = 0.33
im = cv.imread("Data/iron-nut-250x250.jpeg")
im= np.uint8(im)
im = cv.resize(im, (800,800))
v = np.median(im)
imcopy = im.copy()
gray_im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
'''edges = cv.Canny(im, 50, 200)
cv.imshow("Source", edges)
cv.waitKey(0)
cv.destroyAllWindows()'''
#hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
#s = hsv [:,:,1]
gaus = cv.adaptiveThreshold(gray_im, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
retval2, otsu = cv.threshold(gaus, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

sobelx = cv.Sobel(otsu,cv.CV_64F,1,0,ksize=5)  # x
sobely = cv.Sobel(otsu,cv.CV_64F,0,1,ksize=5)  # y


lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edges = cv.Canny(otsu, lower, upper)
# Detect points that form a line
# circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1.5, 75)
# if circles is not None:
#     circles = np.round(circles[0, :]).astype("int")

#     for (x, y, r) in circles:
#         cv.circle(imcopy, (x, y), r, (0, 255, 0), 4)
#         cv.rectangle(imcopy, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
size = len(gray_im[0])/16
circles = cv.HoughCircles(otsu, cv.HOUGH_GRADIENT, 1.1, size, param1=50, param2=30, minRadius=0,maxRadius=70)
print(len(circles[0]))
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        cv.circle(imcopy, (x, y), r, (0, 255, 0), 4)
        cv.rectangle(imcopy, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


s_rs = cv.resize(edges, (800,800))
sobelx_rs = cv.resize(sobely, (800,800))
sobely_rs = cv.resize(sobelx, (800,800))
ga_rs = cv.resize(gaus, (800,800))
ot_rs = cv.resize(otsu, (800,800))
#images = [im_rs, s_rs, sobelx_rs, sobely_rs, ga_rs, ot_rs]
'''
cv.imshow("Source", im_rs)
cv.imshow("GRAY", gr_rs)
cv.imshow("Gaus", ga_rs)
cv.imshow("Otsu", ot_rs)
cv.imshow("HSV", hsv_rs)
cv.waitKey(0)
cv.destroyAllWindows()
'''


plt.figure(figsize=(20, 8))

plt.subplot(1,6,1)
plt.imshow(im)
plt.subplot(1,6,2)
plt.imshow(imcopy)
plt.subplot(1,6,3)
plt.imshow(sobelx_rs)
plt.subplot(1,6,4)
plt.imshow(sobely_rs)
plt.subplot(1,6,5)
plt.imshow(ga_rs)
plt.subplot(1,6,6)
plt.imshow(ot_rs)
    
plt.show()

