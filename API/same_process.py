import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt

ROOT_NODE = -1
fidelity = False
fidelityValue = .7

im = cv.imread("Data/iron-nut-250x250.jpeg")
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


img2 = closing.copy()
c, h = cv.findContours(img2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
fidelityRange = 0
if fidelity:
	maxArea = .0
	for i in c: # With images it is convenient to know the greater area
		area = cv.contourArea(i)
		if area > maxArea:
			maxArea = area
	fidelityRange = maxArea - (maxArea * fidelityValue) # If objects have same size it prevents false detection

totalContours = 0

br = []
for i in range(len(c)):
	if h[0][i][3] == -1 and cv.contourArea(c[i]) >= fidelityRange:
		totalContours += 1
		approx = cv.approxPolyDP(c[i], 3, True)
		br.append(cv.boundingRect(approx))
for b in br:
	cv.rectangle(imcopy, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (255, 255, 0), 3)

cv.putText(imcopy, "Total number of objects:%3s" % (totalContours),
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
