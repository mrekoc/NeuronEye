# For images
#github/aouthors 
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT_NODE = -1
fidelity = False
fidelityValue = 1.7

img = cv2.imread('Data/iron-nut-250x250.jpeg')
imgg = img.copy()
imgCopy = img.copy()
img = cv2.medianBlur(img, 7)
#img = cv2.adaptiveBilateralFilter(img, (5, 5), 150) # Preserve edges
#img = cv2.blur(img, (3,3))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#imgt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 10)
_, imgt = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#_, imgt = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY_INV)
imgt = cv2.morphologyEx(imgt, cv2.MORPH_OPEN, (3, 3))

img2 = imgt.copy()
c, h = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
fidelityRange = 0
if fidelity:
	maxArea = .0
	for i in c: # With images it is convenient to know the greater area
		area = cv2.contourArea(i)
		if area > maxArea:
			maxArea = area
	fidelityRange = maxArea - (maxArea * fidelityValue) # If objects have same size it prevents false detection

totalContours = 0

br = []
for i in range(len(c)):
	if h[0][i][3] == -1 and cv2.contourArea(c[i]) >= fidelityRange:
		totalContours += 1
		approx = cv2.approxPolyDP(c[i], 3, True)
		br.append(cv2.boundingRect(approx))
for b in br:
	cv2.rectangle(imgCopy, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (255, 255, 0), 3)


plt.figure(figsize=(20, 8))

plt.subplot(1,3,1)
plt.imshow(imgg, cmap = 'gray')
plt.subplot(1,3,2)
plt.imshow(imgt, cmap = 'gray')
plt.subplot(1,3,3)
plt.imshow(imgCopy, cmap = 'gray')
    
plt.show()

