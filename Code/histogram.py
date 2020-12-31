# importing required libraries of opencv 
import cv2 
  
# importing library for plotting 
from matplotlib import pyplot as plt 
import numpy as np
  
# reads an input image 
img = cv2.imread('test_data/ting_roi.png')
img = cv2.resize(img, (800, 600))
img_plt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(np.shape(img_plt))
#red = img[...,0]
#img = cv2.blur(img,(5,5)) 
# cv2.imshow("Red", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.plot(img_plt) 
plt.show()

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

# find frequency of pixels in range 0-255 
histr = cv2.calcHist([img],[0],None,[256],[0,256]) 

# show the plotting graph of an image 
plt.plot(histr) 
plt.show()