#!/usr/bin/env python
# coding: utf-8

# In[51]:

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg  
import matplotlib.colors as clr
import numpy as np 
import cv2 as cv
import os
import time

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100 

imge = cv.imread("../test_data/screws_006.png")
imge = cv.cvtColor(imge, cv.COLOR_BGR2RGB)
imge = cv.resize(imge, (480, 360))


def read_imgs(folder_name):
    imgs = []
    files = []
    folder = folder_name

    for filename in sorted(os.listdir(folder)):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (480, 360))
            imgs.append(img)
            files.append(filename)

    return imgs, files

def saturation_process(imgs):
    
    ret = []
    for img in imgs:
        kernel = np.ones((3,3),np.uint8)
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        hsv = hsv[:,:,1]

        dilation = cv.morphologyEx(hsv, cv.MORPH_OPEN, kernel)

        blur = cv.GaussianBlur(dilation, (3,3), 0)

        gaus = cv.Canny(blur,250,255)
        gaus = cv.dilate(gaus, kernel, iterations = 1)
        gaus = cv.erode(gaus, kernel, iterations = 1)
        
        ret.append(gaus)
        
    return ret


def find_contour(img, target):
    fidelity = False
    fidelityValue = 1.7
    detected = img.copy()
    _, c, h = cv.findContours(target, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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
        cv.rectangle(detected, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (255, 0, 0), 3) 
    
    return totalContours, detected



def main():
    imgs, files = read_imgs("../test_data")
    threshes = saturation_process(imgs)
    finals = []
    totals = []
    for (img, thresh, file) in zip(imgs, threshes, files):
        total, final = find_contour(img, thresh)
        finals.append(final)
        totals.append(total)
        #fig, axs = plt.subplots(1,2)
        #axs[0].imshow(img, cmap = 'gray')
        #axs[0].set_title('Original image')
        #axs[1].imshow(final, cmap = 'gray')
        #axs[1].set_title('Image w/ detected objects')
        
        print("Filename:" , file, "Detected objects:", total)
        
    save = zip(imgs, threshes, files, finals, totals)


if __name__ == '__main__':
    start = time.process_time()
    main()
    print(time.process_time() - start)

# In[ ]:
