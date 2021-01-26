import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  
import matplotlib.colors as clr
import numpy as np 
import cv2 as cv
import os
from scipy import ndimage as nd


plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100 

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

def thresh_process(imgs):

    ret = []
    for img in imgs:
        
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        median = cv.medianBlur(gray, 3)
        kernel = np.ones((3,3),np.uint8)

        dilation = cv.morphologyEx(median, cv.MORPH_DILATE, kernel)
        closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel)
        _, thresh = cv.threshold(closing, 250, 255, cv.THRESH_BINARY)
        
        ret.append(thresh)
        
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
    threshes = thresh_process(imgs)
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
    main()
