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

def morph_saturation(imgs):
    
    ret = []
    
    for img in imgs:
        kernel = np.ones((3,3),np.uint8)

        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        hsv = hsv[:,:,1]

        blur = cv.medianBlur(hsv, 5)
        gaus = cv.Canny(blur,250,255)
        gaus = cv.medianBlur(gaus, 1)
        gaus = cv.dilate(gaus, kernel, iterations = 1)
        
        ret.append(gaus)
        
    return ret

def hough_circles(img, target):
    copy = img.copy()
    size = (np.max(target))/16
    circles = cv.HoughCircles(target, cv.HOUGH_GRADIENT, 1.5, size, param1=300, param2=25, minRadius=0,maxRadius=50)
    num = 0
    
    if circles is not None:
        num = len(circles[0])
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            cv.circle(copy, (x, y), r, (0, 0, 255), 2)
            cv.rectangle(copy, (x - 3, y - 3), (x + 3, y + 3), (0, 128, 255), -1)

    return num, copy

def main():
    imgs, files = read_imgs("../test_data")
    threshes = morph_saturation(imgs)
    finals = []
    totals = []
    for (img, thresh, file) in zip(imgs, threshes, files):
        total, final = hough_circles(img, thresh)
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
