import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  
import matplotlib.colors as clr
import numpy as np 
import cv2 as cv
import os
from scipy import ndimage as nd
import time


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

def blob_detection(img, target):
    imcopy = img.copy()

    params = cv.SimpleBlobDetector_Params()

    params.filterByColor = True
    params.blobColor = 0

    params.filterByArea = True
    params.minArea = 50

    params.filterByCircularity = True
    params.minCircularity = 0.785
    params.maxCircularity = 1.0

    params.filterByConvexity = True
    params.minConvexity = 0.4
    params.maxConvexity = 1.0

    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    detector = cv.SimpleBlobDetector_create(params)

    keypoints = detector.detect(target)
    total = len(keypoints)

    im_with_keypoints = cv.drawKeypoints (imcopy, keypoints, np.array([]), (255,0,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return total, im_with_keypoints

def main():
    imgs, files = read_imgs("../test_data")
    threshes = morph_saturation(imgs)
    finals = []
    totals = []
    for (img, thresh, file) in zip(imgs, threshes, files):
        total, final = blob_detection(img, thresh)
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
