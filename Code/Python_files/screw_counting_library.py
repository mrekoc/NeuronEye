import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  
import matplotlib.colors as clr
import numpy as np 
import cv2 as cv
import os
from scipy import ndimage as nd

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

def read_values(folder_name):
	#Reads base values from a text file, returns only 
	#the values based in an alphabetical order
    values = []
    pack = np.loadtxt(folder_name,dtype={'names': ('filenames', 'values'),
                         'formats': ('S20', 'i4')})
    i = np.argsort(pack)
    pack = pack[i]
    for value in pack[['values']]:
        value = value[0]
        values.append(value)
    
    return values

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

def find_contour(img, target):
    fidelity = False
    fidelityValue = 1.7
    detected = img.copy()
    c, h = cv.findContours(target, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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

def hough_circles(img, target):
    copy = img.copy()
    size = (np.max(target))/16
    if size <= 0:
        size = 1
    circles = cv.HoughCircles(target, cv.HOUGH_GRADIENT, 1, size, param1=21, param2=6, minRadius=5,maxRadius=21)
    num = 0
    
    if circles is not None:
        num = len(circles[0])
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            cv.circle(copy, (x, y), r, (0, 0, 255), 2)
            cv.rectangle(copy, (x - 3, y - 3), (x + 3, y + 3), (0, 128, 255), -1)

    return num, copy

def f1_measure(base, total):
	#Calculates the f1 measure on a single image, returns the score
    
    tp = 0
    fp = 0
    fn = 0
    score = 0
    
    if base==total:
        tp = total
    elif total > base:
        tp = base
        fp = total - base
    elif total < base:
        tp = total
        fn = base - total
    
    if base == 0 and (fp+fn) == 0:
        score = 1
    else:
        score = tp / (tp + (0.5) * (fp + fn))
        
    return score
    
def show_results(img):
    #fig, axs = plt.subplots(1,2)
    #axs[0].imshow(img, cmap = 'gray')
    #axs[0].set_title('Original image')
    plt.imshow(final, cmap = 'gray')
    plt.set_title('Image w/ detected objects')

def example_implementation(folder, value_file):
    
    imgs, files = read_imgs(folder)
    values = read_values(value_file)
    
    threshes = morph_saturation(imgs)
    
    finals = []
    totals = []
    scores = []
    anomalies = []
    
    for (img, thresh, file, value) in zip(imgs, threshes, files, values):
        
        total, final = blob_detection(img, thresh)
        score = f1_measure(value, total)
        
        if score < 0.750:
            anomalies.append(file)
        
        scores.append(score)
        finals.append(final)
        totals.append(total)
        
        #fig, axs = plt.subplots(1,2)
        #axs[0].imshow(img, cmap = 'gray')
        #axs[0].set_title('Original image')
        #axs[1].imshow(final, cmap = 'gray')
        #axs[1].set_title('Image w/ detected objects')

        #print("Filename:" , file, "Detected objects:", total)
        
    average_success = np.sum(scores) / len(scores)
    save = zip(files, imgs, threshes, finals, values, totals, scores)
    
    return save, average_success

#folder = "../../../dataset/training"
#value_file = folder + "/bases.txt"

#save, rate = example_implementation(folder, value_file)

#for file, img, thresh, final, value, total, score in save:
#    print(file, total, value, score)

#print("\n ######## \n ######## \n Average success in this batch: %5.3f \n" % (rate))

#show_results(save[3][5])
