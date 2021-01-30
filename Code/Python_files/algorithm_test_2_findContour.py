import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  
import matplotlib.colors as clr
import numpy as np 
import cv2 as cv
import os
import time
import screw_counting_library as sc

def main():
    folder = "../../../dataset/images"
    value_file = folder + "/bases.txt"
    
    imgs, files = sc.read_imgs(folder)
    values = sc.read_values(value_file)
    
    threshes = sc.morph_saturation(imgs)
    
    finals = []
    totals = []
    scores = []
    
    for (img, thresh, file, value) in zip(imgs, threshes, files, values):
        
        total, final = sc.find_contour(img, thresh)
        score = sc.f1_measure(value, total)
        
        finals.append(final)
        totals.append(total)
        scores.append(score)
        
    save = zip(files, imgs, threshes, finals, values, totals, scores)
    rate = np.sum(scores) / len (scores)
    
    for (file, img, thresh, final, value, total, score) in save:
        print("Filename: %s, Detected objects: %d, Actual value: %d, Score: %5.3f" % (file, total, value, score))
    
    print("\n ######## \n ######## \n Average success in this batch: %5.3f \n" % (rate))

if __name__ == '__main__':
    start = time.process_time()
    main()
    print(time.process_time() - start)
