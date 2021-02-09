import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  
import matplotlib.colors as clr
import numpy as np 
import cv2 as cv
import os
import time
import screw_counting_library as sc

def main():
    folder = "../../../Opencv_Test/DataSet1"
    value_file = folder + "/bases.txt"
    
    imgs, files = sc.read_imgs(folder)
    values = sc.read_values(value_file)
    
    threshes = sc.thresh_process(imgs)
    
    finals = []
    totals = []
    scores = []
    recalls = []
    precisions = []
    
    for (img, thresh, file, value) in zip(imgs, threshes, files, values):
        
        total, final = sc.blob_detection(img, thresh)
        score, recall, precision = sc.f1_measure(value, total)
        
        finals.append(final)
        totals.append(total)
        scores.append(score)
        recalls.append(recall)
        precisions.append(precision)
        
    save = zip(files, imgs, threshes, finals, values, totals, scores, recalls, precisions)
    rate = np.sum(scores) / len (scores)
    rate_r = np.sum(recalls) / len (recalls)
    rate_p = np.sum(precisions) / len (precisions)
    
    for (file, img, thresh, final, value, total, score, recall, precision) in save:
        print("Filename: %s, Detected objects: %d, Actual value: %d, Score: %5.3f" % (file, total, value, score))
    
    print("\n ######## \n ######## \n Average success in this batch: %5.3f \n Average recall: %5.3f \n Average precision: %5.3f" % (rate, rate_r, rate_p))

if __name__ == '__main__':
    start = time.process_time()
    main()
    print(time.process_time() - start)
