import os
import cv2
import sys
import time
import copy
import random
import thread
import numpy as np

org_img   = cv2.imread("C:/Users/lenovo/Desktop/t.jpg")
gray      = cv2.cvtColor(org_img,cv2.COLOR_BGR2GRAY)
roi_height,roi_width = gray.shape
pooled_width = 256
pooled_height = 256
outputs = np.zeros((pooled_width, pooled_height))
bin_size_w = float(roi_width) / float(pooled_width)
bin_size_h = float(roi_height) / float(pooled_height)

print roi_height,roi_width,

for ph in range(pooled_height):
    hstart = int(np.floor(ph * bin_size_h))
    hend = int(np.ceil((ph + 1) * bin_size_h))
    for pw in range(pooled_width):
        wstart = int(np.floor(pw * bin_size_w))
        wend = int(np.ceil((pw + 1) * bin_size_w))
        is_empty = (hend <= hstart) or(wend <= wstart)
        if is_empty:
            outputs[ph, pw] = 0
        else:
            t_ar = gray[hstart:hend, wstart:wend]
            t_ar = t_ar.flatten()
            outputs[ph, pw] = t_ar[np.argmax(t_ar)]

cv2.imwrite("C:/Users/lenovo/Desktop/p.jpg", outputs)
