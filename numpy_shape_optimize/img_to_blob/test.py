# -*- coding:utf-8 -*-
import numpy as np
import os, cv2, time, struct
from socket import *
from ctypes import *
np.set_printoptions(threshold=np.nan)

pre_img = np.load("./pre_img.npy")
src  = pre_img.ctypes.data_as(POINTER(c_ubyte))

img_format = cdll.LoadLibrary('./img_format.so')

dest = (c_short * (pre_img.shape[0] * pre_img.shape[1] * 4))()

s=time.time()
img_format.img_to_hardrock_format(dest, src, pre_img.nbytes, 7)
my_img = np.frombuffer(dest, dtype=np.int16)
my_img = my_img.reshape((pre_img.shape[0], pre_img.shape[1], 4))
print "img_to_hardrock_format using time:",time.time() - s

def fix_to_int(data, point):
    data = data*2**point
    data = np.around(data)
    data = data.astype(np.int16)
    return data

img = pre_img
PIXEL_MEANS   = np.array([[[102.9801, 115.9465, 122.7717]]])

s=time.time()
#减mean值
img = img - PIXEL_MEANS
#定点化
img = fix_to_int(img, 7)
#补0，凑够4个channel
img = np.pad(img, ((0,0),(0,0),(0,1)), 'constant') 
print "numpy using time:",time.time() - s

#检查两种方法结果
print ((my_img == img).all())
