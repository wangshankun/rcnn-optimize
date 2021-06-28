#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2

path = "stu_quan_imgs_500" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称
num = 0
for file in files: #遍历文件夹
     f = path + "/" + file
     if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (56, 56))
        img = (img-127.5)*0.0078125
        img = img.astype(np.float32)
        img.tofile(str(num) + ".bin")
        num = num + 1
 
