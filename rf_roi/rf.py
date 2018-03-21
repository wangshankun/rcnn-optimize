# -*- coding: utf-8 -*-
import cv2
import numpy as np
#roi图片，就想要找的的图片
roi = cv2.imread('roi.jpg')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

#目标搜索图片
target = cv2.imread('tar.jpg')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

#计算目标直方图
roihist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
#归一化，参数为原图像和输出图像，归一化后值全部在2到255范围
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)




#卷积连接分散的点
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
dst = cv2.filter2D(dst,-1,disc)

ret,thresh = cv2.threshold(dst,50,255,0)
#使用merge变成通道图像
thresh = cv2.merge((thresh,thresh,thresh))

#蒙板
res = cv2.bitwise_and(target,thresh)
#矩阵按列合并,就是把target,thresh和res三个图片横着拼在一起
res = np.hstack((target,thresh,res))

cv2.imwrite('res.jpg',res)
#显示图像
cv2.imshow('1',res)
cv2.waitKey(0)
