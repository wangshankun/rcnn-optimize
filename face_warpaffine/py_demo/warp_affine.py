# _*_ coding:utf-8 _*_
import os
import cv2
import numpy
import numpy as np

imgSize2 = [112,112]
coord5point2 = [[38.2946, 51.6963],
               [77.5318, 51.6963],
               [56.0252, 71.7366],
               [41.5493, 92.3655],
               [70.7299, 92.3655]]

x = np.array(coord5point2)
x = x * 246.0/112.0
print(x)
imgSize2 = [246,246]
coord5point2 = [[ 84.11135357 ,113.54723036],
 [170.29306071 ,113.54723036],
 [123.05535    ,157.56431786],
 [ 91.26006964 ,202.87422321],
 [155.35317321 ,202.87422321]]

#96x112
src = np.array([
  [30.2946, 51.6963],
  [65.5318, 51.5014],
  [48.0252, 71.7366],
  [33.5493, 92.3655],
  [62.7299, 92.2041] ], dtype=np.float32 )

src[:,0]= src[:,0] +  8.0
print(src)

def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),numpy.matrix([0., 0., 1.])])
 
def warp_im(img_im, orgi_landmarks, tar_landmarks):
    pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    print(M)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    return dst


test_landmarks = [  [74.432381, 106.324806],
                    [137.880966, 81.760826],
                    [108.673759, 137.702362],
                    [102.738121, 176.072586],
                    [162.386627, 159.218475],
                 ]

test = cv2.imread("crop_img.jpg",cv2.IMREAD_GRAYSCALE)
dst_img = warp_im(test, test_landmarks, coord5point2)
cv2.imwrite("dst_img_python.jpg",dst_img)
