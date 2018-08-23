# -*- coding:utf-8 -*-
import numpy as np
import os,  time, struct
from socket import *
from ctypes import *
from after_rfcn import *
from cStringIO import StringIO

np.set_printoptions(threshold=np.nan)

def fix_to_int(data, point):
    data = data*2**point
    data = np.around(data)
    data = data.astype(np.int16)
    return data

def int_to_float(data, point):
    data = data.astype(np.float32)
    data = data/(2**point)
    return data

def to_caffe_format(output_np, c, c_org, h , w, w_org):
    output_np   = output_np.reshape((int(c/4), h, w, 4))
    output_np   = output_np.transpose((0,3,1,2))
    output_np   = output_np.reshape((1, c, h, w))
    output_np   = output_np[:,0:c_org,:,0:w_org]
    output_np   = int_to_float(output_np,7)
    return output_np

def img_2_blob(img_org):
    target_size   = 360 
    cfg_max_size  = 480
    PIXEL_MEANS   = np.array([[[102.9801, 115.9465, 122.7717]]])

    im_shape = img_org.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    data_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(data_scale * im_size_max) > cfg_max_size:
        data_scale = float(cfg_max_size) / float(im_size_max)

    img = cv2.resize(img_org, None, None, fx=data_scale, fy=data_scale,
                    interpolation=cv2.INTER_LINEAR)

    img = img - PIXEL_MEANS

    img = fix_to_int(img, 7)

    img = np.pad(img, ((0,0),(0,(img.shape[1])%8),(0,1)), 'constant')  
    
    return img, data_scale 


CLASSES = ('__background__','truck')
           
POST_NMS_NUM = 300#最多300个roi结果
OUT_CLS_NUM  = 2#两类
OUT_BOX_NUM  = 8#8个坐标
AVE_POOLED   = 7#psroi投票框为7*7

im_info    = np.load("./im_info.npy")
im_info    = im_info.reshape(3,-1)
data_h     = im_info[0]
data_w     = im_info[1]
data_scale = im_info[2]
             
out_rois   = (c_float * (POST_NMS_NUM * 5))()
out_scores = (c_float * (POST_NMS_NUM * OUT_CLS_NUM))()
out_deltas = (c_float * (POST_NMS_NUM * OUT_BOX_NUM * AVE_POOLED * AVE_POOLED))()

result     = np.zeros((30, 6),np.float32)
    
cnn = cdll.LoadLibrary('./libnet.so')
cnn.sub_pthreads_setup()
cnn.net_prepare_memory("./base_anchor.bin")

cls_score_np = np.load('rpn_cls_score.npy')
cls_score    = cls_score_np.ctypes.data_as(POINTER(c_float))

box_delta_np = np.load('rpn_bbox_pred.npy')
box_delta    = box_delta_np.ctypes.data_as(POINTER(c_float))

cls_data_np  = np.load('rfcn_cls.npy')
cls_data     = cls_data_np.ctypes.data_as(POINTER(c_float))

box_data_np  = np.load('rfcn_bbox.npy')
box_data     = box_data_np.ctypes.data_as(POINTER(c_float))

im_info      = np.load("./im_info.npy")
im_info      = im_info.reshape(3,-1)
             

input = input_t(data_w, data_h, data_scale, cls_score, box_delta, cls_data, box_data, 
                    out_rois, out_scores, out_deltas)
                
s = time.time()

cnn.net_update_relation(input)

num_rois = cnn.net_forward()

rois   = np.frombuffer(out_rois,   dtype=np.float32)
scores = np.frombuffer(out_scores, dtype=np.float32)
deltas = np.frombuffer(out_deltas, dtype=np.float32)

#后续处理
rois   = rois[0:5 * num_rois]
rois   = rois.reshape(num_rois, 5)

scores = scores[0:OUT_CLS_NUM * num_rois]
scores = scores.reshape(num_rois, OUT_CLS_NUM)

box_deltas = deltas[0:OUT_BOX_NUM * num_rois]
box_deltas = box_deltas.reshape(num_rois, OUT_BOX_NUM)

keep = []
for index in range(0, scores.shape[0]):
    if scores[index][0] < 0.5:#保留背景小于0.5的
        keep.append(index)

rois       = rois[keep]
scores     = scores[keep]
box_deltas = box_deltas[keep]
boxes      = rois[:, 1:5] / data_scale


im_h = data_h/data_scale#获得原始图像的长宽
im_w = data_w/data_scale
pred_boxes = bbox_transform_inv(boxes, box_deltas)
pred_boxes = clip_boxes(pred_boxes, (im_h, im_w))



CONF_THRESH = 0.8
NMS_THRESH  = 0.3
t_ix = 0
for cls_ind, cls in enumerate(CLASSES[1:]):
    cls_ind += 1 # because we skipped background
    cls_boxes = pred_boxes[:, 4 : 8]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)

    if len(dets) == 0:
        continue

    keep = py_cpu_nms(dets, NMS_THRESH, CONF_THRESH)
    dets = dets[keep, :]

    if len(dets) == 0:
        continue

    for x in dets:
        #bbox  = x[:4]
        #score = x[-1]
        result[t_ix,:4] = x[:4]
        result[t_ix,4]  = x[-1]
        result[t_ix,5]  = cls_ind
        t_ix = t_ix + 1

#print result

print time.time() - s


