# -*- coding:utf-8 -*- 
from ctypes import *
import time
import numpy as np
np.set_printoptions(threshold=np.nan)

CLASSES = ('__background__',
           'a', 'b', 'c', 'd',
           'e', 'f', 'g', 'h')
           
POST_NMS_NUM = 300
OUT_CLS_NUM  = 9
OUT_BOX_NUM  = 8
AVE_POOLED   = 7

def py_cpu_nms(dets, nms_thresh, con_thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    keep = []
    if np.max(scores) < con_thresh:#最大得分都不够thresh那么全部舍弃
        return keep
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    while order.size > 0:
        i = order[0]
        if scores[i] < con_thresh:#因为已经排序,只有后面得分小于thresh,后面就不需要继续
            break
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    return keep


def bbox_transform_inv(boxes, deltas):
    deltas = np.squeeze(deltas) ######ADD
    boxes = np.squeeze(boxes) ######ADD
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

class input_t(Structure):
    _fields_ = [("data_w",      c_float),
                ("data_h",      c_float),
                ("data_scale",  c_float),
                ("cls_score",   POINTER(c_float)),
                ("box_delta",   POINTER(c_float)),
                ("cls_data",    POINTER(c_float)),
                ("box_data",    POINTER(c_float)),
                ("out_rois",    POINTER(c_float)),
                ("out_scores",  POINTER(c_float)),
                ("out_deltas",  POINTER(c_float))]

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
             
out_rois     = (c_float * (POST_NMS_NUM * 5))()
out_scores   = (c_float * (POST_NMS_NUM * OUT_CLS_NUM))()
out_deltas   = (c_float * (POST_NMS_NUM * OUT_BOX_NUM * AVE_POOLED * AVE_POOLED))()

data_w = im_info[1]#400.0
data_h = im_info[0]#225.0
data_scale = im_info[2]#0.3125

input = input_t(data_w, data_h, data_scale, cls_score, box_delta, cls_data, box_data, 
                    out_rois, out_scores, out_deltas)

cnn = cdll.LoadLibrary('./libnet.so')
cnn.sub_pthreads_setup()

cnn.net_prepare_memory()

cnn.net_update_relation(input)

num_rois = cnn.net_forward()
rois   = np.frombuffer(out_rois,   dtype=np.float32)
scores = np.frombuffer(out_scores, dtype=np.float32)
deltas = np.frombuffer(out_deltas, dtype=np.float32)
cnn.net_release_memory()

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
boxes      = rois[:, 1:5] / im_info[2]


im_h = im_info[0]/im_info[2]#获得原始图像的长宽
im_w = im_info[1]/im_info[2]
pred_boxes = bbox_transform_inv(boxes, box_deltas)
pred_boxes = clip_boxes(pred_boxes, (im_h, im_w))


CONF_THRESH = 0.8
NMS_THRESH  = 0.3
for cls_ind, cls in enumerate(CLASSES[1:]):
    cls_ind += 1 # because we skipped background
    cls_boxes = pred_boxes[:, 4 : 8]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep = py_cpu_nms(dets, NMS_THRESH, CONF_THRESH)
    dets = dets[keep, :]
    for x in dets:
        bbox  = x[:4]
        score = x[-1]
        print bbox, score, CLASSES[cls_ind]

