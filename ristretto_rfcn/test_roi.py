#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os, sys, cv2
import argparse
import xml.etree.ElementTree as ET
from easydict import EasyDict as edict

CAFFE_PATH = os.path.join('./caffe/python')
LIB_PATH = os.path.join('./lib')
sys.path.insert(0, CAFFE_PATH)
sys.path.insert(0, LIB_PATH)

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.test import hardorck_detect 
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import caffe

class evalModels(object):
    def __init__(self, mat_fpath, test_img_dir, test_ano_dir, test_result_dir):
        self.cfg = edict()
        self.cfg.mat_fpath = mat_fpath
        self.test_img_dir = test_img_dir
        self.test_ano_dir = test_ano_dir
        self.result_data = os.path.join(test_result_dir, 'data')
        self.stat_count = {}
        self.stat_total = {}
        self.NMS_THRESH = 0.3
        self.CONF_THRESH = 0.5
        self.OVERLAP_THRESH = 0.5
        self.CLASSES = ()
        self.pred_result = []
        self.element_list = ['blue','white','yellow','new_energy']
        self.missing_image_list = []
        self.img_format = 'jpg'
        for item in self.element_list:
            self.stat_count[item] = 0
            self.stat_total[item] = 0
            
    def set_thresh(self, nms_thresh=0.3, conf_thresh=0.5, overlap_thresh=0.5):
        self.NMS_THRESH = nms_thresh
        self.CONF_THRESH = conf_thresh
        self.OVERLAP_THRESH = overlap_thresh

    def read_test_result(self, mat_fpath=None):
        if mat_fpath is None:
            mat_fpath = self.cfg.mat_fpath
        mat_file = sio.loadmat(mat_fpath)
        test_result = mat_file['pred_result'][0]
        pred_result = {}
        for i in range(len(test_result)):
            im_name = test_result[i][0, 0]['name'][0]
            tmp = test_result[i][0, 0]['result']
            tmp_type = list(tmp.dtype.names)
            single_pred_result = {}
            for j in range(len(tmp_type)):
                single_pred_result[tmp_type[j]] = tmp[tmp_type[j]][0, 0]
            pred_result[im_name] = single_pred_result
        return pred_result

    def readAnoXML(self, filefpath_xml):
        tree = ET.parse(filefpath_xml)
        root = tree.getroot()
        charlist = []
        charcoord = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            x1 = int(bndbox.find('xmin').text)
            y1 = int(bndbox.find('ymin').text)
            x2 = int(bndbox.find('xmax').text)
            y2 = int(bndbox.find('ymax').text)
            charcoord.append([x1, y1, x2, y2])
            charlist.append(obj.find('name').text)
        return charlist, charcoord

    def bbox_overlap(self, g_bbox, p_bbox):
        area1 = (g_bbox[3] - g_bbox[1]) * (g_bbox[2] - g_bbox[0])
        area2 = (p_bbox[3] - p_bbox[1]) * (p_bbox[2] - p_bbox[0])
        xx1 = np.maximum(g_bbox[0], p_bbox[0])
        yy1 = np.maximum(g_bbox[1], p_bbox[1])
        xx2 = np.minimum(g_bbox[2], p_bbox[2])
        yy2 = np.minimum(g_bbox[3], p_bbox[3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        overlap = inter * 1.0 / (area1 + area2 - inter)
        return overlap

    def eval_bbox(self, charlist, charcoord, pred_result):
        correct_result = []
        correct_classes = []
        for i in range(len(charlist)):
            char = charlist[i]
            g_bbox = charcoord[i]
            if char in pred_result:
                dets = pred_result[char]
                overarea = [self.bbox_overlap(g_bbox, det[:4]) for det in dets]
                inds = np.where(np.asarray(overarea) > self.OVERLAP_THRESH)[0]
                if len(inds) == 0:
                    correct_result.append([0,0,0,0,0])
                    correct_classes.append(' ')
                else:
                    correct_result.append(dets[inds[0]])
                    correct_classes.append(char)
            else:
                correct_result.append([0,0,0,0,0])
                correct_classes.append(' ')
        return correct_classes, correct_result

    def eval_image(self, image_name, preds):
        filefpath_xml = os.path.join(self.test_ano_dir, image_name + '.xml')
        if not os.path.exists(filefpath_xml):
            return
        charlist, charcoord = self.readAnoXML(filefpath_xml)

        im_file = os.path.join(self.test_img_dir, image_name + '.' + self.img_format)

        CONF_THRESH = self.CONF_THRESH
        NMS_THRESH = self.NMS_THRESH
        pred_result = {}
        for cls, dets in preds.iteritems():
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            dets = [dets[i, :] for i in inds]
            if not len(inds) == 0:
                pred_result[cls] = np.asarray(dets)
        tmp_charlist = []
        tmp_coords = []
        for i in range(len(charlist)):
            if charlist[i] not in ['foot','numhsq']:
                tmp_charlist.append(charlist[i])
                tmp_coords.append(charcoord[i])
        charlist = tmp_charlist
        charcoord = tmp_coords
        correct_classes, correct_coords = self.eval_bbox(charlist, charcoord, pred_result)
        if correct_classes != charlist:
            self.missing_image_list.append(image_name)
        for i in range(len(charlist)):
            self.stat_total[charlist[i]] += 1
            if not correct_classes[i] == ' ':
                self.stat_count[correct_classes[i]] += 1
    def confmatrix2precision(self):
        return True
    def confmatrix2recall(self):
        return True
    def div0(self, a, b):
        """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~ np.isfinite(c)] = 0  # -inf inf NaN
        return c

class test_ult(object):
    def __init__(self, test_prototxt, test_caffemodel, test_img_dir, test_ano_dir, test_result_dir):
        self.prototxt = test_prototxt
        self.caffemodel = test_caffemodel
        self.test_img_dir = test_img_dir
        self.test_ano_dir = test_ano_dir
        self.result_dir = test_result_dir
        self.pred_result = []

        self.element_list = ['blue','white','yellow','new_energy']
        self.img_format = 'jpg'

    def eval_image(self, net, image_name):
        """Detect object classes in an image using pre-computed object proposals."""

        # Load the demo image
        filefpath_xml = os.path.join(self.test_ano_dir, image_name[:-4] + '.xml')
        if not os.path.exists(filefpath_xml):
            return

        im_file = os.path.join(self.test_img_dir, image_name)

        img = cv2.imread(im_file)
        im = img
        # print im_file
        if not self.valid_img_size(im):
            return
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im)# for faster rcnn
        #scores, boxes = im_detect(net, im) # for rfcn
        timer.toc()
        # print image_name[:-4]
        print ('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, boxes.shape[0])

        
        print boxes.shape
        print scores.shape
        single_result_all = {}
        single_result_all["name"] = image_name[:-4]
        single_result_all["result"] = {}
        for cls_ind, cls in enumerate(self.element_list):
            cls_ind += 1  # because we skipped background
            # cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)] #for faster rcnn
            cls_boxes = boxes[:, 4 : 8] # rfcn
            #print scores.shape,cls_ind
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            # print cls, max(dets[:,-1])
            single_result_all["result"][cls] = dets
        self.pred_result.append(single_result_all)

    def hardorck_eval_image(self, net, image_name):
        filefpath_xml = os.path.join(self.test_ano_dir, image_name[:-4] + '.xml')
        if not os.path.exists(filefpath_xml):
            return

        im_file = os.path.join(self.test_img_dir, image_name)
        img = cv2.imread(im_file)
        im = img
        if not self.valid_img_size(im):
            return
        timer = Timer()
        timer.tic()
        scores, boxes = hardorck_detect(net, im)# for faster rcnn
        timer.toc()
        print ('HardRock Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, boxes.shape[0])

        single_result_all = {}
        single_result_all["name"] = image_name[:-4]
        single_result_all["result"] = {}
        for cls_ind, cls in enumerate(self.element_list):
            cls_ind += 1  # because we skipped background
            # cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)] #for faster rcnn
            cls_boxes = boxes[:, 4 : 8] # rfcn
            #print scores.shape,cls_ind
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            # print cls, max(dets[:,-1])
            single_result_all["result"][cls] = dets
        self.pred_result.append(single_result_all)

    def valid_img_size(self, img):
        w, h, ch = img.shape
        s = min(w, h)
        l = max(w, h)
        ratio = min(600.0 / s, 1000.0 / l)
        # print w, h, ratio
        if (w * ratio / 16 < 6) or (h * ratio / 16 < 6) or (w <= 16) or (h <= 16):
            return False
        return True


if __name__ == '__main__':
    test_prototxt  = './models/normal/test_zf_rfcn.prototxt'
    test_caffemodel  = './models/normal/zf_rfcn.caffemodel'

    test_img_dir = './data/roi/data/JPEGImages'
    test_ano_dir = './data/roi/data/Annotations'
    test_result_dir = "./data/"
    mat_file = "result.mat"
    mat_fpath = os.path.join(test_result_dir, mat_file)

    if not os.path.isfile(test_caffemodel):
        raise IOError(('{:s} not found.\n').format(test_caffemodel))

    cfg.TEST.HAS_RPN = True
    caffe.set_mode_cpu()
    #caffe.set_mode_gpu()
    #caffe.set_device(0)
    cfg.TEST.RPN_PRE_NMS_TOP_N = 6000
    cfg.TEST.RPN_POST_NMS_TOP_N = 300

    test_obj = test_ult(test_prototxt, test_caffemodel, test_img_dir, test_ano_dir, test_result_dir)
    timer = Timer()
    timer.tic()
    net = caffe.Net(test_prototxt, test_caffemodel, caffe.TEST)
    timer.toc()
    print '\n\nLoaded network {:s} for {:.3f}s'.format(test_caffemodel, timer.total_time)
    im_names = os.listdir(test_img_dir)
    for im_name in im_names[:1]:
        test_obj.eval_image(net, im_name)
        print im_name
    pred_result = test_obj.pred_result
    sio.savemat(os.path.join(test_result_dir, 'result.mat'), {"pred_result": pred_result})

    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    em = evalModels(mat_fpath, test_img_dir, test_ano_dir, test_result_dir)
    pred_result = em.read_test_result()
    nms_thresh = 0.3
    overlap_thresh = 0.5
    conf_thresh = 0.5
    em.set_thresh(nms_thresh=nms_thresh, conf_thresh=conf_thresh, overlap_thresh=overlap_thresh)

    image_num = 0
    for k, v in pred_result.iteritems():
        image_num = image_num + 1
        em.eval_image(image_name=k, preds=v)

    stat_count = em.stat_count
    stat_total = em.stat_total

    total_correct = 0
    total_num = 0
    for k in em.element_list:
        if k not in stat_count.keys():
            continue
        v = stat_count[k]
        if stat_total[k] != 0:
            print k, stat_count[k], stat_total[k], stat_count[k] * 1.0 / stat_total[k]
            total_correct += stat_count[k]
            total_num += stat_total[k]
    incorrect_image_num = len(em.missing_image_list)
    if total_num != 0:
        print total_correct, total_num, total_correct * 1.0 / total_num
        print "whole image stat:"
        print image_num - incorrect_image_num, image_num, (image_num - incorrect_image_num) * 1.0 / image_num
    print "Done"

