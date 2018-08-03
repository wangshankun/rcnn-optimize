#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import matplotlib.pyplot as savefig
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

import errno
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5 (except OSError, exc: for Python <2.5)
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


CLASSES = ('__background__', # always index 0
           'zero','one','two','three','four','five','six','seven','eight','nine')

CLASSES = ('__background__', # always index 0
           'zero','one','two','three','four','five','six','seven','eight','nine')

CLASSES = ('__background__', # always index 0
           'zero','one','two','three','four','five','six','seven','eight','nine')

CLASSES = ['__background__','blue','white','yellow','new_energy']

CLASSES =['__background__', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's',
                                      't', 'u', 'v', 'w', 'x', 'y', 'z', 'sh', 'js', 'ha', 'ah', 'zj', 'bj', 'tj', 'cq',
                                      'he', 'yn', 'ln', 'hl', 'hn', 'sd', 'sc', 'jx', 'hb', 'gs',
                                      'sx', 'sn', 'jl', 'fj', 'gz', 'gd', 'gx', 'qh', 'hi', 'nx', 'xz', 'nm', 'xj',
                                      'xue', 'gua', 'jing', 'shi', 'ling', 'jun']
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),

        'alexnet': ('AlexNet',
                  'AlexNet_faster_rcnn_final.caffemodel'),
        
        'vgg_1024': ('VGG_CNN_M_1024',
                           'INRIA_Person_faster_rcnn_final.caffemodel'),
        'darknet': ('darknet',
                           'tiny-yolo.caffemodel'),

        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')

        }

# convert float to fixed
def float2fixed(mat_float,n_bits):
   lshift = 1 << n_bits
   fixed = np.round(mat_float * lshift)*1.0/lshift
   return fixed

def savefixed2model(net,layer,n_bits):
    weight = net.params[layer][0].data
    bias = net.params[layer][1].data
    weight = float2fixed(weight,n_bits)
    bias = float2fixed(bias,n_bits)
    net.params[layer][0].data[...] = weight
    net.params[layer][1].data[...] = bias
    net.save('/home/westwell/Documents/py-faster-rcnn/output/container_digits/alexnet_quantize/container_digits_train/AlexNet_quantize_faster_rcnn_final.caffemodel') 

def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    
    #plt.draw()
    #plt.show()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    # im_file = os.path.join('/home/westwell/Documents/py-faster-rcnn/data/INRIA_Person_devkit/data/Images',image_name)
    im_file = os.path.join('data/demo',image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    #print 'debug0>>>---------------this is tools/container_digits_test.py/im_detect--------' 
    #print 'debug0>>>cfg.TEST.HAS_RPN',cfg.TEST.HAS_RPN
    scores, boxes = im_detect(net, im)
    #print 'debug >>>score shape',scores.shape
    sum = np.sum(scores,axis=1)
    #print 'debug >>>sum shape',sum.shape
    #print 'debug >>>boxes shape',boxes.shape
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.59
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        #cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets,ax,thresh=CONF_THRESH)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    caffemodel = '/home/cnn/py-R-FCN/rfcn/char/gs_plate_char_global_float_iter_100000.caffemodel'
    prototxt   = '/home/cnn/py-R-FCN/rfcn/char/test_agnostic_global.prototxt'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    im_names = ['10.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)
    plt.show()
    
