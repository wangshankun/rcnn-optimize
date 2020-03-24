#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os, sys, cv2
sys.path.insert(0, '/root/caffe/python')
sys.path.insert(0, 'lib')
import caffe
import numpy as np
import time

#np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5, suppress=True)
#np.set_printoptions(threshold=np.nan, formatter={'all':lambda x: '%8.4f'%x})

caffe.set_mode_cpu()

net = caffe.Net('blstm.prototxt', 'blstm.caffemodel', caffe.TEST)
input_lstm  = np.zeros((175,1,384), dtype=np.float32)
input_cont  = np.ones((175,1), dtype=np.float32)
#input_cont[0][0] = 0.0;
net.blobs['data'].data[...] = input_lstm
net.blobs['cont'].data[...] = input_cont
#安装顺序取出cudnnblstm的weight
cudnnblstm_weight = np.fromfile('cudnnblstm_weight.bin',dtype=np.float32)
fw = cudnnblstm_weight[0:768*384]
fh = cudnnblstm_weight[768*384:768*384 + 768*192] 
bw = cudnnblstm_weight[768*384 + 768*192:768*384 + 768*192 + 768*384] 
bh = cudnnblstm_weight[768*384 + 768*192 + 768*384:768*384 + 768*192 + 768*384 +  768*192] 
fwb = cudnnblstm_weight[768*384 + 768*192 + 768*384 + 768*192:768*384 + 768*192 + 768*384 + 768*192 + 768]
fhb = cudnnblstm_weight[768*384 + 768*192 + 768*384 + 768*192 + 768:768*384 + 768*192 + 768*384 + 768*192 + 768*2]
bwb = cudnnblstm_weight[768*384 + 768*192 + 768*384 + 768*192 + 768*2:768*384 + 768*192 + 768*384 + 768*192 + 768*3]
bhb = cudnnblstm_weight[768*384 + 768*192 + 768*384 + 768*192 + 768*3:768*384 + 768*192 + 768*384 + 768*192 + 768*4]

#四个门的顺序对齐
blobs = [fw,fh,bw,bh,fwb,fhb,bwb,bhb]
for idx in range(8):
    i_, f_, c_, o_ = np.split(blobs[idx], 4, axis=0)
    blobs[idx] = np.expand_dims(np.concatenate([i_, f_, o_, c_], axis=0), axis=0)
fw,fh,bw,bh,fwb,fhb,bwb,bhb = blobs
#反向lstm的权重 reverse 下
bw  = bw[::-1]
bh  = bh[::-1]
bwb = bwb[::-1]
bhb = bhb[::-1]

net.params['lstm1'][0].data[...] = fw.reshape((768,384))
#B=Bih+Bhh
net.params['lstm1'][1].data[...] = fwb.reshape((768,)) + fhb.reshape((768,))
net.params['lstm1'][2].data[...] = fh.reshape((768,192))
net.params['lstm2'][0].data[...] = bw.reshape((768,384))
#B=Bih+Bhh
net.params['lstm2'][1].data[...] = bwb.reshape((768,)) + bhb.reshape((768,))
net.params['lstm2'][2].data[...] = bh.reshape((768,192))

out = net.forward()

print(net.blobs['merge_lstm_rlstm'].data.shape)
print(net.blobs['merge_lstm_rlstm'].data)

net.save('blstm.caffemodel')
'''
blstm_out = np.fromfile('0_blstm_output.bin',dtype=np.float32)
np.savetxt("blstm_0_out.txt", blstm_out,"%.4f",'\r\n')
np.savetxt("caffe_blstm_0_out.txt", net.blobs['merge_lstm_rlstm'].data.flatten(),"%.4f",'\r\n')
'''
