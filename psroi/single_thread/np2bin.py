import numpy as np
np.set_printoptions(threshold=np.nan)
rois     = np.load('./rois.npy')
rois.tofile('./rois.bin')
rois_bin = np.fromfile('rois.bin',dtype=np.float32)
rfcn_cls = np.load('./rfcn_cls.npy')
rfcn_cls.tofile('./rfcn_cls.bin')
rfcn_cls_bin = np.fromfile('rfcn_cls.bin',dtype=np.float32)

cls_rois     = np.load('./psroipooled_cls_rois.npy')
cls_rois.tofile('./psroipooled_cls_rois.bin')

ctop = np.fromfile('ctop.bin',dtype=np.float32)
ctop_org = np.fromfile('psroipooled_cls_rois.bin',dtype=np.float32)
