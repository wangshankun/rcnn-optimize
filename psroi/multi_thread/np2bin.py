import numpy as np
np.set_printoptions(threshold=np.nan)

ctop = np.fromfile('ctop.bin',dtype=np.float32)
ctop_org = np.fromfile('psroipooled_cls_rois.bin',dtype=np.float32)

print ctop
print "===================================================================="
print ctop_org
