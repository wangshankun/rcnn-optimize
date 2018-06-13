import numpy as np
np.set_printoptions(threshold=np.nan)
cls_score     = np.load('cls_score.npy')
cls_score.tofile('cls_score.bin')

bc =  np.fromfile('cls_score.bin',dtype=np.float32)

mbc = np.fromfile('mcls_score.bin',dtype=np.float32)


print bc
print "====================================================="
print mbc


