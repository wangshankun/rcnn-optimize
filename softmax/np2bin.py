import numpy as np
np.set_printoptions(threshold=np.nan)
cls_score     = np.load('cls_score.npy')
cls_score.tofile('cls_score.bin')

cls_prob_pre     = np.load('cls_prob_pre.npy')
cls_prob_pre.tofile('cls_prob_pre.bin')

