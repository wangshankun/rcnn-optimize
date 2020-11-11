import numpy as np


x = np.random.random((75, 8000))
x = x.astype(np.float32)
x.tofile('prob.bin')

x = np.random.random((1024*1024, 3))
x = x.astype(np.float32)
x.tofile('softmax.bin')
