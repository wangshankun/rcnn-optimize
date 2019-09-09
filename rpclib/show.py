import numpy as np
import cv2

result = np.fromfile("res.bin",dtype=np.float32)
result = result.reshape(1024,1024)
result = np.around(result) * 255
result = result.astype(np.uint8)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

