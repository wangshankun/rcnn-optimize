# -*- coding:utf-8 -*- 
import ctypes,time
import numpy as np
np.set_printoptions(threshold=np.nan)

class psroi_pool_arg_float32(ctypes.Structure):
    _fields_ = [("bottom_data",    ctypes.POINTER(ctypes.c_float)),#float*
                ("bottom_rois",    ctypes.POINTER(ctypes.c_float)),
                ("top_data",       ctypes.POINTER(ctypes.c_float)),
                ("num_rois",       ctypes.c_int),
                ("pooled_height",  ctypes.c_int),
                ("pooled_width",   ctypes.c_int),
                ("width",          ctypes.c_int),
                ("height",         ctypes.c_int),
                ("channels",       ctypes.c_int),
                ("spatial_scale",  ctypes.c_float),
                ("output_dim",     ctypes.c_int),
                ("group_size",     ctypes.c_int)]

num_rois       =  300
pooled_height  =  7
pooled_width   =  7
width          =  64
height         =  36
channels       =  245
spatial_scale  =  0.062500
output_dim     =  5
group_size     =  7

#load元素caffe的psroi计算结果
org_np = bottom_data_np = np.load('psroipooled_cls_rois.npy')
org_np = org_np.flatten()

#申请空的float数组
#bottom_data = (ctypes.c_float * (channels * height * width))()
#bottom_rois = (ctypes.c_float * (num_rois * 5))()

#从npy文件中，获得数据
bottom_data_np = np.load('rfcn_cls.npy')
#从numpy数组中转为ctypes的float array
bottom_data    = bottom_data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

bottom_rois_np = np.load('rois.npy')
bottom_rois    = bottom_rois_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#申请float array内存接收从C的lib库中返回的结果
top_data    = (ctypes.c_float * (output_dim * num_rois *  pooled_height * pooled_width))()
#初始化结构体参数
arg  = psroi_pool_arg_float32(bottom_data, bottom_rois, top_data, num_rois,
                            pooled_height, pooled_width, width, height, channels,
                             spatial_scale, output_dim, group_size)
#加载库
libpsroi= ctypes.cdll.LoadLibrary('./libpsroi.so')

'''
libpsroi.sub_pthread_init()
c = 0
s=time.time()
while (c < 100):
    libpsroi.psroi_pooling_multithreading(arg)
    c = c + 1
print time.time() - s
libpsroi.sub_pthread_exit()
'''

libpsroi.sub_pthread_init()
c = 0
while (c < 100):#100次循环测试
    #执行psroi函数，返回结果存储在top_data中
    libpsroi.psroi_pooling_multithreading(arg)
    #将float array类型的top_data转为numpy格式，方便比较和查看
    top_np = np.frombuffer(top_data, dtype=np.float32)
    if np.array_equal(top_np, org_np) != True:
        print "Error"
    c = c + 1
    print c
libpsroi.sub_pthread_exit()
