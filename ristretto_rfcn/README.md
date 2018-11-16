1.先评估原始网络，生成quantized.prototxt里面有对定点层修改
./caffe/build/tools/ristretto quantize --model=./train_zf_rfcn.prototxt --weights=./rfcn/normal/gs_roi_rpn.caffemodel --model_quantized=./quantized.prototxt --iterations=100 --gpu=0 --trimming_mode=dynamic_fixed_point --error_margin=3

cpu执行的命令
./caffe/build/tools/ristretto quantize --model=./train_zf_rfcn.prototxt --weights=./rfcn/normal/gs_roi_rpn.caffemodel --model_quantized=./quantized.prototxt --iterations=100 --trimming_mode=dynamic_fixed_point --error_margin=3

降低error_margin 增加iterations次数
./caffe/build/tools/ristretto quantize --model=./train_zf_rfcn.prototxt --weights=./rfcn/normal/gs_roi_rpn.caffemodel --model_quantized=./quantized.prototxt --iterations=2000 --gpu=0 --trimming_mode=dynamic_fixed_point --error_margin=0.5

2.微调原始的weight生成新的weight
./caffe/build/tools/caffe train --solver=./solver_finetune.prototxt --weights=./rfcn/normal/gs_roi_rpn.caffemodel


环境变量：因为ristretto 需要调用python，既c++调python，因此python使用的环境变量必须导入

    export PYTHONPATH=./caffe/python:$PYTHONPATH
    
    export PYTHONPATH=./lib:$PYTHONPATH

因为安装了anaconda，与系统自带的python冲突，因此修改可执行路径环境变量，排除掉anaconda的python路径
export PATH=/usr/local/cuda-         8.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

移植注意点：
1. faster rcnn/rfcn 训练网络的输入数据是python处理的，而使用ristretto 去调用那么必须提前生成IMDB数据，
所以新写了一个数据处理的层roi_data_layer/layer_quantized.py，把训练准备数据的内容实现在数据层中

2.lib/fast_rcnn/config.py，配置文件必须与训练时候参数一样，faster rcnn训练时候经常使用外挂solver或者yml配置做真实的
训练参数配置，而不实际使用config.py，但是在ristretto 中，因为没有外挂solver和yml，必须在config.py把所以参数配置起来

3.对比
caffe代码https://github.com/BVLC/caffe
Ristretto caffe代码https://github.com/pmgysel/caffe
将ristretto实现的代码移植到faster rcnn代码中

=================================================================================
增加了对cpu版本的支持在微软caffe的基础上引入一些intel caffe layer的cpu 实现逻辑(编译选项加入c++11)
intel 作为cpu厂商会有优先支持cpu 逻辑的实现，因此如果找不到cpu的实现逻辑可以去intel的caffe找找
但是intel caffe很多不必要的实现，比如intel 自己的mkl_cdnn，mkl gemm等等···因此还是需要基于相对
干净的微软版本去实现cpu的rfcn训练。  

纯cpu版本  
cp  ./lib/setup_cpu_only.py  ./lib/setup.py                       
cp  ./lib/fast_rcnn/nms_wrapper_cpu_only.py ./lib/fast_rcnn/nms_wrapper.py   
cp  ./caffe/Makefile.config.cpu_only ./caffe/Makefile.config  
支持gpu版本  
cp  ./lib/setup_gpu_support.py  ./lib/setup.py                       
cp  ./lib/fast_rcnn/nms_wrapper_gpu_support.py ./lib/fast_rcnn/nms_wrapper.py  
cp  ./caffe/Makefile.config.gpu_support ./caffe/Makefile.config   
  
cd caffe  
make -j8  
make pycaffe  
cd ..  
cd lib  
make  
cd ..  
chmod a+x ./example.sh  
python example.sh  
