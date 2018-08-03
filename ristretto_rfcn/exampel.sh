#!/bin/sh
#1.先评估原始网络，生成quantized.prototxt里面有对定点层修改
./caffe/build/tools/ristretto quantize --model=./models/normal/train_zf_rfcn.prototxt --weights=./models/normal/zf_rfcn.caffemodel --model_quantized=./models/ristretto/quantized.prototxt --iterations=2000 --gpu=0 --trimming_mode=dynamic_fixed_point --error_margin=0.5

#2.微调原始的weight生成新的weight
./caffe/build/tools/caffe train --solver=./odels/ristretto/solver_finetune.prototxt --weights=./models/normal/zf_rfcn.caffemodel 


#环境变量：因为ristretto 需要调用python，既c++调python，因此python使用的环境变量必须导入
#    export PYTHONPATH=./caffe/python:$PYTHONPATH
#    export PYTHONPATH=./lib:$PYTHONPATH
