#!/bin/sh
python test.py --src_model shufflenet-v2-10.onnx --hd_type=t31 --inputs "input;1;1;3;224;224"
#python test.py --src_model ssd_mobilenet_v1_10.onnx --hd_type=t31 --inputs "image_tensor:0;1;1;224;224;3"




