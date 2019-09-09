#!/bin/sh
g++ -w cli.cpp -o cli -std=gnu++11 -L/usr/local/lib/ -L/usr/lib/ -L/usr/lib/aarch64-linux-gnu -lrpc -lpthread -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui  

g++  -w ser.cpp -o ser -std=gnu++11 -I"/usr/local/cuda/include" -I"/usr/include" -I"../include" -I"../../include" -D_REENTRANT -L"/usr/local/cuda/lib64" -L"/usr/local/lib/" -L"/usr/lib/" -L"/usr/lib/aarch64-linux-gnu" -L"../lib" -L"../../lib" -L../../bin -Wl,--start-group -lrpc -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser -lcudnn -lcublas -lcudart  -lrt -ldl -lpthread -Wl,--end-group

