#!/bin/sh
#https://zhuanlan.zhihu.com/p/142573902

g++  SingleInstance.cpp -o SingleInstance_11 -lpthread -std=c++0x
./SingleInstance_11
#g++  SingleInstance.cpp -o SingleInstance_c99 -lpthread -std=c++98
#./SingleInstance_c99
