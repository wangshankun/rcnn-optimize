#!/bin/sh
g++ argmax.cpp  -std=c++11 -O3 -fopenmp -o argmax

g++ softmax.cpp -std=c++11 -O3 -fopenmp -o softmax

#/opt/rh/devtoolset-7/root/usr/bin/c++ \
# -std=c++11 -D__STRICT_ANSI__ -O3 -fopenmp  -fvisibility-inlines-hidden -fvisibility=hidden -fomit-frame-pointer \
# -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions   \
# -fPIC -std=gnu++11 sf.cpp -o sf

c++  -std=c++11 -O3 -fopenmp -fPIC -std=gnu++11 sf.cpp -o sf

python test.py

argmax && softmax && sf
