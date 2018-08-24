#!/bin/sh
gcc img_format.c -w -O3 -fPIC  -ffast-math -shared -lm -lpthread -o img_format.so

python  test.py
