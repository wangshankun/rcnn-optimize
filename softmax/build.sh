#!/bin/sh
rm -rf softmax
#gcc softmax.c -w -lm -O3 -o softmax
gcc softmax.c -lm -w -O3 -o softmax
