#!/bin/sh
rm -rf softmax
gcc softmax.c -lm -w -O3 -o softmax
./softmax

diff my_cls_prob_pre.bin cls_prob_pre.bin
