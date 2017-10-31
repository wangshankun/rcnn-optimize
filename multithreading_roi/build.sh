#!/bin/sh
rm -rf roi
gcc -c parallel.c -O3 -o parallel.o
gcc -c roi_pooling.c -w -ffast-math -O3 -o roi_pooling.o
gcc parallel.o roi_pooling.o -lpthread -o  roi
rm -rf *.o
