#!/bin/sh
rm -rf roi single_roi roi_position
gcc -c parallel.c -O3 -o parallel.o
gcc -c roi_pooling.c -w -ffast-math -O3 -o roi_pooling.o
gcc parallel.o roi_pooling.o -lpthread -o  roi

gcc single_roi.c -w -ffast-math -O3  -lm -o  single_roi
gcc roi_position.c -w -ffast-math -O3  -lm -o  roi_position
rm -rf *.o
