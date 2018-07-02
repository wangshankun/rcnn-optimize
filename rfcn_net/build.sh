#!/bin/sh
rm -rf *.so
gcc -c -w -O3 -fPIC  parallel.c cnn.c 
gcc -c -w -O3 -fPIC -ffast-math rpn.c
cc -o libcnn.so -shared *.o
rm -rf *.o

gcc net.c -w -O3 -fPIC -shared ./libcnn.so -lm -lpthread -o libnet.so 

python main.py

