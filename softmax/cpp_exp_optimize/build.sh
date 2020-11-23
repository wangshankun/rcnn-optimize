#!/bin/sh
c++  -std=c++11 -O3 -fopenmp -fPIC -std=gnu++11 ops.cpp  main.cpp MNNExpC8.S  -o test

#c++  -std=c++11 -g -fopenmp -fPIC -std=gnu++11 ops.cpp  main.cpp MNNExpC8.S  -o test

