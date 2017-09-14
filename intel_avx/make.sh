#!/bin/sh
rm -rf GFLOPS dot huge_vector
gcc -msse4 -mavx -O3 GFLOPS.c -o GFLOPS
g++ -O3 -fopenmp -mavx -ffast-math dot.cpp -o dot
gcc -w -msse4 -mavx -O3 huge_vector.c -o huge_vector

