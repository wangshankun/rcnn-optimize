#!/bin/sh
rm -rf GFLOPS dot gemm_cmp_huge_vector
gcc -msse4 -mavx -O3 -ffast-math GFLOPS.c -o GFLOPS
g++ -O3 -fopenmp -mavx -ffast-math dot.cpp -o dot
gcc -std=c99 -w -msse4 -mavx -fopenmp -ffast-math -O3 gemm_cmp_huge_vector.c -o gemm_cmp_huge_vector

