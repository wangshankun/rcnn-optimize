#!/bin/sh
g++ qsmt.cpp -lpthread --fast-math -O3 -o qsmt
g++ qsmg.cpp -lpthread --fast-math -O3 -o qsmg
gcc topk.c --fast-math -O3 -o topk

./qsmt
./qsmg
./topk
