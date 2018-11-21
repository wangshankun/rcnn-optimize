#!/bin/sh
g++ qsmt.cpp -lpthread --fast-math -O3 -o qsmt
g++ qsmg.cpp -lpthread --fast-math -O3 -o qsmg
./qsmt
./qsmg

