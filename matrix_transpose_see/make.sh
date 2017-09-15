#!/bin/sh
rm -rf transpose
gcc mxn_transport.c  -g -O3 -o transpose -w

./transpose
