#!/bin/sh
gcc format.c -w  -lm -lpthread -O3 -ffast-math  -o format
./format
