#!/bin/sh
gcc -w -O3 -fPIC -shared -lm -lpthread   *.c -o libpsroi.so

python pypsroi.py
