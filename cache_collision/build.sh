#!/bin/sh
gcc -O3 cache_collision_511.c -o  test511.bin
gcc -O3 cache_collision_512.c -o  test512.bin
gcc -O3 cache_collision_513.c -o  test513.bin

./test511.bin && ./test512.bin && ./test513.bin
