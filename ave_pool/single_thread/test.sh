#!/bin/sh

gcc ave_pool.c -w -O3 -lm -o ave_pool
./ave_pool

diff  cls_score.bin mcls_score.bin
