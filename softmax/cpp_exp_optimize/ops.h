#ifndef OPS_H
#define OPS_H

#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "omp.h"

#define MNN_CONCURRENCY_BEGIN(__iter__, __num__) \
_Pragma("omp parallel for") for (int __iter__ = 0; __iter__ < __num__; __iter__++) {

#define MNN_CONCURRENCY_END() }

#define ALIMIN(x, y) ((x) < (y) ? (x) : (y))
#define ALIMAX(x, y) ((x) > (y) ? (x) : (y))
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))


void argmax(const float *srcData, float *dstData, int outside, int channel, int n, int threadNum);
void softmax(const float *srcData, float *dstData, int outside, int channel, int threadNum);

#endif
