#include <string.h>
#include <iostream>
#include <vector>
#include <iostream>
#include <algorithm>

#include<cmath>
#include <arm_neon.h> 
#include "omp.h"


#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)

#define readfile(name, buffer, elem_size) do\
{\
  FILE *out = fopen(name, "rb");\
  if(out != NULL)\
  {\
        fread (buffer , sizeof(buffer[0]), elem_size, out);\
        fclose (out);\
  }\
} while(0)
    

#define MNN_CONCURRENCY_BEGIN(__iter__, __num__) \
_Pragma("omp parallel for") for (int __iter__ = 0; __iter__ < __num__; __iter__++) {
#define MNN_CONCURRENCY_END() }


#define ALIMIN(x, y) ((x) < (y) ? (x) : (y))
#define ALIMAX(x, y) ((x) > (y) ? (x) : (y))
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))

void MNNExpC8(float* dest, const float* source, const float* parameters, size_t countC8) {
    auto count = countC8 * 8;
    auto param = parameters[0];
    float xLimit = 87;
    for (int i = 0; i < count; ++i) {
        auto x         = -source[i];
        x = ALIMAX(x, -xLimit);
        x = ALIMIN(x, xLimit);
        int div        = (x * parameters[1]);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *(float*)(&div2);
        auto t = xReamin;
        auto expRemain =
            ((((parameters[7] * t + parameters[6]) * t + parameters[5]) * t + parameters[4]) * t + parameters[3]) * t +
            parameters[2];
        dest[i] = expBasic * expRemain;
    }
}

void MNNExp(float* dst, const float* src, size_t dataSize) {
    int countC8        = (int)dataSize / 8;
    if (countC8 > 0) {
        // Align to eight so asm is easier to write
        static float parameters[] = {
            (float)log(2.0f), 1.0f / (float)log(2.0f), 1.0f, 1.0f, 0.5f, 1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 120.0f};
        MNNExpC8(dst, src, parameters, countC8);
    }
    int remain = countC8 * 8;
    auto param = log(2.0f);
    float xLimit = 87;
    for (int i = remain; i < dataSize; i++) {
        /*Origin Function*/
        //dst[i] = expf(-src[i]);
        /*Approciate Function*/
        
        auto x         = -src[i];
        x = ALIMAX(x, -xLimit);
        x = ALIMIN(x, xLimit);
        
        int div        = (x / param);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *(float*)(&div2);
        
        auto t         = xReamin;
        auto expRemain = ((((1.0f / 120 * t + 1.0f / 24) * t + 1.0f / 6) * t + 0.5f) * t + 1.0f) * t + 1.0f;
        dst[i]  = expBasic * expRemain;
    }
}

std::pair<int, int> multiThreadDivide(int size, int threadNum)
{
    int sizeDivide = size / threadNum;
    sizeDivide = UP_DIV(sizeDivide, 4) * 4;
    int scheduleNumber = 1;
    if (sizeDivide > 0) {
        scheduleNumber = UP_DIV(size, sizeDivide);
    }
    return std::make_pair(sizeDivide, scheduleNumber);
}

int softmax(const float *srcData, float *dstData, int outside, int channel, int threadNum) {
    // Max and sub
    MNN_CONCURRENCY_BEGIN(tId, threadNum)
    {
        const float *srcY = srcData + tId * channel;
        float *dstY       = dstData + tId * channel;
        for (int y = (int)tId; y < outside; y += threadNum, srcY += channel * threadNum, dstY += channel * threadNum) {
            float maxValue = srcY[0];
            {
                int c = 1;
                for (; c < channel; ++c) {
                    float value = srcY[c];
                    if (value > maxValue)
                        maxValue = value;
                }
            }

            for (int c = 0; c < channel; ++c) {
                dstY[c] = -srcY[c] + maxValue;
            }
        }
    }
    MNN_CONCURRENCY_END();
    
    //Exp
    auto schedule = multiThreadDivide(channel * outside, threadNum);
    int  sizeDivide = schedule.first;
    int  scheduleNumber = schedule.second;

    MNN_CONCURRENCY_BEGIN(tId, scheduleNumber) {
        int start = sizeDivide * (int)tId;
        int realSize = sizeDivide;
        if (tId == scheduleNumber -1 ) {
            realSize = channel * outside - start;
        }
        if (realSize > 0) {
            MNNExp(dstData + start, dstData + start, realSize);
        }
    }
    MNN_CONCURRENCY_END();

    // Sum and div
    MNN_CONCURRENCY_BEGIN(tId, threadNum);
    {
        float *dstY       = dstData + tId * channel;
        for (int y = (int)tId; y < outside; y += threadNum, dstY += channel * threadNum) {
            // sum
            float sumValue = 0;

            for (int c = 0; c < channel; ++c) {
                sumValue += dstY[c];
            }

            // div
            {
                int c = 0;
                for (; c < channel; ++c) {
                    dstY[c] /= sumValue;
                }
            }
        }
    }
    MNN_CONCURRENCY_END();

    return 0;
}




int main()
{
  int threadNum = 1;
/*
  int outside   = 75;
  int channel   = 7367;
  float* data = (float*)malloc(outside * channel * sizeof(float));
  readfile("caffe_ip1_out.bin", data, outside * channel * sizeof(float));
*/
  int outside   = 1024*1024;
  int channel   = 3;
  float* data = (float*)malloc(outside * channel * sizeof(float));
  readfile("softmax.bin", data, outside * channel * sizeof(float));

  float* out = (float*)malloc(outside * channel * sizeof(float));

    double elapsed;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    softmax(data, out, outside, channel, threadNum);
    
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("elapsed time:%f out[x]:%f\r\n",elapsed, out[888888]);
    return 0;
}
