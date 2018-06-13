#define  _GNU_SOURCE
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <error.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#define EPSILON 0.000001 //根据精度需要

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)


void ave_pool(void)
{
    int num            = 300;
    int channels_      = 5;
    int height_        = 7;
    int width_         = 7;
    int pooled_height_ = 1;
    int pooled_width_  = 1;
    int kernel_h_      = 7;
    int kernel_w_      = 7;
    int pad_h_         = 0;
    int pad_w_         = 0;
    int stride_h_      = 7;
    int stride_w_      = 7;

    //准备数据
    int fd_data;
    if((fd_data = fopen("./psroipooled_cls_rois.bin", "rb")) == -1)
    {
      printf("creat file wrong!");
    }
    float* bottom_data = malloc(num * channels_ * height_ * width_  * sizeof(float));
    float* top_data    = malloc(num * channels_ * sizeof(float));
    memset(top_data, 0 , num * channels_ * sizeof(float));
    fread(bottom_data, sizeof(float), num * channels_ * height_ * width_  , fd_data);
    close(fd_data);
    struct timespec start, finish;
    double elapsed = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int n = 0; n < num; ++n)
    {
        for (int c = 0; c < channels_; ++c)
        {
            for (int ph = 0; ph < pooled_height_; ++ph)
            {
                for (int pw = 0; pw < pooled_width_; ++pw)
                {
                    int hstart = ph * stride_h_ - pad_h_;
                    int wstart = pw * stride_w_ - pad_w_;
                    int hend = fminf(hstart + kernel_h_, height_ + pad_h_);
                    int wend = fminf(wstart + kernel_w_, width_ + pad_w_);
                    int pool_size = (hend - hstart) * (wend - wstart);
                    hstart = fmaxf(hstart, 0);
                    wstart = fmaxf(wstart, 0);
                    hend = fminf(hend, height_);
                    wend = fminf(wend, width_);
                    for (int h = hstart; h < hend; ++h)
                    {
                        for (int w = wstart; w < wend; ++w)
                        {
                            top_data[ph * pooled_width_ + pw] +=bottom_data[h * width_ + w];
                        }
                    }
                    top_data[ph * pooled_width_ + pw] /= pool_size;
                }
            }
            bottom_data += 49;
            top_data    += 1;
        }
    }
    bottom_data -= num * channels_ * height_ * width_ ;
    top_data    -= num * channels_ ;
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("elapsed time:%f\r\n",elapsed);
    savefile("./mcls_score.bin",top_data, num * channels_ * sizeof(float));
    free(top_data);
    free(bottom_data);
}

void main()
{
    ave_pool();
}
