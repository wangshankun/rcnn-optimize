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
#include <float.h>

#define max(a, b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
     
#define min(a, b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _b : _a; })

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)


void softmax()
{
    int outer_num_   = 300;
    int channels     = 5;
    int inner_num_   = 1;
    int dim          = channels * inner_num_;
    float* in_data   = calloc(outer_num_ * dim, sizeof(float));
    float* out_data  = calloc(outer_num_ * dim, sizeof(float));
    float* max_data  = calloc(inner_num_, sizeof(float));
    float* sum_data  = calloc(inner_num_, sizeof(float));

    //准备数据
    int fd_data;
    if((fd_data = fopen("./cls_score.bin", "rb")) == -1)
    {
        printf("creat file wrong!\r\n");
    }
    fread(in_data, sizeof(float), outer_num_ * dim , fd_data);
    close(fd_data);

    //初始化时间戳
    double softmax_elapsed_time = 0;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int test_count = 0;
    for (test_count = 0; test_count < 100; test_count++)//测试100次
    {
    //计算softmax
    register unsigned int n, c, k;
    memcpy(out_data, in_data, outer_num_ * dim * sizeof(float));
    for (n = 0; n < outer_num_; n++)
    {
        for (k = 0; k < inner_num_; k++)
        {
            max_data[k] = -FLT_MAX;
        }  
        
        register unsigned int out_index = n * dim;
        for (c = 0; c < channels; c++)
        {
            for (k = 0; k < inner_num_; k++)
            {
                max_data[k] = max(max_data[k], in_data[out_index + c * inner_num_ + k]);
            }
        }

        memset(sum_data, 0x0, inner_num_ * sizeof(float));
        for (c = 0; c < channels; c++)
        {
            for (k = 0; k < inner_num_; k++)
            {
                register unsigned int inner_index  = out_index + c * inner_num_ + k;
                out_data[inner_index]              = out_data[inner_index] - max_data[k];
                out_data[inner_index]              = (float)exp(out_data[inner_index]);
                sum_data[k]                        = sum_data[k] + out_data[inner_index];
            }
        }
        
        for (c = 0; c < channels; c++)
        {
            for (k = 0; k < inner_num_; k++)
            {
                register unsigned int inner_index = out_index + c * inner_num_ + k;
                out_data[inner_index]             = out_data[inner_index]/sum_data[k];
            }
        }    
    }
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    softmax_elapsed_time += (finish.tv_sec - start.tv_sec);
    softmax_elapsed_time += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("softmax test 100 times elapsed time:%f \r\n", softmax_elapsed_time);
    savefile("my_cls_prob_pre.bin", out_data, outer_num_ * dim * sizeof(float));
    free(in_data);
    free(out_data);
    free(max_data);
    free(sum_data);
}

void main()
{
    softmax();
}
