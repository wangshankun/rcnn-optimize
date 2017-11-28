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

typedef struct{
    int hstart;
    int hend  ;
    int wstart;
    int wend  ;
    int is_empty;
    int pool_index;
} roi_position;

void roi(void)
{
    int   num_rois       =  300;
    int   pooled_height_ =  6;
    int   pooled_width_  =  6;
    int   width_         =  64;
    int   height_        =  25;
    int   channels_      =  256;
    float spatial_scale_ =  0.062500;
    int   in_area        = height_        * width_;
    int   out_area       = pooled_height_ * pooled_width_;
    //准备数据
    int fd_data, fd_rois;
    if((fd_data = fopen("./roi_data", "rb")) == -1)
    {
      printf("creat file wrong!");
    }
    if((fd_rois = fopen("./roi_prop", "rb")) == -1)
    {
      printf("creat file wrong!");
    }
    float* bottom_data = malloc(channels_ * in_area  * sizeof(float));
    float* bottom_rois = malloc(5         * num_rois * sizeof(float));
    float* top_data    = malloc(channels_ * num_rois * out_area * sizeof(float));
    
    roi_position* positions  = calloc(num_rois * channels_, sizeof(roi_position));
    fread(bottom_data, sizeof(float), channels_ * width_ * height_ , fd_data);
    fread(bottom_rois, sizeof(float), num_rois  * 5 , fd_rois);
    close(fd_data);
    close(fd_rois);

    struct timespec start, finish;
    double elapsed = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int n = 0;
    while(n < num_rois)
    {
        
        int roi_start_w = round(bottom_rois[n*5 + 1] * spatial_scale_);
        int roi_start_h = round(bottom_rois[n*5 + 2] * spatial_scale_);
        int roi_end_w   = round(bottom_rois[n*5 + 3] * spatial_scale_);
        int roi_end_h   = round(bottom_rois[n*5 + 4] * spatial_scale_);

        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        int roi_width  = max(roi_end_w - roi_start_w + 1, 1);
        const float bin_size_h = (float)(roi_height) / (float)(pooled_height_);
        const float bin_size_w = (float)(roi_width) / (float)(pooled_width_);

        const float* batch_data = bottom_data;

        int c, ph, pw, h, w;
        for (c = 0; c < channels_; ++c)
        {
            for (ph = 0; ph < pooled_height_; ++ph)
            {
                for (pw = 0; pw < pooled_width_; ++pw)
                {
                    int hstart = (int)(floor((float)(ph)    * bin_size_h));
                    int wstart = (int)(floor((float)(pw)    * bin_size_w));
                    int hend   = (int)(ceil((float)(ph + 1) * bin_size_h));
                    int wend   = (int)(ceil((float)(pw + 1) * bin_size_w));

                    hstart = min(max(hstart + roi_start_h, 0), height_);
                    hend   = min(max(hend   + roi_start_h, 0), height_);
                    wstart = min(max(wstart + roi_start_w, 0), width_);
                    wend   = min(max(wend   + roi_start_w, 0), width_);

                    bool is_empty = (hend <= hstart) || (wend <= wstart);

                    const int pre_pool_index = n * channels_ * out_area + c * out_area;
                    const int pool_index     = pre_pool_index + ph * pooled_width_ + pw;

                    positions[n * channels_ + c].hstart = hstart;
                    positions[n * channels_ + c].hend   = hend;
                    positions[n * channels_ + c].wstart = wstart;
                    positions[n * channels_ + c].wend   = wend;
                    positions[n * channels_ + c].is_empty = is_empty;
                    positions[n * channels_ + c].pool_index = pool_index;
                }
            }
        }
        
        n++;
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("single roi positions elapsed time:%f\r\n",elapsed);
    free(bottom_data);
    free(bottom_rois);
    free(top_data);
    free(positions);
}

void main()
{
    roi();
}
