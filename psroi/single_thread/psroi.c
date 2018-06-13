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


void psroi(void)
{
    int   num_rois       =  300;
    int   pooled_height  =  7;
    int   pooled_width   =  7;
    int   width          =  64;
    int   height         =  36;
    int   channels       =  245;
    float spatial_scale  =  0.062500;
    int   output_dim     =  5;
    int   group_size     =  7;
    int   in_area        = height        * width;
    int   out_area       = pooled_height * pooled_width;
    //准备数据
    int fd_data, fd_rois;
    if((fd_data = fopen("./rfcn_cls.bin", "rb")) == -1)
    {
      printf("creat file wrong!");
    }
    if((fd_rois = fopen("./rois.bin", "rb")) == -1)
    {
      printf("creat file wrong!");
    }
    float* bottom_data = malloc(channels   * in_area  * sizeof(float));
    float* bottom_rois = malloc(5          * num_rois * sizeof(float));
    float* top_data    = malloc(output_dim * num_rois * out_area * sizeof(float));
    fread(bottom_data, sizeof(float), channels * width * height , fd_data);
    fread(bottom_rois, sizeof(float), num_rois * 5 , fd_rois);
    close(fd_data);
    close(fd_rois);
    struct timespec start, finish;
    double elapsed = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int n = 0;
    for (n = 0; n < num_rois; ++n)
    {
        int   roi_add = n * 5;
        float roi_start_w = (float)(round(bottom_rois[roi_add + 1]) * spatial_scale);
        float roi_start_h = (float)(round(bottom_rois[roi_add + 2]) * spatial_scale);
        float roi_end_w   = (float)(round(bottom_rois[roi_add + 3] + 1.0) * spatial_scale);
        float roi_end_h   = (float)(round(bottom_rois[roi_add + 4] + 1.0) * spatial_scale);

        float roi_height = fmaxf(roi_end_h - roi_start_h, 0.1);
        float roi_width  = fmaxf(roi_end_w - roi_start_w, 0.1);
        float bin_size_h = (float)(roi_height) / (float)(pooled_height);
        float bin_size_w = (float)(roi_width) / (float)(pooled_width);

        float* batch_data = bottom_data;

        int ctop, ph, pw, h, w;
        for (ctop = 0; ctop < output_dim; ++ctop)
        {
            for (ph = 0; ph < pooled_height; ++ph)
            {
                for (pw = 0; pw < pooled_width; ++pw)
                {
                    int index  = n * output_dim * pooled_height * pooled_width + \ 
                                    ctop * pooled_height * pooled_width + ph * pooled_width + pw;

                    int hstart = (int)(floor((float)(ph)    * bin_size_h + roi_start_h));
                    int wstart = (int)(floor((float)(pw)    * bin_size_w + roi_start_w));
                    int hend   = (int)(ceil((float)(ph + 1) * bin_size_h + roi_start_h));
                    int wend   = (int)(ceil((float)(pw + 1) * bin_size_w + roi_start_w));
                    hstart = fminf(fmaxf(hstart, 0), height);
                    hend   = fminf(fmaxf(hend  , 0), height);
                    wstart = fminf(fmaxf(wstart, 0), width);
                    wend   = fminf(fmaxf(wend  , 0), width);

                    bool is_empty = (hend <= hstart) || (wend <= wstart);
                    int gw = pw;
                    int gh = ph;
                    int c = (ctop*group_size + gh)*group_size + gw;
                    float out_sum = 0;
                    for (h = hstart; h < hend; ++h)
                    {
                        for (w = wstart; w < wend; ++w)
                        {
                            int bottom_index = h * width + w;
                            out_sum += bottom_data[c * height * width + bottom_index];
                        }
                    }
                    float bin_area = (hend - hstart) * (wend - wstart);
                    if (is_empty)
                    {
                        top_data[index] = 0;
                    }
                    else
                    {
                        top_data[index] = out_sum/bin_area;
                    }
                    //mapping_channel[index] = c;
                }
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("elapsed time:%f\r\n",elapsed);
    free(bottom_data);
    free(bottom_rois);
    savefile("./ctop.bin",top_data,294000);
    free(top_data);
}

void main()
{
    psroi();
}
