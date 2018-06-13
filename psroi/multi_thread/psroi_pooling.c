#include "psroi_pooling.h"
#include <sys/time.h> // for clock_gettime()

pthread_mutex_t counter_lock;
static int counter = -1;
static __inline unsigned int get_counter(void)
{
    pthread_mutex_lock(&counter_lock);
    counter++;
    pthread_mutex_unlock(&counter_lock);
    return (unsigned int)counter;
}

void psroi_pool_inner_thread(void* set_args, int pos)
{
    psroi_pool_arg_t* args       = (psroi_pool_arg_t*)set_args;
    const float* bottom_data     = args -> bottom_data;
    const float* bottom_rois     = args -> bottom_rois;
    float*       top_data        = args -> top_data;
    int          num_rois        = args -> num_rois;
    int          pooled_height   = args -> pooled_height;
    int          pooled_width    = args -> pooled_width;
    int          width           = args -> width;
    int          height          = args -> height;
    int          channels        = args -> channels;
    float        spatial_scale   = args -> spatial_scale;
    int          output_dim      = args -> output_dim;
    int          group_size      = args -> group_size;

    while (1)
    {
        int n = get_counter();
        if(n >= num_rois) break;

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
}

void psroi_pooling_multithreading()
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
    //初始化时间戳
    double psroipool_elapsed_time = 0;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int test_count = 0;
    for (test_count = 0; test_count < 100; test_count++)//测试100次
    {
        //初始化多线程参数
        queue_t                    queue_Q[MAX_CPU_NUMBER];
        psroi_pool_arg_t           ins_args;
        ins_args.bottom_data    =  (float*)(bottom_data);
        ins_args.bottom_rois    =  (float*)(bottom_rois);
        ins_args.top_data       =  (float*)(top_data);
        ins_args.num_rois       =  num_rois;
        ins_args.pooled_height  =  pooled_height;
        ins_args.pooled_width   =  pooled_width;
        ins_args.width          =  width;
        ins_args.height         =  height;
        ins_args.channels       =  channels;
        ins_args.spatial_scale  =  spatial_scale;
        ins_args.output_dim     =  output_dim;
        ins_args.group_size     =  group_size;
        counter = -1;
        //为每个cpu上的线程分配好自己的参数
        int i;
        for (i = 0; i < MAX_CPU_NUMBER; i++)
        {
            queue_Q[i].routine     = psroi_pool_inner_thread;
            queue_Q[i].position    = i;
            queue_Q[i].args        = &ins_args;
        }
        all_sub_pthread_exec(queue_Q, MAX_CPU_NUMBER);
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    psroipool_elapsed_time += (finish.tv_sec - start.tv_sec);
    psroipool_elapsed_time += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("multithreading psroi test 100 times elapsed time:%f \r\n", psroipool_elapsed_time);
    savefile("ctop.bin", top_data, 294000);
    free(bottom_data);
    free(bottom_rois);
    free(top_data);
}

void main()
{
    sub_pthread_init();
    psroi_pooling_multithreading();
    sub_pthread_exit();
}
