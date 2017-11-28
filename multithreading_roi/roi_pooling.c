#include "roi_pooling.h"
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

void roi_pool_inner_thread(void* set_args, int pos)
{
    roi_pool_arg_t* args        = (roi_pool_arg_t*)set_args;
    const float* bottom_data    = args -> bottom_data;
    const float* bottom_rois    = args -> bottom_rois;
    float*       top_data       = args -> top_data;
    int          num_rois       = args -> num_rois;
    int          pooled_height_ = args -> pooled_height_;
    int          pooled_width_  = args -> pooled_width_;
    int          width_         = args -> width_;
    int          height_        = args -> height_;
    int          channels_      = args -> channels_;
    float        spatial_scale_ = args -> spatial_scale_;

    int in_area  = height_        * width_;
    int out_area = pooled_height_ * pooled_width_;

    while (1)
    {
        int n = get_counter();
        if(n >= num_rois) break;

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
                    if (is_empty) 
                    {
                        top_data[pool_index]    = 0;
                    }

                    register int   max_index  = 0;
                    register float out_data   = 0;
                    for (h = hstart; h < hend; ++h)
                    {
                        for (w = wstart; w < wend; ++w)
                        {
                            const int pre_index = c * in_area;
                            const int index     = pre_index + h * width_ + w;
                            if (batch_data[index] > out_data)
                            {
                                out_data          = batch_data[index];
                                max_index         = index;
                            }
                        }
                    }
                    top_data[pool_index]    = out_data;
                }
            }
        }
    }
}

void roi_pooling_multithreading()
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
    fread(bottom_data, sizeof(float), channels_ * width_ * height_ , fd_data);
    fread(bottom_rois, sizeof(float), num_rois  * 5 , fd_rois);
    close(fd_data);
    close(fd_rois);
    //初始化时间戳
    double roipool_elapsed_time = 0;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int test_count = 0;
    for (test_count = 0; test_count < 100; test_count++)//测试100次
    {
        //初始化多线程参数
        queue_t                    queue_Q[MAX_CPU_NUMBER];
        roi_pool_arg_t             ins_args;
        ins_args.bottom_data    =  (float*)(bottom_data);
        ins_args.bottom_rois    =  (float*)(bottom_rois);
        ins_args.top_data       =  (float*)(top_data);
        ins_args.num_rois       =  num_rois;
        ins_args.pooled_height_ =  pooled_height_;
        ins_args.pooled_width_  =  pooled_width_;
        ins_args.width_         =  width_;
        ins_args.height_        =  height_;
        ins_args.channels_      =  channels_;
        ins_args.spatial_scale_ =  spatial_scale_;
        counter = -1;
        //为每个cpu上的线程分配好自己的参数
        int i;
        for (i = 0; i < MAX_CPU_NUMBER; i++)
        {
            queue_Q[i].routine     = roi_pool_inner_thread;
            queue_Q[i].position    = i;
            queue_Q[i].args        = &ins_args;
        }
        all_sub_pthread_exec(queue_Q, MAX_CPU_NUMBER);
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    roipool_elapsed_time += (finish.tv_sec - start.tv_sec);
    roipool_elapsed_time += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("roipool_elapsed_time:%f test count :%d \r\n", roipool_elapsed_time, test_count);
    //savefile("my_roi_out", top_data, channels_ * num_rois * out_area * sizeof(float));
    free(bottom_data);
    free(bottom_rois);
    free(top_data);
}

void main()
{
    sub_pthread_init();
    roi_pooling_multithreading();
    sub_pthread_exit();
}
