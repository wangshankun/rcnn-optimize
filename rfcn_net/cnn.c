#include "cnn.h"
#include <sys/time.h> // for clock_gettime()

pthread_spinlock_t g_spin;
/*在Linux 4.4.77(38) SMP aarch64中发现mutex会在首次执行时候大概率性的失效；
  因此换为spinlock，本库多线程小于等于cpu个数，且仅仅锁住一个自的加操作，采用
  spinlock会更合适更高效
*/
//pthread_mutex_t counter_lock;
volatile int counter = -1;
static __inline int get_counter(void)
{
    //pthread_mutex_lock(&counter_lock);
    pthread_spin_lock(&g_spin);
    counter++;
    //pthread_mutex_unlock(&counter_lock);
    pthread_spin_unlock(&g_spin);
    return counter;
}

void psroi_pool_inner_thread(void* set_args, int pos)
{
    psroi_pool_arg_t* args       = (psroi_pool_arg_t*)set_args;
    float*       bottom_data     = args -> bottom_data;
    float*       bottom_rois     = args -> bottom_rois;
    float*       top_data        = args -> top_data;
    int          num             = args -> num;
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
        if(n >= num) break;
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


void psroi_pooling_multithreading(psroi_pool_arg_t* arg)
{
    pthread_spin_init(&g_spin, 0);

    //初始化多线程参数
    queue_t                    queue_Q[MAX_CPU_NUMBER];
    psroi_pool_arg_t           ins_args;
    ins_args.bottom_data    =  arg->bottom_data;
    ins_args.bottom_rois    =  arg->bottom_rois;
    ins_args.top_data       =  arg->top_data;
    ins_args.num            =  arg->num;
    ins_args.pooled_height  =  arg->pooled_height;
    ins_args.pooled_width   =  arg->pooled_width;
    ins_args.width          =  arg->width;
    ins_args.height         =  arg->height;
    ins_args.channels       =  arg->channels;
    ins_args.spatial_scale  =  arg->spatial_scale;
    ins_args.output_dim     =  arg->output_dim;
    ins_args.group_size     =  arg->group_size;
    counter = -1;
    //为每个cpu上的线程分配好自己的参数
    int i;
    for (i = 0; i < MAX_CPU_NUMBER; i++)
    {
        queue_Q[i].routine     = psroi_pool_inner_thread;
        queue_Q[i].position    = i;
        queue_Q[i].args        = &ins_args;
    }
    //执行所有线程
    all_sub_pthread_exec(queue_Q, MAX_CPU_NUMBER);
    pthread_spin_destroy(&g_spin);
}


void ave_pool(ave_pool_arg_t* arg)
{
    int num            = arg->num;
    int channels       = arg->channels;
    int height         = arg->height;
    int width          = arg->width;
    int pooled_height  = arg->pooled_height;
    int pooled_width   = arg->pooled_width;
    int kernel_h       = arg->kernel_h;
    int kernel_w       = arg->kernel_w;
    int pad_h          = arg->pad_h;
    int pad_w          = arg->pad_w;
    int stride_h       = arg->stride_h;
    int stride_w       = arg->stride_w;
    float* bottom_data = arg->bottom_data;
    float* top_data    = arg->top_data;
    memset(top_data, 0 , num * channels * sizeof(float));

    register int bottom_offset = height * height;
    register int top_offset    = pooled_height * pooled_width;

    for (int n = 0; n < num; ++n)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int ph = 0; ph < pooled_height; ++ph)
            {
                for (int pw = 0; pw < pooled_width; ++pw)
                {
                    int hstart = ph * stride_h - pad_h;
                    int wstart = pw * stride_w - pad_w;
                    int hend = fminf(hstart + kernel_h, height + pad_h);
                    int wend = fminf(wstart + kernel_w, width + pad_w);
                    int pool_size = (hend - hstart) * (wend - wstart);
                    hstart = fmaxf(hstart, 0);
                    wstart = fmaxf(wstart, 0);
                    hend = fminf(hend, height);
                    wend = fminf(wend, width);
                    for (int h = hstart; h < hend; ++h)
                    {
                        for (int w = wstart; w < wend; ++w)
                        {
                            top_data[ph * pooled_width + pw] +=bottom_data[h * width + w];
                        }
                    }
                    top_data[ph * pooled_width + pw] /= pool_size;
                }
            }
            bottom_data += bottom_offset;
            top_data    += top_offset;
        }
    }

    //恢复到指针的初始位置
    //bottom_data -= num * channels * height * width ;
    //top_data    -= num * channels ;
}

//N,C,H,W
//N = outer_num; C = channels; H*W = inner_num
void softmax(softmax_arg_t* arg)
{
    int outer_num      = arg->outer_num;
    int channels       = arg->channels;
    int inner_num      = arg->inner_num;
    float* bottom_data = arg->bottom_data;
    float* top_data    = arg->top_data;
    float* max_data    = calloc(inner_num, sizeof(float));
    float* sum_data    = calloc(inner_num, sizeof(float));
    register int dim   = channels * inner_num;
    //计算softmax
    register unsigned int n, c, k;
    memcpy(top_data, bottom_data, outer_num * dim * sizeof(float));//将bottom内容复制到top中,后续计算用到
    for (n = 0; n < outer_num; n++)
    {
        for (k = 0; k < inner_num; k++)
        {
            max_data[k] = FLT_MIN;//max_data预设置最小值,后面有比较大小时候用
        }  
                                                                          /* N    C  inner_num     */
        register unsigned int out_index = n * dim;                        /* 1 x  3 x 4 * max_data */
        for (c = 0; c < channels; c++)//按照channel遍历出inner_num个最大值/*  4  -1   7 *   7      */
        {                                                                 /*  3   9  -6 *   9      */
            for (k = 0; k < inner_num; k++)                               /* -2   1   0 *   1      */
            {                                                             /*  7   3  -2 *   7      */
                max_data[k] = fmaxf(max_data[k], bottom_data[out_index + c * inner_num + k]);
            }
        }

        memset(sum_data, 0x0, inner_num * sizeof(float));
        for (c = 0; c < channels; c++)
        {
            for (k = 0; k < inner_num; k++)
            {
                register unsigned int inner_index  = out_index + c * inner_num + k;
                top_data[inner_index]              = top_data[inner_index] - max_data[k];
                top_data[inner_index]              = (float)exp(top_data[inner_index]);
                sum_data[k]                        = sum_data[k] + top_data[inner_index];
            }
        }
        
        for (c = 0; c < channels; c++)
        {
            for (k = 0; k < inner_num; k++)
            {
                register unsigned int inner_index = out_index + c * inner_num + k;
                top_data[inner_index]             = top_data[inner_index]/sum_data[k];
            }
        }    
    }
    free(max_data);
    free(sum_data);
}
