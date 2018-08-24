#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>

#define A  102.9801
#define B  115.9465
#define C  122.7717

typedef struct sub_thread_para
{
    short*         dest_offset;
    unsigned char* src_offset;
    int            process_num;
    int            fix_point;

} sub_thread_para_t;


void * im_format(void *arg)
{    
    short*         dest_offset  = ((sub_thread_para_t *)(arg))->dest_offset;
    unsigned char* src_offset   = ((sub_thread_para_t *)(arg))->src_offset;
    int process_num             = ((sub_thread_para_t *)(arg))->process_num;
    int fix_point               = ((sub_thread_para_t *)(arg))->fix_point;

    if (process_num%3 !=0)
    {
        printf("%s  %d error int process_num \r\n",__FUNCTION__,__LINE__);
        return NULL;
    }
    
    int cyc_times = process_num/3;
    
    short scale   = (short)pow(2, fix_point);

    int  i = 0;
    for (i = 0; i < cyc_times; i++)
    {
        dest_offset[i * 4 + 0] = (short)roundf((src_offset[i * 3 + 0] - A) * scale);
        dest_offset[i * 4 + 1] = (short)roundf((src_offset[i * 3 + 1] - B) * scale);
        dest_offset[i * 4 + 2] = (short)roundf((src_offset[i * 3 + 2] - C) * scale);
        dest_offset[i * 4 + 3] = 0;
    }
    

    return NULL;
}


int img_to_hardrock_format(short* dest, unsigned char* src, int src_bytes, int fix_point)
{
    int total_num = src_bytes/sizeof(unsigned char);
    if (total_num%3 != 0)
    {
        printf("need rgb img \r\n");
        return -1; 
    }
    
    int single_channel_num = total_num/3;
    
    int process_num0 = (single_channel_num/3)*3;
    int process_num1 = (single_channel_num/3)*3;
    int process_num2 = total_num - process_num0 - process_num1;
    
    pthread_t t0;
    pthread_t t1;
    pthread_t t2;
    
    sub_thread_para_t para0;
    sub_thread_para_t para1;
    sub_thread_para_t para2;
    
    para0.dest_offset  = dest;
    para0.src_offset   = src;
    para0.process_num  = process_num0;
    para0.fix_point    = fix_point;
    
    if(pthread_create(&t0, NULL, im_format, &para0) == -1){
        printf("fail to create pthread t0 \r\n");
        return -1;
    }

    para1.dest_offset  = para0.dest_offset + para0.process_num + para0.process_num/3;
    para1.src_offset   = para0.src_offset  + para0.process_num;
    para1.process_num  = process_num1;
    para1.fix_point    = fix_point;
    
    if(pthread_create(&t1, NULL, im_format, &para1) == -1){
        printf("fail to create pthread t1 \r\n");
        return -1;
    }

    para2.dest_offset  = para1.dest_offset + para1.process_num + para1.process_num/3;
    para2.src_offset   = para1.src_offset  + para1.process_num;
    para2.process_num  = process_num2;
    para2.fix_point    = fix_point;
    
    if(pthread_create(&t2, NULL, im_format, &para2) == -1){
        printf("fail to create pthread t2 \r\n");
        return -1;
    }

    void * result;
    if(pthread_join(t0, &result) == -1){
        printf("fail to recollect t0 \r\n");
        return -1;
    }

    if(pthread_join(t1, &result) == -1){
        printf("fail to recollect t1 \r\n");
        return -1;
    }

    if(pthread_join(t2, &result) == -1){
        printf("fail to recollect t1 \r\n");
        return -1;
    }
    
    return 0;
}
