#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<unistd.h>

#define  max_height 599
#define  max_width  876
#define  anchor_num 19656
#define  pre_nms_top 6000
#define  post_nms_top 300

float thresh_float = 0.7;
float min_size = 1.496;

typedef struct{
    float x1, y1, x2, y2;
} anchor;

typedef struct{
    float x, y, w, h;
} delta;

typedef struct{
    float x1, y1, x2, y2;
    float x, y, w, h;
    float score;
} delta_box;

typedef struct{
    float x1, y1, x2, y2;
    float score, prob_area;
} box;

static float box_intersection(box a, box b)
{    
    float w = fminf(a.x2, b.x2) - fmaxf(a.x1, b.x1) + 1;
    float h = fminf(a.y2, b.y2) - fmaxf(a.y1, b.y1) + 1;
    //if(w < 0 || h < 0) return 0;减少一个判断节省30%时间，这种情况下证明是没有交集的
    //因为最终是与thresh_float对比，即使重合面积是负数不影响判断,因为负数肯定小于0，比较大小足够了，没必要非得用0
    return w*h;
}

static float box_iou(box a, box b)
{
    float inter = box_intersection(a, b);
    return (float)((float)inter/(a.prob_area + b.prob_area - inter));
}

int nms_comparator(const void *pa, const void *pb)
{
    return ((((delta_box*)pa)->score - ((delta_box*)pb)->score > 0)? -1 : 1);
}

int rpn_proposal()
{
    delta_box* delta_boxs  = calloc(anchor_num, sizeof(delta_box));
    box*       pre_boxs    = calloc(pre_nms_top,    sizeof(box));
    anchor*    top_data    = calloc(post_nms_top, sizeof(anchor));
    float*   top_score   = calloc(post_nms_top, sizeof(float));
    
    anchor*  anchor_buf    = calloc(anchor_num, sizeof(anchor));
    delta*   delta_buf     = calloc(anchor_num, sizeof(delta));
    float* score_buf     = calloc(anchor_num, sizeof(float));  
    int fd_anchor, fd_delta, fd_score;

    if((fd_anchor = fopen("./anchors","rb")) ==-1)
    {
        printf("anchors creat file wrong!");
    }    
    if((fd_delta = fopen("./deltas","rb")) ==-1)
    {
        printf("deltas creat file wrong!");
    }
    if((fd_score = fopen("./scores","rb")) ==-1)
    {
        printf("scores creat file wrong!");
    }
    
    printf("fd_anchor read size:%d \r\n", fread(anchor_buf, sizeof(anchor),  anchor_num,  fd_anchor));
    printf("fd_delta read size:%d  \r\n", fread(delta_buf,  sizeof(delta),   anchor_num,  fd_delta));    
    printf("fd_score read size:%d  \r\n", fread(score_buf,  sizeof(float), anchor_num,  fd_score));
   

    close(fd_delta);
    close(fd_score);
    close(fd_anchor);
    
    int i, j, k = 0, sc = 0, test_count = 0;
    struct timespec start, finish;
    double elapsed = 0;


    float widths, heights, ctr_x, ctr_y, dx, dy, dw, dh;
    float pred_ctr_x, pred_ctr_y, pred_w, pred_h, x1, x2, y1, y2, tmp_w, tmp_h;

    clock_gettime(CLOCK_REALTIME, &start);
    while(test_count < 100)//测试100次
    {

        for(i = 0; i < anchor_num; i++)
        {
            delta_boxs[i].x1 = anchor_buf[i].x1;
            delta_boxs[i].y1 = anchor_buf[i].y1;
            delta_boxs[i].x2 = anchor_buf[i].x2;
            delta_boxs[i].y2 = anchor_buf[i].y2;
            delta_boxs[i].x  = delta_buf[i].x;
            delta_boxs[i].y  = delta_buf[i].y;
            delta_boxs[i].w  = delta_buf[i].w;
            delta_boxs[i].h  = delta_buf[i].h;
            delta_boxs[i].score = score_buf[i];
    //        printf("%f %f %f %f %f %f %f %f %f\r\n",delta_boxs[i].x1, delta_boxs[i].y1, delta_boxs[i].x2, delta_boxs[i].y2, \
                    delta_boxs[i].x,delta_boxs[i].y,delta_boxs[i].w,delta_boxs[i].h,delta_boxs[i].score);
        }
        qsort(delta_boxs, anchor_num, sizeof(delta_box), nms_comparator);

        for(i = 0; i < anchor_num; i++)
        {
            widths  = delta_boxs[i].x2 - delta_boxs[i].x1 + 1;
            heights = delta_boxs[i].y2 - delta_boxs[i].y1 + 1;
            ctr_x   = delta_boxs[i].x1 + 0.5 * widths;
            ctr_y   = delta_boxs[i].y1 + 0.5 * heights;
            
            dx = delta_boxs[i].x;
            dy = delta_boxs[i].y;
            dw = delta_boxs[i].w;
            dh = delta_boxs[i].h;

            pred_ctr_x = dx * widths + ctr_x;
            pred_ctr_y = dy * heights + ctr_y;
            pred_w     = exp(dw) * widths;
            pred_h     = exp(dh) * heights;

            x1 = pred_ctr_x - 0.5 * pred_w;
            x1 = x1<max_width? x1: max_width;
            x1 = x1>0? x1: 0;
            x2 = pred_ctr_x + 0.5 * pred_w;
            x2 = x2<max_width? x2: max_width;
            x2 = x2>0? x2: 0;
            tmp_w = x2 - x1 + 1;

            y1 = pred_ctr_y - 0.5 * pred_h;
            y1 = y1<max_height? y1: max_height;
            y1 = y1>0? y1: 0;
            y2 = pred_ctr_y + 0.5 * pred_h;
            y2 = y2<max_height? y2: max_height;
            y2 = y2>0? y2: 0;
            tmp_h = y2 - y1 + 1;
            
            if((tmp_w >= min_size) & (tmp_h >= min_size))
            {
                if(k < pre_nms_top)
                {
                    pre_boxs[k].x1 = x1;
                    pre_boxs[k].x2 = x2;
                    pre_boxs[k].y1 = y1;
                    pre_boxs[k].y2 = y2;
                    pre_boxs[k].score = delta_boxs[i].score;
                    pre_boxs[k].prob_area = tmp_w * tmp_h;
                    //printf("k:%d i:%d  %f %f %f %f %f\r\n",k,i,pre_boxs[k].x1, pre_boxs[k].y1, pre_boxs[k].x2, pre_boxs[k].y2, pre_boxs[k].score);
                    k++;
                }
                else
                {   
                    k  = 0;
                    break;
                }
            }
        }
       

        sc = 0;
        for(i = 0; i < pre_nms_top; i++)
        {
            if(pre_boxs[i].prob_area == 0) continue;
            for(j = i+1; j < pre_nms_top; j++)
            {
                if (box_iou(pre_boxs[i], pre_boxs[j]) > thresh_float)
                {
                    pre_boxs[j].prob_area = 0;
                }
            }
            sc++;
            top_data[sc].x1 = pre_boxs[i].x1;
            top_data[sc].y1 = pre_boxs[i].y1;
            top_data[sc].x2 = pre_boxs[i].x2;
            top_data[sc].y2 = pre_boxs[i].y2;
            top_score[sc] = pre_boxs[i].score;
//            printf("sc:%d i:%d %f %f %f %f %f\r\n",sc,i,pre_boxs[i].x1, pre_boxs[i].y1, pre_boxs[i].x2, pre_boxs[i].y2, pre_boxs[i].score);
            if(sc == post_nms_top) break;
  //          if(pre_boxs[i].score < 0.6) break;
        }
        test_count++;
    }
    
        clock_gettime(CLOCK_REALTIME, &finish);
        elapsed += (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("elapsed time:%f test count:%d \r\n",elapsed, test_count);
    free(anchor_buf);
    free(delta_buf);
    free(score_buf);
    free(pre_boxs);
    free(delta_boxs);
        
    return sc;
}

void main()
{
   printf("######get prob count:%d\r\n",rpn_proposal()); 
}
