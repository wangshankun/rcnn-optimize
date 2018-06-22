#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <alloca.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/cdefs.h>

int    feature_w = 26;
int    feature_h = 15;
int    anchor_c  = 21;//(3*7 == 21),3是anchor的3类scale,7是anchor的7个长宽比

float  input_data_w     = 400;
float  input_data_h     = 225;
float  input_data_scale = 0.3125;

int  pre_nms_top        = 6000;
int  post_nms_top       = 300;
float iou_thresh        = 0.7;
float  rpn_min_size     = 4;


#define SCORE_THRESH   0.5
#define PREC           1000000
int     score_thresh_int   = (int)(SCORE_THRESH * PREC);
typedef struct{                                                                    
    int score;                                                                     
    int index;                                                                     
} index_score_t;

typedef struct{
    float x1, y1, x2, y2;
} anchor_t;

typedef struct{
    float x, y, w, h;
} delta_t;

typedef struct{                                                                    
    float x1, y1, x2, y2;                                                                                                                       
    int score;
    int prob_area;                                                                    
} pre_rois_t;

typedef struct{                                                                    
    float x1, y1, x2, y2;                                                                                                                       
    float score;                                                                  
} top_rois_t;

static float box_intersection(pre_rois_t a, pre_rois_t b)                                        
{                                                                                  
    float w = fminf(a.x2, b.x2) - fmaxf(a.x1, b.x1) + 1;                           
    float h = fminf(a.y2, b.y2) - fmaxf(a.y1, b.y1) + 1;                           
    //if(w < 0 || h < 0) return 0;减少一个判断节省30%时间                          
    //㮠为最终是与thresh_float对比，即使重合面积是负数不影响判断                   
    return w*h;                                                                        
}                                                                                      
                                                                                       
static float box_iou(pre_rois_t a, pre_rois_t b)                                                     
{                                                                                      
    float inter = box_intersection(a, b);                                              
    return (float)((float)inter/(a.prob_area + b.prob_area - inter));                  
}                                                                                      
                                                                                                                                                                          
int score_reversed(const void *pa, const void *pb)                                     
{                                                                                      
   return (((index_score_t*)pb)->score - ((index_score_t*)pa)->score);                 
}

