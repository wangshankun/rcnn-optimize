#include "config.h"

int score_thresh_int = (int)(SCORE_THRESH * PREC);

typedef struct{                                                                    
    int score;                                                                     
    int index;                                                                     
} index_score_t;

typedef struct{
    float x1, y1, x2, y2;
} anchor_t;

typedef struct{                                                                    
    float x1, y1, x2, y2;                                                                                                                       
    int score;
    int prob_area;                                                                    
} pre_rois_t;

typedef struct{                                                                    
    float score;                                                                  
    float x1, y1, x2, y2;                                                                                                                       
} top_rois_t;

typedef struct {
    anchor_t*   base_anchor;
    float*      bottom_score;
    float*      bottom_delta;
    top_rois_t* top_rois;
    int         feature_w;
    int         feature_h;
    int         anchor_c;
    float       input_data_w;
    float       input_data_h;
    float       input_data_scale;
} rpn_arg_t;

static float box_intersection(pre_rois_t a, pre_rois_t b)                                        
{                                                                                  
    float w = fminf(a.x2, b.x2) - fmaxf(a.x1, b.x1) + 1;                           
    float h = fminf(a.y2, b.y2) - fmaxf(a.y1, b.y1) + 1;                           
    //if(w < 0 || h < 0) return 0;减少一个判断节省30%时间                          
    //最终是与thresh_float对比，即使重合面积是负数不影响判断                   
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

