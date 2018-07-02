#include "rpn.h"
#include "cnn.h"

typedef struct {
    float  data_w    ;//400.0
    float  data_h    ;//225.0
    float  data_scale;//0.3125
    float* cls_score; //4个输入 3个输出
    float* box_delta;
    float* cls_data;
    float* box_data;
    float* out_rois;
    float* out_scores;
    float* out_deltas;
} input_t;

typedef struct {
    //7层layer
    softmax_arg_t      softmax_rpn_arg;
    rpn_arg_t          rpn_arg;
    psroi_pool_arg_t   psroi_cls_arg;
    ave_pool_arg_t     ave_cls_arg;
    softmax_arg_t      softmax_rfcn_arg;

    psroi_pool_arg_t   psroi_box_arg;
    ave_pool_arg_t     ave_box_arg;
    
    //bottom data buffer
    float*      bottom_rpn_cls_score ;
    anchor_t*   base_anchor          ;
    float*      bottom_delta         ;
    float*      psroi_cls_bottom_data;
    float*      psroi_box_bottom_data;
    //middle top data buffer                
    float*      rpn_softmax_top_data ;
    float*      psroi_cls_top_data   ;
    float*      ave_cls_top_data     ;
    float*      psroi_box_top_data   ;
} net_t;

net_t net;

void net_prepare_memory()
{
    net.rpn_softmax_top_data  = calloc(MAX_ANCHOR_NUM * RPN_CLS_NUM, sizeof(float));
    net.psroi_cls_top_data    = calloc(OUT_CLS_NUM    * POST_NMS_NUM * POOLING_SIZE * POOLING_SIZE, sizeof(float));
    net.ave_cls_top_data      = calloc(POST_NMS_NUM   * OUT_CLS_NUM  * AVE_POOLED   * AVE_POOLED,   sizeof(float));
    net.psroi_box_top_data    = calloc(OUT_BOX_NUM    * POST_NMS_NUM * POOLING_SIZE * POOLING_SIZE, sizeof(float));

    int fd_anchor = 0;    
    net.base_anchor           = calloc(RPN_ANCHOR_CHANNEL, sizeof(anchor_t));
    if((fd_anchor             = fopen("./base_anchor.bin","rb"))   == -1) return -1;
    fread(net.base_anchor, sizeof(anchor_t), RPN_ANCHOR_CHANNEL, fd_anchor);
    close(fd_anchor);
}

void net_release_memory()
{
    free(net.rpn_softmax_top_data);
    free(net.psroi_cls_top_data);
    free(net.ave_cls_top_data);
    free(net.psroi_box_top_data);

    free(net.base_anchor);
}

void net_update_relation(input_t input)
{
    net.softmax_rpn_arg.bottom_data  = input.cls_score;
    net.softmax_rpn_arg.top_data     = net.rpn_softmax_top_data;
    net.softmax_rpn_arg.outer_num    = 1;
    net.softmax_rpn_arg.channels     = RPN_CLS_NUM;
    net.softmax_rpn_arg.inner_num    = FEATURE_SIZE(input.data_h) * FEATURE_SIZE(input.data_w) * RPN_ANCHOR_CHANNEL;

    net.rpn_arg.base_anchor          = net.base_anchor;
    net.rpn_arg.bottom_score         = net.softmax_rpn_arg.top_data;
    net.rpn_arg.bottom_delta         = input.box_delta;
    net.rpn_arg.top_rois             = (top_rois_t*)(input.out_rois);
    net.rpn_arg.feature_w            = FEATURE_SIZE(input.data_w);
    net.rpn_arg.feature_h            = FEATURE_SIZE(input.data_h);
    net.rpn_arg.anchor_c             = RPN_ANCHOR_CHANNEL;
    net.rpn_arg.input_data_w         = input.data_w;
    net.rpn_arg.input_data_h         = input.data_h;
    net.rpn_arg.input_data_scale     = input.data_scale;

    net.psroi_cls_arg.bottom_data    = input.cls_data;
    net.psroi_cls_arg.bottom_rois    = net.rpn_arg.top_rois;
    net.psroi_cls_arg.top_data       = net.psroi_cls_top_data;
    net.psroi_cls_arg.num            = 0;
    net.psroi_cls_arg.pooled_height  = POOLING_SIZE;
    net.psroi_cls_arg.pooled_width   = POOLING_SIZE;
    net.psroi_cls_arg.width          = FEATURE_SIZE(input.data_w);
    net.psroi_cls_arg.height         = FEATURE_SIZE(input.data_h);
    net.psroi_cls_arg.channels       = OUT_CLS_NUM * POOLING_SIZE * POOLING_SIZE;
    net.psroi_cls_arg.spatial_scale  = 1.0/FEAT_STRIDE;
    net.psroi_cls_arg.output_dim     = OUT_CLS_NUM;
    net.psroi_cls_arg.group_size     = POOLING_SIZE;

    net.ave_cls_arg.bottom_data      = net.psroi_cls_arg.top_data;
    net.ave_cls_arg.top_data         = net.ave_cls_top_data      ;
    net.ave_cls_arg.num              = 0                ;
    net.ave_cls_arg.channels         = OUT_CLS_NUM      ;
    net.ave_cls_arg.height           = POOLING_SIZE     ;
    net.ave_cls_arg.width            = POOLING_SIZE     ;
    net.ave_cls_arg.pooled_height    = AVE_POOLED       ;
    net.ave_cls_arg.pooled_width     = AVE_POOLED       ;
    net.ave_cls_arg.kernel_h         = POOLING_SIZE     ;
    net.ave_cls_arg.kernel_w         = POOLING_SIZE     ;
    net.ave_cls_arg.pad_h            = AVE_PAD          ;
    net.ave_cls_arg.pad_w            = AVE_PAD          ;
    net.ave_cls_arg.stride_h         = POOLING_SIZE     ;
    net.ave_cls_arg.stride_w         = POOLING_SIZE     ;

    net.softmax_rfcn_arg.bottom_data = net.ave_cls_arg.top_data;
    net.softmax_rfcn_arg.top_data    = input.out_scores;
    net.softmax_rfcn_arg.outer_num   = 0;
    net.softmax_rfcn_arg.channels    = OUT_CLS_NUM;
    net.softmax_rfcn_arg.inner_num   = 1;

    //
    net.psroi_box_arg.bottom_data    = input.box_data;
    net.psroi_box_arg.bottom_rois    = net.rpn_arg.top_rois;
    net.psroi_box_arg.top_data       = net.psroi_box_top_data;
    net.psroi_box_arg.num            = 0;
    net.psroi_box_arg.pooled_height  = POOLING_SIZE;
    net.psroi_box_arg.pooled_width   = POOLING_SIZE;
    net.psroi_box_arg.width          = FEATURE_SIZE(input.data_w);
    net.psroi_box_arg.height         = FEATURE_SIZE(input.data_h);
    net.psroi_box_arg.channels       = OUT_BOX_NUM * POOLING_SIZE * POOLING_SIZE;
    net.psroi_box_arg.spatial_scale  = 1.0/FEAT_STRIDE;
    net.psroi_box_arg.output_dim     = OUT_BOX_NUM;
    net.psroi_box_arg.group_size     = POOLING_SIZE;
    
    net.ave_box_arg.bottom_data      = net.psroi_box_arg.top_data;
    net.ave_box_arg.top_data         = input.out_deltas ;
    net.ave_box_arg.num              = 0                ;
    net.ave_box_arg.channels         = OUT_BOX_NUM      ;
    net.ave_box_arg.height           = POOLING_SIZE     ;
    net.ave_box_arg.width            = POOLING_SIZE     ;
    net.ave_box_arg.pooled_height    = AVE_POOLED       ;
    net.ave_box_arg.pooled_width     = AVE_POOLED       ;
    net.ave_box_arg.kernel_h         = POOLING_SIZE     ;
    net.ave_box_arg.kernel_w         = POOLING_SIZE     ;
    net.ave_box_arg.pad_h            = AVE_PAD          ;
    net.ave_box_arg.pad_w            = AVE_PAD          ;
    net.ave_box_arg.stride_h         = POOLING_SIZE     ;
    net.ave_box_arg.stride_w         = POOLING_SIZE     ;
}

int net_forward()
{
    softmax(&(net.softmax_rpn_arg));
    
    int num_rois = rpn(&(net.rpn_arg));
    if (num_rois >= POST_NMS_NUM) num_rois = POST_NMS_NUM;//roi的个数最多为POST_NMS_NUM
    //更新各layer roi的个数
    net.psroi_cls_arg.num           = num_rois;
    net.ave_cls_arg.num             = num_rois;
    net.softmax_rfcn_arg.outer_num  = num_rois;
    net.psroi_box_arg.num           = num_rois;    
    net.ave_box_arg.num             = num_rois;
    
    psroi_pooling_multithreading(&(net.psroi_cls_arg));
    ave_pool(&(net.ave_cls_arg));
    softmax(&(net.softmax_rfcn_arg)); 
    psroi_pooling_multithreading(&(net.psroi_box_arg));
    ave_pool(&(net.ave_box_arg));

    return num_rois;
}

