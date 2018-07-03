#include "rpn.h"
int rpn(rpn_arg_t *arg)
{
    anchor_t*   base_anchor      = arg->base_anchor;
    float*      bottom_score     = arg->bottom_score;
    float*      bottom_delta     = arg->bottom_delta;
    top_rois_t* top_rois         = arg->top_rois;  
    int         feature_w        = arg->feature_w;
    int         feature_h        = arg->feature_h;
    int         anchor_c         = arg->anchor_c;
    float       input_data_w     = arg->input_data_w;
    float       input_data_h     = arg->input_data_h;
    float       input_data_scale = arg->input_data_scale;

    int   feature_area = feature_w * feature_h;
    int   anchor_num   = feature_w * feature_h * anchor_c;
    float min_size     = RPN_MIN_SIZE * input_data_scale;

    //仅获取前景得分,安caffe默认格式(1,A,H,W),只取后一半数据是我们需要的前景得分
    bottom_score  = bottom_score + anchor_num;

    pre_rois_t* pre_rois   = calloc(PRE_NMS_NUM,  sizeof(pre_rois_t));
    index_score_t* s2index = calloc(anchor_num, sizeof(index_score_t));
    
    int i = 0, select_num = 0, pre_roi_num = 0;
    for(i = 0; i < anchor_num; i++)
    {
        if(bottom_score[i] < SCORE_THRESH)//筛掉分数不合格的anchor,省大部分计算量
        {
            continue;
        }
        //score放大指定精度倍数(float32类型是10的6次方小数)后强转为整形,快排中有速度优势
        s2index[select_num].score = (int)(bottom_score[i] * PREC);
        s2index[select_num].index = i;
        select_num                = select_num + 1;
    }
    
    //排序从高到底，起始地址为s2index，只排序前select_num个元素
    qsort(s2index, select_num, sizeof(index_score_t), score_reversed);

    float anchor_w, anchor_h, anchor_ctr_x, anchor_ctr_y;
    float pred_ctr_x, pred_ctr_y, pred_w, pred_h, x1, x2, y1, y2, tmp_w, tmp_h;

//    printf("select_num:%d\r\n",select_num);
    for(i = 0; i < select_num; i++)           
    {
        //让score index 与 顺序排列的anchor的index对应起来
        int raw_score_index    =  s2index[i].index;
        int anchor_in_x        =  raw_score_index/feature_area;
        int area_in_x          =  raw_score_index%feature_area;
        int total_anchor_index =  area_in_x * anchor_c + anchor_in_x;//得到anchor index

        int point_anchor_offset  = total_anchor_index%anchor_c;//得到属于某点n个(此例子是21个)anchor第几个
        int point_anchor_index   = total_anchor_index/anchor_c;//依据anchor_index得到在feature中的点的index
        int point_anchor_y       = point_anchor_index/feature_w;//将这个index转换为feature对应的x，y坐标
        int point_anchor_x       = point_anchor_index%feature_w;
        
        //根据x,y,offset得知此这个anchor如何是根据base_anchor如何平移获得
        float anchor_x1 = base_anchor[point_anchor_offset].x1 + FEAT_STRIDE * point_anchor_x;//FEAT_STRIDE是anchor的步长                               
        float anchor_y1 = base_anchor[point_anchor_offset].y1 + FEAT_STRIDE * point_anchor_y;                         
        float anchor_x2 = base_anchor[point_anchor_offset].x2 + FEAT_STRIDE * point_anchor_x;                         
        float anchor_y2 = base_anchor[point_anchor_offset].y2 + FEAT_STRIDE * point_anchor_y;
        //printf("%f %f %f %f %d\r\n",anchor_x1,anchor_y1,anchor_x2,anchor_y2,total_anchor_index);
        int delta_index = point_anchor_offset * feature_area * 4 + point_anchor_index;
        float delta_x   = bottom_delta[delta_index + 0 * feature_area];                           
        float delta_y   = bottom_delta[delta_index + 1 * feature_area];                           
        float delta_w   = bottom_delta[delta_index + 2 * feature_area];                           
        float delta_h   = bottom_delta[delta_index + 3 * feature_area];                           
        //printf("%f %f %f %f %d delta_index:%d score:%d\r\n",delta_x,delta_y,delta_w,delta_h,total_anchor_index,delta_index,s2index[i].score);

        anchor_w     = anchor_x2 - anchor_x1 + 1;
        anchor_h     = anchor_y2 - anchor_y1 + 1;
        anchor_ctr_x = anchor_x1 + 0.5 * anchor_w;
        anchor_ctr_y = anchor_y1 + 0.5 * anchor_h;

        pred_ctr_x = delta_x * anchor_w + anchor_ctr_x;
        pred_ctr_y = delta_y * anchor_h + anchor_ctr_y;
        pred_w     = exp(delta_w) * anchor_w;
        pred_h     = exp(delta_h) * anchor_h;

        x1 = pred_ctr_x - 0.5 * pred_w;
        x1 = x1<input_data_w? x1: input_data_w;
        x1 = x1>0? x1: 0;
        x2 = pred_ctr_x + 0.5 * pred_w;
        x2 = x2<input_data_w? x2: input_data_w;
        x2 = x2>0? x2: 0;
        tmp_w = x2 - x1 + 1;

        y1 = pred_ctr_y - 0.5 * pred_h;
        y1 = y1<input_data_h? y1: input_data_h;
        y1 = y1>0? y1: 0;
        y2 = pred_ctr_y + 0.5 * pred_h;
        y2 = y2<input_data_h? y2: input_data_h;
        y2 = y2>0? y2: 0;
        tmp_h = y2 - y1 + 1;
        
        if((tmp_w >= min_size) & (tmp_h >= min_size))
        {
            pre_rois[pre_roi_num].x1 = x1;
            pre_rois[pre_roi_num].x2 = x2;
            pre_rois[pre_roi_num].y1 = y1;
            pre_rois[pre_roi_num].y2 = y2;
            pre_rois[pre_roi_num].score     = s2index[i].score;
            pre_rois[pre_roi_num].prob_area = tmp_w * tmp_h;
            pre_roi_num++;
        }
        if (pre_roi_num >= PRE_NMS_NUM) break;//pre roi数量够,跳出循环
    }
    
    free(s2index);//释放s2index空间

    int j = 0, post_roi_num = 0;
    for(i = 0; i < pre_roi_num; i++)
    {
        if(pre_rois[i].prob_area == 0) continue;
        for(j = i + 1; j < pre_roi_num; j++)
        {
            if (box_iou(pre_rois[i], pre_rois[j]) > IOU_THRESH)
            {
                pre_rois[j].prob_area = 0;
            }
        }
        post_roi_num++;
        top_rois[post_roi_num].x1 = pre_rois[i].x1;
        top_rois[post_roi_num].y1 = pre_rois[i].y1;
        top_rois[post_roi_num].x2 = pre_rois[i].x2;
        top_rois[post_roi_num].y2 = pre_rois[i].y2;
        top_rois[post_roi_num].score = (float)((float)(pre_rois[i].score)/PREC);//除以精度后,强转回float类型
//        printf("index:%d %f %f %f %f %f\r\n",post_roi_num, \
            pre_rois[i].x1,pre_rois[i].y1,pre_rois[i].x2,pre_rois[i].y2,top_rois[post_roi_num].score);
        if(post_roi_num >= POST_NMS_NUM) break;// post roi数量够,跳出循环
    }
    free(pre_rois);  
    return post_roi_num;
}
