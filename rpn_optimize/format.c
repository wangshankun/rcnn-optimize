#include"format.h"
void main()
{
    int feature_area = feature_w * feature_h;
    int anchor_num   = feature_w * feature_h * anchor_c;
    float  min_size  = rpn_min_size * input_data_scale;

    pre_rois_t* pre_rois = calloc(pre_nms_top,  sizeof(pre_rois_t));
    top_rois_t* top_rois = calloc(post_nms_top,  sizeof(top_rois_t));
    
    int fd_anchor, fd_score, fd_delta;

    anchor_t* base_anchor = calloc(anchor_c,   sizeof(anchor_t));
    float* score_buf      = calloc(anchor_num, sizeof(float));
    delta_t* delta_buf    = calloc(anchor_num, sizeof(delta_t));

    if((fd_anchor = fopen("./base_anchor.bin","rb")) ==-1) return -1;
    if((fd_score = fopen("./rpn_cls_prob_reshape.bin","rb")) ==-1) return -1;
    if((fd_delta = fopen("./rpn_bbox_pred.bin","rb")) ==-1) return -1;

    fread(base_anchor, sizeof(anchor_t), anchor_c,  fd_anchor);
    //获取前景得分,安caffe默认格式(1,A,H,W),后一半数据是我们需要的前景得分
    int offset_bytes = sizeof(float) * anchor_num;
    fseek(fd_score, offset_bytes, 0);//只读取后半部分
    fread(score_buf, sizeof(float), anchor_num,  fd_score);
    fread(delta_buf, sizeof(delta_t), anchor_num,  fd_delta);
    
    close(fd_anchor);
    close(fd_score);
    close(fd_delta);
    
    index_score_t* s2index = calloc(anchor_num, sizeof(index_score_t));
    

    struct timespec start, finish;
    double elapsed = 0;
    clock_gettime(CLOCK_REALTIME, &start);
    int test_count = 0;
    
    for(test_count=0; test_count < 100; test_count++)//测试100次统计时间
    {
        int i = 0, select_num = 0, pre_roi_num = 0;

        for(i = 0; i < anchor_num; i++)
        {
            if(score_buf[i] < SCORE_THRESH)//筛掉分数不合格的anchor,省大部分计算量
            {
                continue;
            }
            //score放大指定精度倍数(float32类型是10的6次方小数)后强转为整形,快排中有速度优势
            s2index[select_num].score = (int)(score_buf[i] * PREC);
            s2index[select_num].index = i;
            select_num                = select_num + 1;
        }
        
        //排序从高到底，起始地址为s2index，只排序前select_num个元素
        qsort(s2index, select_num, sizeof(index_score_t), score_reversed);

        float anchor_w, anchor_h, anchor_ctr_x, anchor_ctr_y;
        float pred_ctr_x, pred_ctr_y, pred_w, pred_h, x1, x2, y1, y2, tmp_w, tmp_h;

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
            float anchor_x1 = base_anchor[point_anchor_offset].x1 + 16 * point_anchor_x;//16是anchor的步长                               
            float anchor_y1 = base_anchor[point_anchor_offset].y1 + 16 * point_anchor_y;                         
            float anchor_x2 = base_anchor[point_anchor_offset].x2 + 16 * point_anchor_x;                         
            float anchor_y2 = base_anchor[point_anchor_offset].y2 + 16 * point_anchor_y;
            float delta_x   = delta_buf[total_anchor_index].x;                           
            float delta_y   = delta_buf[total_anchor_index].y;                           
            float delta_w   = delta_buf[total_anchor_index].w;                           
            float delta_h   = delta_buf[total_anchor_index].h;                           
            //int score  = s2index[i].score;

            //开始做bounding box regression
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
            
            if((tmp_w >= min_size) & (tmp_h >= min_size))//过小尺寸不进入预选
            {
                pre_rois[pre_roi_num].x1 = x1;
                pre_rois[pre_roi_num].x2 = x2;
                pre_rois[pre_roi_num].y1 = y1;
                pre_rois[pre_roi_num].y2 = y2;
                pre_rois[pre_roi_num].score     = s2index[i].score;
                pre_rois[pre_roi_num].prob_area = tmp_w * tmp_h;
                pre_roi_num++;
            }
            if (pre_roi_num >= pre_nms_top) break;//pre roi数量够,跳出循环
        }
        //开始做NMS, prob_area是存储box面积的，同时也做nms筛选的flag,如果面积被设置为0就被筛掉 
        int j = 0, post_roi_num = 0;
        for(i = 0; i < pre_roi_num; i++)
        {
            if(pre_rois[i].prob_area == 0) continue;
            for(j = i + 1; j < pre_roi_num; j++)
            {
                if (box_iou(pre_rois[i], pre_rois[j]) > iou_thresh)
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
            if(post_roi_num >= post_nms_top) break;// post roi数量够,跳出循环
        }   
    }
    clock_gettime(CLOCK_REALTIME, &finish);
    elapsed += (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("test 100 times elapsed time:%f  \r\n",elapsed);
    free(base_anchor);
    free(score_buf);
    free(delta_buf);
    free(pre_rois);
    free(top_rois);   
}
