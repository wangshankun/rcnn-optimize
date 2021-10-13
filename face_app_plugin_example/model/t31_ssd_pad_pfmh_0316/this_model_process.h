#ifndef _THIS_T31_0316_SSD_MODEL_PROCESS_BASE_H
#define _THIS_T31_0316_SSD_MODEL_PROCESS_BASE_H

#define printf(MESSAGE,args...) { \
  const char *A[] = {MESSAGE}; \
  printf("%s:%d",__FILE__,__LINE__); fflush(stdout);\
  if(sizeof(A) > 0) {\
    printf("||"); \
    printf(*A,##args); \
  } else {\
    printf("\n"); \
  }\
}

#define IS_PADDING true
int IS_DRAW  = 0;
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

#define max_valid_anchor_num 60

//预设,模型本身相关的参数
static const vector<vector<float>> min_boxes = {
                                                  {10.0f,  16.0f,  24.0f},
                                                  {32.0f,  48.0f},
                                                  {64.0f,  96.0f},
                                                  {128.0f, 192.0f, 256.0f}
                                                };

static const vector<float> ratios = {1.0};

/*//预设,模型本身相关的参数
static const vector<vector<float>> min_boxes = {
                                                  {8.0f,  16.0f,  24.0f},
                                                  {32.0f, 40.0f,  48.0f},
                                                  {64.0f, 80.0f,  96.0f},
                                                  {128.0f, 192.0f, 256.0f}
                                                };
static const vector<float> ratios = {0.5, 1.0, 2.0};
 */

static const vector<float> strides = {8.0, 16.0, 32.0, 64.0};
static const float center_variance = 0.1;
static const float size_variance   = 0.2;
static const float iou_threshold = 0.4;

typedef struct {
    int       clsn;
    float     x0;
    float     y0;
    float     x1;
    float     y1;
    float     score;
} ObjMbox_t;

static void trans_coords(int ori_w,int ori_h,int input_w,int input_h, vector<ObjBbox_t> &in_boxes) {
    float scale;
    float pad_x, pad_y;
    scale = min((float(input_w)/ori_w), (float(input_h)/ori_h));

    if( IS_PADDING )
    {
        pad_x = (input_w - ori_w * scale) / 2;
        pad_y = (input_h - ori_h * scale) / 2;
        for(int i = 0; i < in_boxes.size(); i++) {
            in_boxes[i].x0 = clip((in_boxes[i].x0 - pad_x) / scale, ori_w);
            in_boxes[i].x1 = clip((in_boxes[i].x1 - pad_x) / scale, ori_w);
            in_boxes[i].y0 = clip((in_boxes[i].y0 - pad_y) / scale, ori_h);
            in_boxes[i].y1 = clip((in_boxes[i].y1 - pad_y) / scale, ori_h);
        }
    }
    else
    {
        for(int i = 0; i < in_boxes.size(); i++) {
            in_boxes[i].x0 = clip(in_boxes[i].x0 / scale, ori_w);
            in_boxes[i].x1 = clip(in_boxes[i].x1 / scale, ori_w);
            in_boxes[i].y0 = clip(in_boxes[i].y0 / scale, ori_h);
            in_boxes[i].y1 = clip(in_boxes[i].y1 / scale, ori_h);
        } 
    }
}

static void prepare_data_resize_pad(const jzdl::Mat<uint8_t>& in, jzdl::Mat<uint8_t>& out, 
                            int model_in_w, int model_in_h)
{
    jzdl::PadInfo_t padinfo;
    int input_w = in.w;
    int input_h = in.h;
    
    int output_w, output_h;
    int dw, dh;
    int top, bottom, left, right;
    
    int con_w = model_in_w;
    int con_h = model_in_h;

    float h_scale = float(con_h) / float(input_h);
    float w_scale = float(con_w) / float(input_w);

    float scale = min(h_scale, w_scale);
    output_w = int(round(input_w * scale));
    output_h = int(round(input_h * scale));
    dw = (con_w - output_w);
    dh = (con_h - output_h);
    
    jzdl::Mat<uint8_t> temp(output_w, output_h, 3);

    if(output_h != input_h || output_h != input_w){
        jzdl::resize(in, temp);

    } else{
        temp = in.clone();
    }

    top = int(round(float(dh)/2 - 0.1));
    bottom = int(round(float(dh)/2 + 0.1));
    left = int(round(float(dw)/2 - 0.1));
    right = int(round(float(dw)/2 + 0.1));

    padinfo.top = top;
    padinfo.bottom = bottom;
    padinfo.left = left;
    padinfo.right = right;
    padinfo.type = PAD_CONSTANT;
    padinfo.value = 128;
    jzdl::image_pad(temp, out, padinfo);//填充灰色的值(128)
}

static void prepare_data_pad_resize(const jzdl::Mat<uint8_t>& in, jzdl::Mat<uint8_t>& out, 
                            int model_in_w, int model_in_h)
{
    jzdl::PadInfo_t padinfo;
    int input_w = in.w;
    int input_h = in.h;
    
    int output_w, output_h;
    int dw, dh;
    int top, bottom, left, right;
    
    int con_w = model_in_w;
    int con_h = model_in_h;

    float h_scale = float(input_h) / float(con_h);
    float w_scale = float(input_w) / float(con_w);

    float scale = max(h_scale, w_scale);
    output_w = int(round(con_w * scale));
    output_h = int(round(con_h * scale));
    dw = (output_w - input_w);
    dh = (output_h - input_h);

    jzdl::Mat<uint8_t> temp(output_w, output_h, 3);
    
    top = int(round(float(dh)/2 - 0.1));
    bottom = int(round(float(dh)/2 + 0.1));
    left = int(round(float(dw)/2 - 0.1));
    right = int(round(float(dw)/2 + 0.1));

    padinfo.top = top;
    padinfo.bottom = bottom;
    padinfo.left = left;
    padinfo.right = right;
    padinfo.type = PAD_CONSTANT;
    padinfo.value = 128;
    jzdl::image_pad(in, temp, padinfo);//填充灰色的值(128)
    jzdl::resize(temp, out);
}

static int  format_out(jzdl::Mat<float>& out, vector<vector<float>>& featuremap_size, 
                           vector<float>& scores, vector<float>& boxes)
{
    int out_size = (int)(out.w*out.h*out.c);
    int out_num = (int)out.data[out.w*out.h*out.c-1]; //君正out的格式，最后一个int是out的mat个数
    int last_box_channel = (int)out.data[out.w*out.h*out.c-2];//依次cwh的  box shape信息
    int last_score_channel = (int)out.data[out.w*out.h*out.c-5];//score shape信息
    int class_num = last_score_channel / (last_box_channel / 4);//box channel固定从4个坐标乘来的

    float *sc_ptr;
    sc_ptr = out.data;
    float* bb_ptr;

    for (int i=0;i<min_boxes.size();i++)
    {
        int width_  = featuremap_size[0][i];
        int height_ = featuremap_size[1][i];
        int num_anchor = min_boxes[i].size() * ratios.size();
        bb_ptr = sc_ptr + width_* height_* num_anchor * class_num;
        for(int h = 0; h < height_; h++)
        {
            for(int w = 0; w < width_; w++)
            {
                for(int n = 0; n < num_anchor; n++)
                {
                    boxes.push_back(bb_ptr[h*(width_*num_anchor*4)+w*(num_anchor*4)+n*4]);
                    boxes.push_back(bb_ptr[h*(width_*num_anchor*4)+w*(num_anchor*4)+n*4+1]);
                    boxes.push_back(bb_ptr[h*(width_*num_anchor*4)+w*(num_anchor*4)+n*4+2]);
                    boxes.push_back(bb_ptr[h*(width_*num_anchor*4)+w*(num_anchor*4)+n*4+3]);

                     for(int cls=0; cls<class_num; cls++)
                     {
                        scores.push_back(sc_ptr[h*(width_*num_anchor*class_num)+w*(num_anchor*class_num)+n*class_num+cls]);
                     }
                }
            }
        }
        sc_ptr = bb_ptr+width_*height_*num_anchor*4;
    }

    return class_num;
}

static void generate_anchor(vector<vector<float>>& priors, vector<vector<float>>& featuremap_size, int in_w, int in_h)
{
    vector<int> w_h_list = {in_w, in_h};
    vector<vector<float>> shrinkage_size;

    for (auto size : w_h_list) {
        vector<float> fm_item;
        for (float stride : strides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }
    for (auto size : w_h_list) {
        shrinkage_size.push_back(strides);
    }
    /* generate prior anchors */
    for (int index = 0; index < 4; index++) {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;
                for (float k : min_boxes[index]) {
                    for (auto alpha : ratios) {
                        float w = k / in_w;
                        float h = k / in_h;
                        float w1 = sqrt(w * h) * sqrt(alpha);
                        float h1 = sqrt(w * h) / sqrt(alpha);
                        priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w1, 1), clip(h1, 1)});
                    }
                }
            }
        }
    }
}

static float softmax(std::vector<float>& scores, int anc_indx ,int class_indx, int class_numbers)
{
    float max = 0.0;
    float sum = 0.0;

    for (int i = 0; i<class_numbers;i++)
    {
      if(max < scores[anc_indx*class_numbers+i]) {
          max = scores[anc_indx*class_numbers+i];
      }
    }

    float temp_score = 0.;
    for (int j = 0; j<class_numbers;j++)
    {
      float temp_value = exp(scores[anc_indx*class_numbers+j] - max);
      sum += temp_value;
      if (j == class_indx)
          temp_score = temp_value;
    }

    return temp_score / sum;
}

static void anchor_filter(vector<float>& scores, 
                            vector<pair<int, float>>& valiad_anchor_vec,
                            const int anc_indx , 
                            const int class_numbers,
                            const float score_threshold)
{
    float first_max = 0.0;
    float second_max = 0.0;
    for (int i = 0; i < class_numbers; i++)
    {
        if(first_max < scores[anc_indx * class_numbers + i])
        {
            second_max = first_max;
            first_max = scores[anc_indx * class_numbers + i];
        }
    }

    float det_bg_max = scores[anc_indx * class_numbers + 0] - first_max;
    float det_second_max = second_max - first_max;
    if( det_bg_max < 0)//背景类不是max情况下，统计max目标类大于threshold情况
    {
        float sum = 0.0;
        for (int j = 0; j < class_numbers; j++)
        {
            float temp_value = exp(scores[anc_indx * class_numbers + j] - first_max);
            sum += temp_value;
        }

        float target_score = 1.0 / sum;//e^(max-max)=e^0=1.0
        if (target_score >= score_threshold)
        {
            valiad_anchor_vec.push_back(pair<int, float>(anc_indx, target_score));
        }
    }
    else
    {   
        bool  is_need_calculate = false;
        if(score_threshold >= 0.5)
        {
            //index 0是背景类, 背景类是max情况下，目标类得分一定低于0.5
            return;
        }
        if(score_threshold < 0.5 && score_threshold >=0.4)
        {
            //目标类减去max至少要大于-0.405才可能出现超过0.4分
            if(det_second_max >= -0.405) is_need_calculate = true;
        }
        if(score_threshold < 0.4 && score_threshold >=0.3)
        {
            if(det_second_max >= -0.847) is_need_calculate = true;
        }
        if(score_threshold < 0.3 && score_threshold >=0.2)
        {
            if(det_second_max <= -1.386) is_need_calculate = true;
        }
        if(score_threshold < 0.2 && score_threshold >=0.1)
        {
            if(det_second_max <= -2.197) is_need_calculate = true;
        }//score_threshold最低为0.1

        if( is_need_calculate )
        {
            float sum = 0.0;
            for (int j = 0; j < class_numbers; j++)
            {
                float temp_value = exp(scores[anc_indx * class_numbers + j] - first_max);
                sum += temp_value;
            }

            float target_score = exp(det_second_max)/ sum;
            if (target_score >= score_threshold)
            {
                valiad_anchor_vec.push_back(pair<int, float>(anc_indx, target_score));
            }
        }
    }
}

bool cmp(pair<int, float> a, pair<int, float> b) {
    return a.second > b.second;
}

void generateBBox(vector<ObjMbox_t> &detect_list, vector<float> &scores, vector<std::vector<float>> &priors, 
                  vector<float> &boxes, float score_threshold,
                  int num_anchors, int class_num, 
                  int ori_w, int ori_h, 
                  int in_w,int in_h)
{ 

    std::vector<ObjBbox_t> bbox_collection;

    vector<pair<int, float>> valiad_anchor_vec;
    vector<int> pass_index;
    for (int i = 0; i < num_anchors; i++) 
    {
        anchor_filter(scores, valiad_anchor_vec, i, class_num, score_threshold);
    }
    if( valiad_anchor_vec.size() > max_valid_anchor_num)
    {
        sort(valiad_anchor_vec.begin(), valiad_anchor_vec.end(), cmp);//排序取最前面的
    }

    for (int clsn = 1; clsn < class_num; clsn++) 
    {
        for (int j = 0; j < valiad_anchor_vec.size() && j < max_valid_anchor_num; j++)
        {    
            int i = valiad_anchor_vec[j].first;
            //ToDo:
            //如果score_threshold大于等于0.5的话valiad_anchor_vec[j].second可以直接作为最终的score
            //以及对应的clsn做最终的分类;
            //当score_threshold在0.1~0.3时候有一定概率valiad_anchor_vec[j].second做score是不全的
            //不过如果要进一步优化，未必不可以用，只有在少见情况下不适用;
            float temp_score = softmax(scores, i, clsn, class_num);
            if ( temp_score > score_threshold ) 
            {
                ObjBbox_t rects;
                float x_center = boxes[i * 4] * center_variance * priors[i][2] + priors[i][0];
                float y_center = boxes[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
                float w = exp(boxes[i * 4 + 2] * size_variance) * priors[i][2];
                float h = exp(boxes[i * 4 + 3] * size_variance) * priors[i][3];
                rects.x0 = clip(x_center - w / 2.0, 1.0) * in_w;
                rects.y0 = clip(y_center - h / 2.0, 1.0) * in_h;
                rects.x1 = clip(x_center + w / 2.0, 1.0) * in_w;
                rects.y1 = clip(y_center + h / 2.0, 1.0) * in_h;
                rects.score = clip(temp_score, 1.0);
                bbox_collection.push_back(rects);
            }
        }

        vector<ObjBbox_t> box_list;

        nms(bbox_collection, box_list, iou_threshold);
        bbox_collection.clear();

        trans_coords(ori_w, ori_h, in_w, in_h, box_list);

        for (int i = 0; i < box_list.size(); i++) 
        {
            auto box = box_list[i];
            ObjMbox_t  result;
            result.clsn  = clsn;
            result.x0    = box.x0;
            result.y0    = box.y0;
            result.x1    = box.x1;
            result.y1    = box.y1;
            result.score = box.score;
            detect_list.push_back(result);
        }
        box_list.clear();
    }
}

static void NV12_TO_RGB24(unsigned char *yuyv, unsigned char *rgb, int width, int height)
{
        const int nv_start = width * height ;
        int  index = 0, rgb_index = 0;
        uint8_t y, u, v;
        int r, g, b, nv_index = 0,i, j;
 
        for(i = 0; i < height; i++){
            for(j = 0; j < width; j ++){
                //nv_index = (rgb_index / 2 - width / 2 * ((i + 1) / 2)) * 2;
                nv_index = i / 2  * width + j - j % 2;
 
                y = yuyv[rgb_index];
                v = yuyv[nv_start + nv_index ];
                u = yuyv[nv_start + nv_index + 1];
 
                r = y + (140 * (v-128))/100;  //r
                g = y - (34 * (u-128))/100 - (71 * (v-128))/100; //g
                b = y + (177 * (u-128))/100; //b
 
                if(r > 255)   r = 255;
                if(g > 255)   g = 255;
                if(b > 255)   b = 255;
                if(r < 0)     r = 0;
                if(g < 0)     g = 0;
                if(b < 0)     b = 0;
 
                index = rgb_index % width + (height - i - 1) * width;

                rgb[i * width * 3 + 3 * j + 0] = b;
                rgb[i * width * 3 + 3 * j + 1] = g;
                rgb[i * width * 3 + 3 * j + 2] = r;
 
                rgb_index++;
            }
        }
}

static void NV21_TO_RGB24(unsigned char *yuyv, unsigned char *rgb, int width, int height)
{
        const int nv_start = width * height ;
        int  index = 0, rgb_index = 0;
        uint8_t y, u, v;
        int r, g, b, nv_index = 0,i, j;
 
        for(i = 0; i < height; i++){
            for(j = 0; j < width; j ++){
                //nv_index = (rgb_index / 2 - width / 2 * ((i + 1) / 2)) * 2;
                nv_index = i / 2  * width + j - j % 2;
 
                y = yuyv[rgb_index];
                u = yuyv[nv_start + nv_index ];
                v = yuyv[nv_start + nv_index + 1];
 
                r = y + (140 * (v-128))/100;  //r
                g = y - (34 * (u-128))/100 - (71 * (v-128))/100; //g
                b = y + (177 * (u-128))/100; //b
 
                if(r > 255)   r = 255;
                if(g > 255)   g = 255;
                if(b > 255)   b = 255;
                if(r < 0)     r = 0;
                if(g < 0)     g = 0;
                if(b < 0)     b = 0;
 
                index = rgb_index % width + (height - i - 1) * width;

                rgb[i * width * 3 + 3 * j + 0] = b;
                rgb[i * width * 3 + 3 * j + 1] = g;
                rgb[i * width * 3 + 3 * j + 2] = r;
 
                rgb_index++;
            }
        }
}


#endif
