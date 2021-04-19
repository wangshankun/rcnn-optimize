#ifndef _THIS_T31_0316_SSD_MODEL_PROCESS_BASE_H
#define _THIS_T31_0316_SSD_MODEL_PROCESS_BASE_H

#define IS_PADDING true
int IS_DRAW  = 0;
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

//预设,模型本身相关的参数
static const vector<vector<float>> min_boxes = {
                                                  {10.0f,  16.0f,  24.0f},
                                                  {32.0f,  48.0f},
                                                  {64.0f,  96.0f},
                                                  {128.0f, 192.0f, 256.0f}
                                                };

static const vector<float> strides = {8.0, 16.0, 32.0, 64.0};
static const vector<float> ratios = {0.5, 1.0, 2.0};
static const float center_variance = 0.1;
static const float size_variance   = 0.2;

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
            in_boxes[i].x0 = (in_boxes[i].x0 - pad_x) / scale;
            in_boxes[i].x1 = (in_boxes[i].x1 - pad_x) / scale;
            in_boxes[i].y0 = (in_boxes[i].y0 - pad_y) / scale;
            in_boxes[i].y1 = (in_boxes[i].y1 - pad_y) / scale;
        }
    }
    else
    {
        for(int i = 0; i < in_boxes.size(); i++) {
            in_boxes[i].x0 = in_boxes[i].x0 / scale;
            in_boxes[i].x1 = in_boxes[i].x1 / scale;
            in_boxes[i].y0 = in_boxes[i].y0 / scale;
            in_boxes[i].y1 = in_boxes[i].y1 / scale;
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
        int num_anchor = min_boxes[i].size();
        bb_ptr = sc_ptr+width_*height_*num_anchor*class_num;
        for(int h=0; h<height_; h++)
        {
            for(int w=0; w<width_; w++)
            {
                for(int n=0; n<num_anchor; n++)
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
                    float w = k / in_w;
                    float h = k / in_h;
                    priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
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

static void generateBBox(vector<ObjMbox_t> &detect_list, vector<float> &scores, vector<std::vector<float>> &priors, 
                  vector<float> &boxes, float score_threshold,
                  int num_anchors, int class_num, 
                  int ori_w, int ori_h, 
                  int in_w,int in_h)
{ 
    std::vector<ObjBbox_t> bbox_collection;

    vector<int> pass_index;
    float reverse_score = 1.0 - score_threshold;//反向得分
    for (int i = 0; i < num_anchors; i++) 
    {
        float temp_score = softmax(scores, i, 0, class_num);//统计背景类得分
        if(temp_score < reverse_score)
        {
            pass_index.push_back(i);//合格anchors的index记录下来
        }

    }

    for (int clsn = 1; clsn < class_num; clsn++) 
    {
        for (auto i : pass_index)
        {
            float temp_score = softmax(scores, i, clsn, class_num);
            if ( temp_score > score_threshold ) 
            {
                ObjBbox_t rects;
                float x_center = boxes[i * 4] * 0.1 * priors[i][2] + priors[i][0];
                float y_center = boxes[i * 4 + 1] * 0.1 * priors[i][3] + priors[i][1];
                float w = exp(boxes[i * 4 + 2] * 0.2) * priors[i][2];
                float h = exp(boxes[i * 4 + 3] * 0.2) * priors[i][3];

                rects.x0 = clip(x_center - w / 2.0, 1) * in_w;
                rects.y0 = clip(y_center - h / 2.0, 1) * in_h;
                rects.x1 = clip(x_center + w / 2.0, 1) * in_w;
                rects.y1 = clip(y_center + h / 2.0, 1) * in_h;
                rects.score = clip(temp_score, 1);
                bbox_collection.push_back(rects);
            }
        }
        vector<ObjBbox_t> box_list;
        nms(bbox_collection, box_list);
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