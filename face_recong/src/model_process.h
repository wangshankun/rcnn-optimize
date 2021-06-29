#ifndef MODEL_PROCESS_C_H_
#define MODEL_PROCESS_C_H_

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <vector>
#include <math.h>
#include <float.h>
#include <limits.h>

using namespace std;

#define savefile(name_x_, buffer_x_, size_x_) do\
{\
  FILE *fp_out_x_ = fopen(name_x_, "wb");\
  if(fp_out_x_ != NULL)\
  {\
        fwrite (buffer_x_ , sizeof(char), size_x_, fp_out_x_);\
        fclose (fp_out_x_);\
  }\
} while(0)

#define readfile(name_x_, buffer_x_, size_x_) do\
{\
  FILE *fp_out_x_ = fopen(name_x_, "rb");\
  if(fp_out_x_ != NULL)\
  {\
        fread (buffer_x_ , sizeof(char), size_x_, fp_out_x_);\
        fclose (fp_out_x_);\
  }\
} while(0)

////////////////////////////////////人脸检测部分/////////////////////////////////////////////

#include "live_128.h"

GLOW_MEM_ALIGN(LIVE_128_MEM_ALIGN)
uint8_t live_constantWeight[LIVE_128_CONSTANT_MEM_SIZE] = {
#include "live_128.weights.txt"
};

/// Statically allocate memory for mutable weights (model input/output data).
GLOW_MEM_ALIGN(LIVE_128_MEM_ALIGN)
uint8_t live_mutableWeight[LIVE_128_MUTABLE_MEM_SIZE];

/// Statically allocate memory for activations (model intermediate results).
GLOW_MEM_ALIGN(LIVE_128_MEM_ALIGN)
uint8_t live_activations[LIVE_128_ACTIVATIONS_MEM_SIZE];

/// Bundle input data absolute address.
uint8_t *input_live_Addr = GLOW_GET_ADDR(live_mutableWeight, LIVE_128_input);

/// Bundle output data absolute address.
uint8_t *output_score_Addr   = GLOW_GET_ADDR(live_mutableWeight, LIVE_128_scores);

uint8_t *output_boxes_Addr   = GLOW_GET_ADDR(live_mutableWeight, LIVE_128_boxes);

const float center_variance = 0.1;
const float size_variance   = 0.2;
const vector<vector<float>> min_boxes = {
    {32.0f,  48.0f},
    {64.0f,  96.0f},
    {128.0f, 192.0f, 256.0f} };
const vector<float> strides = { 16.0, 32.0, 64.0 };
static const vector<float> ratios = {1.0};

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} ObjBbox_t;

int is_in_array(short x, short y, short height, short width)
{
    if (x >= 0 && x < width && y >= 0 && y < height)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

//图片预处理resize后norm,一个循环做完;
void pre_process_resize_norm(unsigned char* in_array, short width, short height, 
                             float* out_array,  short out_width, short out_height,
                             float mean, float scale)//x = (x - mean)*scale)
{
    float h_times = (float)out_height / (float)height;
    float w_times = (float)out_width / (float)width;
    short x1, y1, x2, y2, f11, f12, f21, f22;
    float x, y;
    
    for (int i = 0; i < out_height; i++)
    {
        for (int j = 0; j < out_width; j++)
        {
            x = j / w_times;
            y = i / h_times;
            x1 = (short)(x - 1);
            x2 = (short)(x + 1);
            y1 = (short)(y + 1);
            y2 = (short)(y - 1);
            f11 = is_in_array(x1, y1, height, width) ? in_array[y1*width+x1] : 0;
            f12 = is_in_array(x1, y2, height, width) ? in_array[y2*width+x1] : 0;
            f21 = is_in_array(x2, y1, height, width) ? in_array[y1*width+x2] : 0;
            f22 = is_in_array(x2, y2, height, width) ? in_array[y2*width+x2] : 0;
            out_array[i*out_width+j] = (
                                          (float)((f11 * (x2 - x) * (y2 - y)) +
                                           (f21 * (x - x1) * (y2 - y)) +
                                           (f12 * (x2 - x) * (y - y1)) +
                                           (f22 * (x - x1) * (y - y1)))
                                           / 
                                          (float)((x2 - x1) * (y2 - y1))
                                        );
            out_array[i*out_width+j] = (out_array[i*out_width+j] - mean) * scale;
        }
    }
}

//产生先验框
void generate_anchor(vector<vector<float>>& priors,
                     int in_w, 
                     int in_h)
{
    vector<int> w_h_list = {in_w, in_h};
    vector<vector<float>> shrinkage_size;
    vector<vector<float>> featuremap_size;
    
    for (auto size : w_h_list)
    {
        vector<float> fm_item;
        for (float stride : strides)
        {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }
    for (auto size : w_h_list)
    {
        shrinkage_size.push_back(strides);
    }
    /* generate prior anchors */
    for (int index = 0; index < min_boxes.size(); index++)
    {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++)
        {
            for (int i = 0; i < featuremap_size[0][index]; i++)
            {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;
                for (float k : min_boxes[index])
                {
                    for (auto alpha : ratios)
                    {
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

//输出目标框(人脸识别、门锁、考勤机，业务只需要一张目标框，用这种后处理方式)
void generate_target_box(ObjBbox_t &target_box, 
                         vector<vector<float>> &priors,
                         float* scores,
                         float* boxes,
                         float threshold,
                         int   in_w,
                         int   in_h)
{
    int num_anchors = priors.size();
    memset(&target_box, 0, sizeof(ObjBbox_t));
    float max_area = 0;
    for (int i = 0; i < num_anchors; i++)
    {
        //选择满足threshold得分中面积最大的一张人脸框返回
        //此方法省去了nms的N^2次循环和众多anchor box的存储
        if (float(scores[i * 2 + 1]) > threshold)//二分类，背景/前景
        {
            float x_center = float(boxes[i * 4]) * center_variance * priors[i][2] + priors[i][0];
            float y_center = float(boxes[i * 4 + 1]) * center_variance * priors[i][3] + priors[i][1];
            float w = exp(float(boxes[i * 4 + 2]) * size_variance) * priors[i][2];
            float h = exp(float(boxes[i * 4 + 3]) * size_variance) * priors[i][3];

            int x1      = clip(x_center - w / 2.0, 1) * in_w;
            int y1      = clip(y_center - h / 2.0, 1) * in_h;
            int x2      = clip(x_center + w / 2.0, 1) * in_w;
            int y2      = clip(y_center + h / 2.0, 1) * in_h;
            float score = clip(float(scores[i * 2 + 1]), 1.0);
            float area  = (y2 - y1) * (x2 - x1);

            if (area > max_area)
            {
                target_box.x1    = x1;
                target_box.y1    = y1;
                target_box.x2    = x2;
                target_box.y2    = y2;
                target_box.score = score;
                max_area = area;
            }
        }
    }
}

////////////////////////////////////landmark部分/////////////////////////////////////////////
#include "lt_floor.h"

GLOW_MEM_ALIGN(LT_FLOOR_MEM_ALIGN)
uint8_t lt_floor_constantWeight[LT_FLOOR_CONSTANT_MEM_SIZE] = {
#include "lt_floor.weights.txt"
};

/// Statically allocate memory for mutable weights (model input/output data).
GLOW_MEM_ALIGN(LT_FLOOR_MEM_ALIGN)
uint8_t lt_floor_mutableWeight[LT_FLOOR_MUTABLE_MEM_SIZE];

/// Statically allocate memory for activations (model intermediate results).
GLOW_MEM_ALIGN(LT_FLOOR_MEM_ALIGN)
uint8_t lt_floor_activations[LT_FLOOR_ACTIVATIONS_MEM_SIZE];

/// Bundle input data absolute address.
uint8_t *input_lt_floor_Addr = GLOW_GET_ADDR(lt_floor_mutableWeight, LT_FLOOR_input_data);

/// Bundle output data absolute address.
uint8_t *output_probe_Addr   = GLOW_GET_ADDR(lt_floor_mutableWeight, LT_FLOOR_probe);
uint8_t *output_probe_x_Addr = GLOW_GET_ADDR(lt_floor_mutableWeight, LT_FLOOR_pred_x);
uint8_t *output_probe_y_Addr = GLOW_GET_ADDR(lt_floor_mutableWeight, LT_FLOOR_pred_y);

unsigned char* crop_pad_square(unsigned char* in_array, 
                               int x1, int y1, int x2, int y2,
                               int org_w, int org_h)
{
    if(x2 <= x1 || y2 <= y1) //左上角、右下角
    {
        return NULL;
    }
    int h = y2 - y1;
    int w = x2 - x1;
    unsigned char* out;
    int det_front,det_back;
    if (h >= w)//高大于宽，补宽度左右两边
    {
        det_front = (h - w) / 2;
        det_back  =  h - w - det_front;
        out = (unsigned char*)malloc( h * h );
        int k = 0;
        for (int j = y1; j < y2; j++)
        {
            for(int i = x1 - det_front; i < x2 + det_back; i++)
            {
                if (i < x1 || i > x2)
                {
                    out[k] = 0;//补0
                }
                else
                {
                    out[k] = in_array[j * org_w + i];
                }
                k++;
            }
        }
    }
    else//宽大于高，补高度上下两边
    {
        det_front = (w - h) / 2;
        det_back  =  w - h - det_front;
        out = (unsigned char*)malloc( w * w );
        int k = 0;
        for (int j = y1 - det_front; j < y2 + det_back; j++)
        {
            if (j < y1 || j > y2)
            {
                memset(&(out[k]), 0 , w);//整行补0
                k = k + w;
            }
            else
            {
                for(int i = x1; i < x2; i++)
                {
                    out[k] = in_array[j * org_w + i];
                    k++;
                }
            }
        }
    }
    
    return out;
}

void pre_process_resize(unsigned char* in_array, short width, short height, 
                        float* out_array, short out_width, short out_height)
{
    float h_times = (float)out_height / (float)height,
          w_times = (float)out_width / (float)width;
    short x1, y1, x2, y2, f11, f12, f21, f22;
    float x, y;
 
    for (int i = 0; i < out_height; i++)
    {
        for (int j = 0; j < out_width; j++)
        {
            x = j / w_times;
            y = i / h_times;
            x1 = (short)(x - 1);
            x2 = (short)(x + 1);
            y1 = (short)(y + 1);
            y2 = (short)(y - 1);
            f11 = is_in_array(x1, y1, height, width) ? in_array[y1*width+x1] : 0;
            f12 = is_in_array(x1, y2, height, width) ? in_array[y2*width+x1] : 0;
            f21 = is_in_array(x2, y1, height, width) ? in_array[y1*width+x2] : 0;
            f22 = is_in_array(x2, y2, height, width) ? in_array[y2*width+x2] : 0;
            out_array[i*out_width+j] = (
                                          (float)((f11 * (x2 - x) * (y2 - y)) +
                                           (f21 * (x - x1) * (y2 - y)) +
                                           (f12 * (x2 - x) * (y - y1)) +
                                           (f22 * (x - x1) * (y - y1)))
                                           / 
                                          (float)((x2 - x1) * (y2 - y1))
                                        );
        }
    }
}

void decode_landmark(float* landmark,
                     float* prob, float* offset_x_map, float* offset_y_map)
{
    for(int i = 0; i < 5; i++)
    {
        float max_score = FLT_MIN;
        int max_x_index = INT_MIN;
        int max_y_index = INT_MIN;
        for(int j = 0; j < 7*7; j++)
        {
            int x_index = j % 7;
            int y_index = j / 7;
            float score = prob[i * 49 + y_index * 7 + x_index];
            if (score > max_score)
            {
                max_score   = score;
                max_x_index = x_index;
                max_y_index = y_index;
            }
        }
        float offset_x = offset_x_map[i * 49 + max_y_index * 7 + max_x_index];
        float offset_y = offset_y_map[i * 49 + max_y_index * 7 + max_x_index];
        float pred_x = (max_x_index + offset_x) * 17.14285;//(17.14285= 120/7)相当于box regression
        float pred_y = (max_y_index + offset_y) * 17.14285;
        landmark[i * 2]     = pred_x;
        landmark[i * 2 + 1] = pred_y;
    }
}

void trans_coords(float* landmark, float scale, int dx = 0, int dy = 0)
{
    for(int i = 0; i < 5; i++)
    {
        landmark[i * 2]     = landmark[i * 2]     * scale - dx;
        landmark[i * 2 + 1] = landmark[i * 2 + 1] * scale - dy;
    }
}

#include "warp_affine.h"

int align_face(uint8_t* src_img, int src_w,
                uint8_t* dst_img, int dst_w,
                float*landmark_p)
{
    //人脸参数校准坐标:对于112x112
    float dst_ldk_x[5] = {38.2946,73.5318,56.0252,41.5493,70.729904};
    float dst_ldk_y[5] = {51.6963,51.5014,71.7366,92.3655,92.2041};
    //适配传入图片尺寸的校准坐标
    dst_ldk_x[0] = src_w / 112.0 * dst_ldk_x[0];
    dst_ldk_y[0] = src_w / 112.0 * dst_ldk_y[0];
    dst_ldk_x[1] = src_w / 112.0 * dst_ldk_x[1];
    dst_ldk_y[1] = src_w / 112.0 * dst_ldk_y[1];
    dst_ldk_x[2] = src_w / 112.0 * dst_ldk_x[2];
    dst_ldk_y[2] = src_w / 112.0 * dst_ldk_y[2];
    dst_ldk_x[3] = src_w / 112.0 * dst_ldk_x[3];
    dst_ldk_y[3] = src_w / 112.0 * dst_ldk_y[3];
    dst_ldk_x[4] = src_w / 112.0 * dst_ldk_x[4];
    dst_ldk_y[4] = src_w / 112.0 * dst_ldk_y[4];
    
    float src_ldk_x[5] = {0};
    float src_ldk_y[5] = {0};
    src_ldk_x[0] = landmark_p[0];
    src_ldk_y[0] = landmark_p[1];
    src_ldk_x[1] = landmark_p[2];
    src_ldk_y[1] = landmark_p[3];
    src_ldk_x[2] = landmark_p[4];
    src_ldk_y[2] = landmark_p[5];
    src_ldk_x[3] = landmark_p[6];
    src_ldk_y[3] = landmark_p[7];
    src_ldk_x[4] = landmark_p[8];
    src_ldk_y[4] = landmark_p[9];
    //拿获取的landmark与校准坐标对比，得到变换矩阵
    Matrix *M = get_similarity_matrix(src_ldk_x, src_ldk_y, dst_ldk_x, dst_ldk_y, 5);
    if(M == NULL)
    {
        return -1;
    }

    dl_matrix3du_t src, dst;
    src.n = 1;
    src.c = 1;
    src.h = src_w;
    src.w = src_w;
    src.item = src_img;
    src.stride = src.c * src.w;
 
    dst.n = 1;
    dst.c = 1;
    dst.h = dst_w;
    dst.w = dst_w;
    dst.item = dst_img;
    dst.stride = dst.c * dst.w;
    //执行仿射变换
    int pad_count =  warp_affine(&src, &dst, M);
    float pad_score = float(pad_count) / float(dst_w*dst_w);
    //matrix_print(M);
    matrix_free(M);
    if(pad_score >= 0.5)//仿射变换后如果太多padding就放弃
    {
	return -1;
    }
    return 0;
}


////////////////////////////////////人脸图片向量化部分/////////////////////////////////////////////
#include "facerecong.h"

GLOW_MEM_ALIGN(FACERECONG_MEM_ALIGN)
uint8_t facerecong_constantWeight[FACERECONG_CONSTANT_MEM_SIZE] = {
#include "facerecong.weights.txt"
};

/// Statically allocate memory for mutable weights (model input/output data).
GLOW_MEM_ALIGN(FACERECONG_MEM_ALIGN)
uint8_t facerecong_mutableWeight[FACERECONG_MUTABLE_MEM_SIZE];

/// Statically allocate memory for activations (model intermediate results).
GLOW_MEM_ALIGN(FACERECONG_MEM_ALIGN)
uint8_t facerecong_activations[FACERECONG_ACTIVATIONS_MEM_SIZE];

/// Bundle input data absolute address.
uint8_t *input_facerecong_Addr  = GLOW_GET_ADDR(facerecong_mutableWeight, FACERECONG_input_1);
/// Bundle output data absolute address.
uint8_t *output_facerecong_Addr = GLOW_GET_ADDR(facerecong_mutableWeight, FACERECONG_A124);


/////////////////////////////模型执行过程封装//////////////////////////////////////////////////

static int face_det(dl_matrix3du_t* org_img, dl_matrix3du_t* det_face_img, float threshold = 0.5)
{
    int in_w = 128; int in_h = 128;//模型的输入尺寸
    float mean = 127.0; float norm = 1.0/128.0;
    float scale_w = (float)org_img->w / (float)in_w;
    float scale_h = (float)org_img->h / (float)in_h;
    
    vector<vector<float>> priors;
    generate_anchor(priors, in_w, in_h);
    //预处理
    pre_process_resize_norm(org_img->item,            org_img->w, org_img->h,
                            (float*)input_live_Addr,  in_w ,      in_h, 
                            mean, norm);
    int errCode = live_128(live_constantWeight, live_mutableWeight, live_activations);
    if (errCode != GLOW_SUCCESS)
    {
        printf("FaceDet: Error running bundle: error code %d\n", errCode);
        return errCode;
    }
    ObjBbox_t target_box; memset(&target_box, 0, sizeof(target_box));
    generate_target_box(target_box, priors, (float*)output_score_Addr, (float*)output_boxes_Addr, threshold, 128, 128);
    
    int x1 = int (target_box.x1 * scale_w);
    int y1 = int (target_box.y1 * scale_h);
    int x2 = int (target_box.x2 * scale_w);
    int y2 = int (target_box.y2 * scale_h);
    x1 = (x1 < 0) ? 0 : x1;
    x2 = (x2 < 0) ? 0 : x2;
    y1 = (y1 < 0) ? 0 : y1;
    y2 = (y2 < 0) ? 0 : y2;
    x1 = (x1 > org_img->w) ? org_img->w: x1;
    x2 = (x2 > org_img->w) ? org_img->w: x2;
    y1 = (y1 > org_img->h) ? org_img->h: y1;
    y2 = (y2 > org_img->h) ? org_img->h: y2;

    //printf("Key: %d %d %d %d %f RegScore: \r\n",x1,y1,x2,y2,target_box.score);
    if(target_box.score < threshold)
    {
        printf("Not Found Face!\n");
        return -1;
    }
    
    unsigned char* crop_img = crop_pad_square(org_img->item, x1, y1, x2, y2, org_img->w, org_img->h);
    int h = y2 - y1;
    int w = x2 - x1;
    int s = (h > w) ? h : w;
    det_face_img->n = 1;
    det_face_img->c = 1;
    det_face_img->h = s;
    det_face_img->w = s;
    det_face_img->item = crop_img;
    det_face_img->stride = s;

    return 0;
}

static int face_align(dl_matrix3du_t* det_face_img, dl_matrix3du_t* face_align_img)
{
    int square = det_face_img->w;//上一步骤已经padding成方形图
    int in_w = 120; int in_h = 120;//模型的输入尺寸,方形
    float scale = (float)square / (float)in_w;

    pre_process_resize(det_face_img->item, square, square, (float*)input_lt_floor_Addr, in_w, in_h);

    int errCode = lt_floor(lt_floor_constantWeight, lt_floor_mutableWeight, lt_floor_activations);
    if (errCode != GLOW_SUCCESS)
    {
        printf("LandMark: Error running bundle: error code %d\n", errCode);
        return -1;
    }
    float landmark[10] = {};
    //后处理获得landmark
    decode_landmark(landmark, (float*)output_probe_Addr, (float*)output_probe_x_Addr, (float*)output_probe_y_Addr);
    trans_coords(landmark, scale);
    //仿射变换对齐人脸
    uint8_t* align_img = (uint8_t*)malloc(square * square);

    if(align_face(det_face_img->item, square, align_img, square, landmark) != 0)
    {
        free(align_img); align_img = NULL;
        printf("Error align face! \r\n");
        return -1;
    }
    face_align_img->n = 1;
    face_align_img->c = 1;
    face_align_img->h = square;
    face_align_img->w = square;
    face_align_img->item = align_img;
    face_align_img->stride = square;
    return 0;
}

static int face_recong(dl_matrix3du_t* face_align_img, dl_matrix3d_t* face_vector)
{
    int in_w = 56; int in_h = 56;//模型的输入尺寸,方形
    float mean = 127.5; float norm = 1.0/128.0;
    int out_size = 64;
    pre_process_resize_norm(face_align_img->item,          face_align_img->w, face_align_img->h,
                            (float*)input_facerecong_Addr, in_w ,      in_h, 
                            mean, norm);

    int errCode = facerecong(facerecong_constantWeight, facerecong_mutableWeight, facerecong_activations);
    if (errCode != GLOW_SUCCESS)
    {
        printf("FaceRecong: Error running bundle: error code %d\n", errCode);
        return -1;
    }
    float* vector = (float*)malloc(out_size * sizeof(float));
    memcpy(vector, output_facerecong_Addr, out_size * sizeof(float));
    face_vector->n = 1;
    face_vector->c = 64;//模型的输出是64个float组成的vector
    face_vector->h = 0;
    face_vector->w = 0;
    face_vector->item = vector;
    face_vector->stride = 0;
    return 0;
}

static int generate_face_id(dl_matrix3du_t* org_img_p, dl_matrix3d_t* face_vector_p, float threshold = 0.5)
{
    dl_matrix3du_t  det_face_img, face_align_img;
    if(face_det(org_img_p, &det_face_img, threshold) != 0)
    {
        free(org_img_p->item); org_img_p->item = NULL;
        return -1;
    }
    free(org_img_p->item); org_img_p->item = NULL;

    if(face_align(&det_face_img, &face_align_img) != 0)
    {
        printf("Face Align Error! \r\n");
        free(det_face_img.item); det_face_img.item = NULL;
        return -1;
    }
    free(det_face_img.item); det_face_img.item = NULL;
    
    if(face_recong(&face_align_img, face_vector_p) != 0)
    {
        printf("Face Recong Error! \r\n");
        free(face_align_img.item); face_align_img.item = NULL;
        return -1;
    }
    free(face_align_img.item); face_align_img.item = NULL;
    //savefile("facerecong_output.bin", face_vector_p->item, face_vector_p->w * sizeof(float));
    return 0;
}

#endif
