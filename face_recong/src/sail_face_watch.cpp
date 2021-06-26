#include "model_process.h"

#if defined(__cplusplus)
extern "C"
#endif
int face_det(dl_matrix3du_t* org_img, dl_matrix3du_t* det_face_img)
{
    int in_w = 128; int in_h = 128;//模型的输入尺寸
    float mean = 127.0; float norm = 1.0/128.0;
    float threshold = 0.5; 
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
    ObjBbox_t target_box;
    generate_target_box(target_box, priors, (float*)output_score_Addr, (float*)output_boxes_Addr, threshold, 128, 128);
    
    printf("%f %f %f %f %f\r\n",target_box.x1 * scale_w, target_box.y1*scale_h,target_box.x2*scale_w,target_box.y2*scale_h,target_box.score);
    
    float x1 = target_box.x1 * scale_w;
    float y1 = target_box.y1 * scale_h;
    float x2 = target_box.x2 * scale_w;
    float y2 = target_box.y2 * scale_h;
    unsigned char* crop_img = crop_pad_square(org_img->item, x1, y1, x2, y2, org_img->w, org_img->h);
    int h = y2 - y1 + 1;
    int w = x2 - x1 + 1;
    int s = (h >= w) ? h : w;
    det_face_img->n = 1;
    det_face_img->c = 1;
    det_face_img->h = s;
    det_face_img->w = s;
    det_face_img->item = crop_img;
    det_face_img->stride = s;
    return 0;
}

#if defined(__cplusplus)
extern "C"
#endif
int face_align(dl_matrix3du_t* det_face_img, dl_matrix3du_t* face_align_img)
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
    float landmark[10];
    //后处理获得landmark
    decode_landmark(landmark, (float*)output_probe_Addr, (float*)output_probe_x_Addr, (float*)output_probe_y_Addr);
    trans_coords(landmark, scale);
    //仿射变换对齐人脸
    uint8_t* align_img = (uint8_t*)malloc(square * square);
    if(align_face(det_face_img->item, square, align_img, square, landmark) != 0)
    {
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

#if defined(__cplusplus)
extern "C"
#endif
int face_recong(dl_matrix3du_t* face_align_img, dl_matrix3d_t* face_vector)
{
    int in_w = 56; int in_h = 56;//模型的输入尺寸,方形
    float mean = 127.5; float norm = 1.0/128.0;
    int out_size = 64;
    pre_process_resize_norm(face_align_img->item,          face_align_img->w, face_align_img->h,
                            (float*)input_facerecong_Addr, in_w ,      in_h, 
                            mean, norm);
    //savefile("facerecong_input.bin", input_facerecong_Addr, 12544);
    int errCode = facerecong(facerecong_constantWeight, facerecong_mutableWeight, facerecong_activations);
    if (errCode != GLOW_SUCCESS)
    {
        printf("FaceRecong: Error running bundle: error code %d\n", errCode);
        return -1;
    }
    float* vector = (float*)malloc(out_size * sizeof(float));
    memcpy(vector, output_facerecong_Addr, out_size * sizeof(float));
    face_vector->n = 1;
    face_vector->c = 1;
    face_vector->h = 1;
    face_vector->w = out_size;
    face_vector->item = vector;
    face_vector->stride = out_size;
    
    return 0;
}
