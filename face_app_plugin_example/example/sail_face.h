#ifndef _SAILFACE_C_H_
#define _SAILFACE_C_H_

#include <stdio.h>
#include <stdlib.h>
#include <string>

typedef struct
{
    int w;      /*!< Width */
    int h;      /*!< Height */
    int c;      /*!< Channel */
    int n;      /*!< Number of filter, input and output must be 1 */
    int stride; /*!< Step between lines */
    unsigned char *data; /*!< Data */
} dl_matrix3du_t;

typedef struct FaceHandle {
    float    reg_th_hold     = 0.45;//人脸识别门限
    float    det_th_hold     = 0.3;//人脸检测门限
    int      debug_level     = 0;//0 is off
    void*    face_detect;
    void*    face_align;
    void*    face_recognize;
    char*    db_file_path;
    int      default_db_item_size;//没有db文件存在的时候,初始化使用
} FaceHandle_t;


extern "C" {
    int Destory(FaceHandle_t* handle);
    int Init(FaceHandle_t* handle);
    int SetConfig(FaceHandle_t* handle);
    int AddOneItem(FaceHandle_t* handle, const char* key, dl_matrix3du_t* input_img);
    int DisableOneItem(FaceHandle_t* handle, const char* key);
    int SearchOneItem(FaceHandle_t* handle, dl_matrix3du_t* intput_img);
}
#endif //_SAILFACE_C_H_
