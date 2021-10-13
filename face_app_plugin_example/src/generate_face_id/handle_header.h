
#ifndef _SAILFACE_CPP_HANDLE_H_
#define _SAILFACE_CPP_HANDLE_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "executor.h"

using namespace std;

typedef struct
{
    int w;      /*!< Width */
    int h;      /*!< Height */
    int c;      /*!< Channel */
    int n;      /*!< Number of filter, input and output must be 1 */
    int stride; /*!< Step between lines */
    unsigned char *data; /*!< Data */
} dl_matrix3du_t;

typedef struct ModelHandle {
    Executor* executor   = NULL;
    string   config_file;
    string   lib_file;
} ModelHandle_t;

typedef struct FaceHandle {
    float    reg_th_hold     = 0.45;//人脸识别门限
    float    det_th_hold     = 0.3;//人脸检测门限
    int      debug_level     = 0;//0 is off
    ModelHandle_t* face_detect;
    ModelHandle_t* face_align;
    ModelHandle_t* face_recognize;
    char*    db_file_path;
    int      default_db_item_size;//没有db文件存在的时候,初始化使用
} FaceHandle_t;

#endif //_SAILFACE_CPP_INNER_H_
