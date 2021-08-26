#ifndef SAIL_FACE_WATCH_C_H_
#define SAIL_FACE_WATCH_C_H_

#include <stdio.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {

typedef struct FaceHandle {
    uint8_t* db_data         = NULL;
    char*    db_path         = NULL;
    int      db_bytes        = -1;//初始化为负数
    float    reg_th_hold     = 0.75;
    float    det_th_hold     = 0.5;
} FaceHandle_t;

typedef struct
{
    int w;      /*!< Width */
    int h;      /*!< Height */
    int c;      /*!< Channel */
    int n;      /*!< Number of filter, input and output must be 1 */
    int stride; /*!< Step between lines */
    unsigned char *item; /*!< Data */
} dl_matrix3du_t;

int  Init(FaceHandle_t* handle);
int  Destory(FaceHandle_t* handle);
int  AddOneItem(FaceHandle_t* handle, const char* key, dl_matrix3du_t* input_img);
int  DisableOneItem(FaceHandle_t* handle, const char* key);
bool SearchOneItem(FaceHandle_t* handle, dl_matrix3du_t* intput_img);

}
#endif//defined(__cplusplus)
#endif//SAIL_FACE_WATCH_C_H_