#include "sail_face_watch.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)

#define readfile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "rb");\
  if(out != NULL)\
  {\
        fread (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)


int main(int argc, const char* argv[])
{
    if(argc < 1)
    {
        printf("./test img_file \r\n");
    }
    
    float threshold = 0.5;
    uint8_t *testpic = (uint8_t*)malloc(480*640*1);
    readfile(argv[1], testpic, 480*640*1); 
    
    /////////////////////////////////face检测模型/////////////////////////////////////////////////
    dl_matrix3du_t org_img, det_face_img, face_align_img;
    dl_matrix3d_t  face_vector;

    org_img.n = 1;
    org_img.c = 1;
    org_img.h = 640;
    org_img.w = 480;
    org_img.item = testpic;
    org_img.stride = org_img.w;
    
    if(face_det(&org_img, &det_face_img) != 0)
    {
        return -1;
    }
    free(org_img.item); org_img.item = NULL;

    if(face_align(&det_face_img, &face_align_img) != 0)
    {
        return -1;
    }
    free(det_face_img.item); det_face_img.item = NULL;
    
    if(face_recong(&face_align_img, &face_vector) != 0)
    {
        return -1;
    }
    free(face_align_img.item); face_align_img.item = NULL;
    
    savefile("facerecong_output.bin", face_vector.item, face_vector.w * sizeof(float));
    return 0;
}
