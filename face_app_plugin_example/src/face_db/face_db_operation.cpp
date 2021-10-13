 #include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include "sail_face.h"

 

static int disable_face_id(FaceHandle_t* handle, const char* key)
{

    printf("disable_face_id error, not found key!\r\n");
    return -1;
}

static int add_face_id(FaceHandle_t* handle, const char* key, array<float, 64> &id)
{

}

static int search_face_id(FaceHandle_t* handle, array<float, 64> &id)
{

}


