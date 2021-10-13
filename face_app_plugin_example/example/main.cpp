#include "sail_face.h"

int main()
{
    FaceHandle_t face_handle;
    face_handle.reg_th_hold  = 0.60;
    face_handle.det_th_hold  = 0.35;
    face_handle.db_file_path = (char*)"test.db";
    face_handle.default_db_item_size = 1000;
    if(Init(&face_handle) != 0)
    {
        printf("Init faild! \r\n");
        return -1;
    }

    dl_matrix3du_t base_img;
    base_img.n = 1;
    base_img.c = 3;
    base_img.h = 224;
    base_img.w = 224;
    base_img.data = (uint8_t*)malloc(3*224*224);
    base_img.stride = base_img.w;

    if(AddOneItem(&face_handle, "test_id_0x111", &base_img) != 0)
    {
        printf("AddOneItem Error\r\n");
    }

    return 0;
}
