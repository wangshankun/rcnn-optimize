#include "model_process.h"
#include "sample_db.h"

#if defined(__cplusplus)
extern "C" {

int Destory(FaceHandle_t* handle)
{
    if(handle->db_data != NULL)
    {
        free(handle->db_data);
    }
    return 0;
}

int Init(FaceHandle_t* handle)
{
    if(handle == NULL)
    {
        printf("FaceHandle is not init!\r\n");
        return -1;
    }
    if(handle->db_path == NULL)
    {
        printf("Init Error: db_path is empty! \r\n");
        return -1;
    }

    int db_bytes = -1;
    uint8_t* db_data = _read_db_file_content(handle->db_path, &db_bytes);
    if(db_bytes == -1)
    {
        handle->db_data  = NULL;
        handle->db_bytes = -1;
        printf("Init Error: Create DB Failed!\r\n");
        return -1;
    }
    handle->db_data  = db_data;
    handle->db_bytes = db_bytes;
    return 0;
}

int AddOneItem(FaceHandle_t* handle, const char* key, dl_matrix3du_t* input_img)
{
    if(handle->db_bytes == -1)
    {
        return -1;
    }
    if(!check_input_img(input_img))
    {
        return -1;
    }
    if(strlen(key) >= 64)//key小于64个字符
    {
        printf("Error, the key's len out of db range! \r\n");
        return -1;
    }
    //输入图片输出vector
    dl_matrix3d_t face_id; face_id.item = NULL;
    if(generate_face_id(input_img, &face_id, handle->det_th_hold) != 0)
    //为了尽快释放input_img内存，input_img内存释放在generate_face_id里面做
    {
        if(face_id.item != NULL)
        {
            free(face_id.item); face_id.item == NULL;
        }
        printf("Error, generate_face_id failed! \r\n");
        return -1;
    }

     if(0 != add_face_id(handle, key, &face_id))
     {
        if(face_id.item != NULL)
        {
            free(face_id.item); face_id.item == NULL;
        } 
         return -1;
     }

    if(face_id.item != NULL)
    {
        free(face_id.item); face_id.item == NULL;
    } 
    return 0;
}

int DisableOneItem(FaceHandle_t* handle, const char* key)
{
    if(!check_input_handle(handle))
    {
        return -1;
    }
    if(strlen(key) >= 64 )//key小于64个字符
    {
        printf("Error, the key's len out of db range! \r\n");
        return -1;
    }
    return disable_face_id(handle, key);
}

bool SearchOneItem(FaceHandle_t* handle, dl_matrix3du_t* intput_img)
{
    if(!check_input_img(intput_img) || !check_input_handle(handle))
    {
        return false;
    }

    if(handle->db_bytes <=0)
    {
        return false;
    }

    dl_matrix3d_t face_id; face_id.item = NULL;
    if (generate_face_id(intput_img, &face_id, handle->det_th_hold) != 0)
    {
        if(face_id.item != NULL)
        {
            free(face_id.item);
        }
        return false;
    }

    bool ret = search_face_id(handle, &face_id);
    if(face_id.item != NULL)
    {
        free(face_id.item);
    }

    return ret;
}

}
#endif