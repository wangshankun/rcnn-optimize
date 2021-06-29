#ifndef SAMPLE_DS_C_H_
#define SAMPLE_DS_C_H_

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>

using namespace std;

typedef struct DbItem {
    bool  flag    = false;
    char  key[64] = {};
    float id[64]  = {};
} DbItem_t;

typedef struct FaceHandle {
    uint8_t* db_data         = NULL;
    char*    db_path         = NULL;
    int      db_bytes        = -1;//初始化为负数
    float    reg_th_hold     = 0.45;//人脸识别门限
    float    det_th_hold     = 0.3;//人脸检测门限
} FaceHandle_t;

static int _write_db_file_content(char *filepath, uint8_t* buffer, int count)
{
    FILE *out = fopen(filepath, "wb");
    if(out != NULL)
    {
        size_t wc = fwrite(buffer , sizeof(char), count, out);
        fclose(out);
        if(wc != count)
        {
            printf("Write count is incorrect!\r\n");
            return -1;
        }
        return 0;
    }
    else
    {
        printf("Write db file content failed!\r\n");
        return -1;
    }
}

static uint8_t* _read_db_file_content(char *filepath, int* size)
{
    uint8_t* content = NULL;
    int file_content_size = 0;//数据库用在小型设备中,size用int表示最大2GByte
    FILE *fp = NULL;
    if(filepath == NULL)
    {
        *size = -1;
        return NULL;
    }

    fp = fopen(filepath, "rb+");
    if(fp == NULL)//文件不存在
    {
        fp = fopen(filepath, "w+");//创建新文件
        if(fp == NULL)
        {
            printf("Cannot create file content: %s\n", filepath);
            *size = -1;
            return NULL;
        }
        else
        {
            printf("Create new db file: %s\n", filepath);
            *size = 0;
            return NULL;
        }
    }

    fseek(fp, 0L, SEEK_END);
    file_content_size = ftell(fp);
    if(file_content_size <= 0)
    {
        printf("file is empty: %s\n", filepath);
        fclose(fp);
        *size = 0;
        return NULL;
    }

    content = (uint8_t *)malloc(file_content_size + 1);
    if(content == NULL)
    {
        printf("malloc error.\n");
        fclose(fp);
        *size = -1;
        return NULL;
    }
    memset(content, 0, file_content_size + 1);
    fseek(fp, 0L, SEEK_SET);
    fread(content, file_content_size, 1, fp);

    fclose(fp);
    //printf("file len:  %d\n", file_content_size);
    if(file_content_size < sizeof(DbItem_t))//如果有内容，只少存储一个item
    {
        printf("file content error.\n");
        *size = -1;
        return NULL;
    }
    *size = file_content_size;
    return content;
}

static bool check_input_handle(FaceHandle_t* handle)
{
    if(handle == NULL)
    {
        printf("FaceHandle is not init!\r\n");
        return false;
    }
    if(handle->db_bytes == -1)
    {
        printf("DataBase is not init!\r\n");
        return false;
    }
    return true;
}
static bool check_input_img(dl_matrix3du_t* intput_img)
{
    if(intput_img == NULL)
    {
        printf("Input img is not init!\r\n");
        return false;
    }
    if(intput_img->item == NULL)
    {
        printf("Input img is empty!\r\n");
        return false;
    }
    if(intput_img->c != 1 )//输入是灰度图
    {
        printf("Input img formal error!\r\n");
        return false;
    }
    return true;
}

static int disable_face_id(FaceHandle_t* handle, const char* key)
{
    if(handle->db_data == NULL || handle->db_bytes < sizeof(DbItem_t))
    {
        return -1;
    }

    int count = handle->db_bytes / sizeof(DbItem_t);
    DbItem_t* head = (DbItem_t*)(handle->db_data);
    for (int i = 0; i < count; i++)//已经存在就修改
    {
        if(head->flag)//只在还有效的item里面寻找
        {
            if (strcmp(head->key, key) == 0)
            {
                head->flag = false;
                int ret = _write_db_file_content(handle->db_path, handle->db_data, handle->db_bytes);
                if(ret == 0)
                {
                    return 0;
                }
                else
                {
                    head->flag = true;//写失败内存数据复原
                    return -1;
                }
            }
        }
        head = head + 1;//结构体指针+1
    }
    return 0;
}
static int add_face_id(FaceHandle_t* handle, const char* key, dl_matrix3d_t* id)
{
    if(handle->db_data == NULL && handle->db_bytes == 0)//数据库为空的状态
    {   //新增第一条空间
        handle->db_data  = (uint8_t*)malloc(sizeof(DbItem_t));
        handle->db_bytes = sizeof(DbItem_t);

        DbItem_t* head = (DbItem_t*)(handle->db_data);
        head->flag = true;
        memcpy(head->key, key, strlen(key) + 1);
        memcpy(head->id, id->item, 64 * sizeof(float));

        int ret = _write_db_file_content(handle->db_path, handle->db_data, handle->db_bytes);
        if(ret == 0)
        {
            return 0;
        }
        else
        {
            //写失败内存数据复原
            free(handle->db_data); handle->db_data = NULL;
            handle->db_bytes = 0;
            return -1;
        }
    }

    int count = handle->db_bytes / sizeof(DbItem_t);
    DbItem_t* head = (DbItem_t*)(handle->db_data);
    for (int i = 0; i < count; i++)//先查找如果已经存在就修改
    {
        if (strcmp(head->key, key) == 0)
        {
            head->flag = true;
            memcpy(head->id, id->item, 64 * sizeof(float));

            _write_db_file_content(handle->db_path, handle->db_data, handle->db_bytes);
            return 0;
        }
        head = head + 1;//结构体指针+1
    }

    //尾部追加一条
    void* new_ptr = realloc(handle->db_data, (count + 1) * sizeof(DbItem_t) );
    if(new_ptr == NULL)
    {
        printf("Realloc Failed!\r\n");
        return -1;
    }
    handle->db_data = (uint8_t*)new_ptr;
    head = (DbItem_t*)(handle->db_data);
    head = head + count;//指针移到末尾Item的前面
    head->flag = true;
    memcpy(head->key, key, strlen(key) + 1);
    memcpy(head->id,  id->item,    64 * sizeof(float));
    handle->db_bytes = (count + 1) * sizeof(DbItem_t);

    _write_db_file_content(handle->db_path, handle->db_data, handle->db_bytes);
    return 0;
}

static bool search_face_id(FaceHandle_t* handle, dl_matrix3d_t* check_id)
{
    if(handle->db_data == NULL || handle->db_bytes < sizeof(DbItem_t))
    {
        return false;
    }

    int count = handle->db_bytes / sizeof(DbItem_t);
    DbItem_t* head = (DbItem_t*)(handle->db_data);
    dl_matrix3d_t id_1; id_1.item = NULL; id_1.c = 64; 
    for (int i = 0; i < count; i++)
    {
        //printf("flag:%d  key:%s value:%f;%f;%f;%f....%f;%f;%f;%f\r\n",head->flag, head->key,
        //head->id[0],head->id[1],head->id[2],head->id[3],
        //head->id[60],head->id[61],head->id[62],head->id[63]);
        if(head->flag)
        {
            //通过数据结构偏移找到数据库向量
            id_1.item = head->id;
            float score = cos_distance(&id_1, check_id);

            printf("Key:%s RegScore:%.4f\r\n",head->key, score);
            if(score >= handle->reg_th_hold)
            {
                return true;//EEEEEEEEEEEEEEEEEEEError全量循环测试使用
            }
        }
        head = head + 1;//结构体指针+1
    }
    return false;
}

#endif
