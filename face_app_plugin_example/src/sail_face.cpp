#include "hashmap_db.h"
#include "generate_face_id.h"

#include <algorithm>

class CInfer {
private:
    SimpleHash db_map;
public:
    CInfer();
    virtual int  Destory(FaceHandle_t* handle);
    virtual int  Init(FaceHandle_t* handle);
    virtual int  SetConfig(FaceHandle_t* handle);
    virtual int  Sync(FaceHandle_t* handle);
    virtual int  AddOneItem(FaceHandle_t* handle, const char* key, dl_matrix3du_t* input_img);
    virtual int  DisableOneItem(FaceHandle_t* handle, const char* key);
    virtual int  SearchOneItem(FaceHandle_t* handle, dl_matrix3du_t* intput_img);
};

CInfer* c_infer = new CInfer();

#if defined(__cplusplus)
extern "C" {
 
int Destory(FaceHandle_t* handle)
{
    c_infer->Destory(handle);
    return 0;
}

int Init(FaceHandle_t* handle)
{
    if(handle == NULL)
    {
        printf("FaceHandle is not init!\r\n");
        return -1;
    }
    //debug_level = handle->debug_level;
    int ret = c_infer->Init(handle);
    if(ret != 0) return ret;

    return 0;
}

int SetConfig(FaceHandle_t* handle)
{
	//动态配置,修改参数
    c_infer->SetConfig(handle);
    return 0;
}

int Sync(FaceHandle_t* handle)
{
	//同步到库中
    c_infer->Sync(handle);
    return 0;
}

int AddOneItem(FaceHandle_t* handle, const char* key, dl_matrix3du_t* input_img)
{

    c_infer->AddOneItem(handle, key, input_img);
    return 0;
}

int DisableOneItem(FaceHandle_t* handle, const char* key)
{
    if(handle == NULL)
    {
        printf("Error, handle is not initialized! \r\n");
        return -1;
    }
    if(strlen(key) >= 64 )//key小于64个字符
    {
        printf("Error, the key's len out of db range! \r\n");
        return -1;
    }
    c_infer->DisableOneItem(handle, key);
	return 0;
}

int SearchOneItem(FaceHandle_t* handle, dl_matrix3du_t* intput_img)
{
    c_infer->SearchOneItem(handle, intput_img);
    return 0;
}
}
#endif


//
//******************接口实现******************
//
CInfer::CInfer() {}
int CInfer::SetConfig(FaceHandle_t* handle) 
{
    return 0;
}

int CInfer::Init(FaceHandle_t* handle)
{
    if( 0 != deep_learning_model_init(handle))//初始化模型
    {
        return -1;
    }

    if( -1 == db_map.loadfile(handle->db_file_path))//不存在就创建
    {
        SimpleHash tmp_map(handle->default_db_item_size * (1.0/0.7));//1.42倍的hash容量
        db_map = tmp_map;
        db_map.tofile(handle->db_file_path);
    }
    return 0;
}

int CInfer::Destory(FaceHandle_t* handle)
{
    return 0;
}

int CInfer::Sync(FaceHandle_t* handle)
{
    db_map.tofile(handle->db_file_path);
}

int CInfer::AddOneItem(FaceHandle_t* handle, const char* key, dl_matrix3du_t* input_img)
{
    vector<shared_ptr<HiTensor>> input_vec;
    vector<shared_ptr<HiTensor>> output_vec;
    shared_ptr<HiTensor> input = shared_ptr<HiTensor>(new HiTensor);
    input->img_type = PIXEL_FORMAT_RGB24_PACKAGE;
    input->n = input_img->n;
    input->w = input_img->w;
    input->h = input_img->h;
    input->c = input_img->c;
    input->data = input_img->data;//图片data
    input_vec.push_back(input);

    //把input_img转成tensor
    int ret = generate_face_id(handle, input_vec, output_vec);
    if (ret != 0) return -1;
    //把face_ids转成float array

    array<float, 64> value;
    //std::copy_n(output_vec[0]->data, value.size(), value.begin());
    
    string key_str(key);
    ret = db_map.insert(key_str, value);
    if (ret != 0) return -1;

    return 0;
}

int CInfer::SearchOneItem(FaceHandle_t* handle, dl_matrix3du_t* intput_img)
{
    //检测input
	//根据input，生成 face id; generate_face_id
	//根据生成的face id 去搜索search_face_id
	//命中返回0.失败返回-1
    return 0;
}

int CInfer::DisableOneItem(FaceHandle_t* handle, const char* key)
{
    return 0;
}
