  
#include "generate_face_id.h"
#include <dlfcn.h>

static int model_init(ModelHandle_t* model)
{
    string config_path = model->config_file;
    string runlib_path = model->lib_file;
    void *handle = dlopen(runlib_path.c_str(), RTLD_LAZY);
    if(handle == NULL) {
        printf("error dlopen - %s \r\n", dlerror());
        return -1;
    }
    char * dl_error;
    CreatorExec create = (CreatorExec)dlsym(handle, "CreateNetExecutor");
    if((dl_error = dlerror()) != NULL) {
        printf("find sym error %s \r\n", dl_error);
            return -1;
    }
    model->executor = (*create)();
    if(model->executor != nullptr)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

int deep_learning_model_init(FaceHandle_t* handle)
{
    //**************************dummy**************************
     return 0;

    if ( model_init(handle->face_detect) == 0   &&
         model_init(handle->face_align) == 0    &&
         model_init(handle->face_recognize) == 0 )
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

/*基础步骤
   1.检测
   2.对齐
   3.识别
*/

int generate_face_id(FaceHandle_t* handle,
                            vector<shared_ptr<HiTensor>>& org_imgs, 
                            vector<shared_ptr<HiTensor>>& face_ids)
{
    //**************************dummy**************************
    return 0;

    int ret = -1;

    vector<shared_ptr<HiTensor>> det_face_imgs;
    ret = handle->face_detect->executor->Exec(org_imgs, det_face_imgs);
    if( 0 != ret )
    {
        printf("Face Det Error! \r\n");
        return -1;
    }

    vector<shared_ptr<HiTensor>> face_align_imgs;
    ret = handle->face_align->executor->Exec(det_face_imgs, face_align_imgs);
    if( 0 != ret )
    {
        printf("Face Align Error! \r\n");
        return -1;
    }

    ret = handle->face_recognize->executor->Exec(face_align_imgs, face_ids);
    if( 0 != ret )
    {
        printf("Face Recong Error! \r\n");
        return -1;
    }
    //savefile("facerecong_output.bin", face_vector_p->item, face_vector_p->w * sizeof(float));
    return 0;
}
