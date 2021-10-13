
#include "handle_header.h"

int deep_learning_model_init(FaceHandle_t* handle);

int generate_face_id(FaceHandle_t* handle,
                            vector<shared_ptr<HiTensor>>& org_imgs, 
                            vector<shared_ptr<HiTensor>>& face_ids);