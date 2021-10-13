#ifndef _THIS_T31_0316_SSD_MODEL_EXE_BASE_H
#define _THIS_T31_0316_SSD_MODEL_EXE_BASE_H
#include "net_executor.h"

#include "jzdl/iaac.h"
#include "jzdl/net.h"
#include "jzdl/utils.h"

#include "ped_tracker_wrapper.hpp"
#include <tracker_utilities.h>

using namespace std;
using namespace jzdl;

#include "this_model_process.h"

class Net {
  public:
    ~Net()
    {
        jzdl::net_destory(base_net_);
        if(model_data_ != NULL)
        {
            free(model_data_);
            model_data_ = NULL;
        }
    }
    int set_model_data(unsigned char* model_data)
    {
        model_data_ = model_data;
    }
  public:
    float          threshold_;
    jzdl::BaseNet  *base_net_;
    fusion_tracking::PedTrackerHandle tracker_handle_;
    vector<vector<float>> priors_ = {};
    vector<vector<float>> featuremap_size_;
    int model_in_w_;
    int model_in_h_;
    int model_in_c_;
  private:
    unsigned char* model_data_ = NULL;    
};


#endif

