#include <fstream>
#include <iostream>
  
#include "executor.h"
#include <mutex>
#include <stdio.h>
#include <string>
#include <vector>
#include <time.h>
#include <string.h>
#include <dlfcn.h>

Executor* net_executor_;

int main(int argc, char* argv[]) 
{
  if (argc < 3) 
  {
    std::cout << "./test model_file runlib_path" << std::endl;
    return -1;
  }

  //std::string runlib_path = config_.detection_config_.runlib_path;
  //std::string model_file  = config_.detection_config_.model_path;

  std::string runlib_path(argv[1]);
  std::string model_file(argv[2]);

  void *handle = dlopen(runlib_path.c_str(), RTLD_LAZY);
  if(handle == NULL)
  {
    debug_print("error dlopen - %s \r\n", dlerror());
    return false;
  }

  char * dl_error;
  CreatorExec create_net_executor = (CreatorExec)dlsym(handle, "CreateNetExecutor");
  if((dl_error = dlerror()) != NULL)
  {
    debug_print("find sym error %s \r\n", dl_error);
    return false;
  }

  net_executor_ = (*create_net_executor)();
  int ret = net_executor_->Init(model_file);
  if (ret != 0) 
  {
    debug_print("Failed to init net_executor_\r\n");
    return false;
  }

  vector<shared_ptr<HiTensor>> input_vec;
  vector<shared_ptr<OutputResult>> output_vec;
  shared_ptr<HiTensor> input = shared_ptr<HiTensor>(new HiTensor);
  input->img_type = PIXEL_FORMAT_RGB24_PACKAGE;
  input->n = 1;
  input->w = 1080;
  input->h = 720;
  input->c = 3;
  input->data = nullptr;//图片data
  input_vec.push_back(input);
  //=====================================执行=============================================
  net_executor_->Exec(input_vec, output_vec);
  //=====================================结果返回=============================================
  for (int i = 0; i < output_vec[0]->boxes.size(); ++i)
  {

  std::cout << output_vec[0]->boxes[i].score << "  "
            << output_vec[0]->boxes[i].clsn  << "  "
            << output_vec[0]->boxes[i].id    << "  "
            << output_vec[0]->boxes[i].x0    << "  "
            << output_vec[0]->boxes[i].y0    << "  "
            << output_vec[0]->boxes[i].x1    << "  " 
            << output_vec[0]->boxes[i].y1    << std::endl;

  //debug_print("%f %d %d %f %f %f %f\r\n", res.score, res.type, res.id, res.box.left,res.box.top,res.box.right,res.box.bottom);
  }

  return 0;
}
