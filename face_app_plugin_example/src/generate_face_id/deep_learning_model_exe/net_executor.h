#ifndef __OP_NET_EXECUTOR_H__
#define __OP_NET_EXECUTOR_H__

#include "executor.h"

class NetExecutor : public Executor
{
private:
    void* hdr_ = nullptr;
public:
    virtual int Init(string &config_file);

    virtual int Config(std::map<string, string>&config_data);

    virtual int Exec(vector<shared_ptr<HiTensor>> &input_vec,
                     vector<shared_ptr<HiTensor>> &output_vec);

    virtual int Destory();
};

extern "C"  Executor* CreateNetExecutor();

#endif