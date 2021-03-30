#ifndef __OP_NET_EXECUTOR_H__
#define __OP_NET_EXECUTOR_H__

#include "executor.h"

class NetExecutor : public Executor
{
private:
    void* hdr_ = nullptr;
public:
    virtual int Init(string configName);

    virtual int Exec(vector<shared_ptr<HiTensor>> &input_vec,
                     vector<shared_ptr<OutputResult>> &output_vec);

    virtual int Destory();
};

extern "C"  Executor* CreateNetExecutor();

#endif