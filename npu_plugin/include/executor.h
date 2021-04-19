#ifndef __OP_EXECUTOR_H__
#define __OP_EXECUTOR_H__

#include <stdint.h>
#include <vector>
#include <string>
#include <memory>

using namespace std;

typedef enum {
    PIXEL_FORMAT_GRAY8              = 0,
    PIXEL_FORMAT_BGR24_PACKAGE      = 1,
    PIXEL_FORMAT_BGR24_PLANNER      = 2,
    PIXEL_FORMAT_RGB24_PACKAGE      = 3,
    PIXEL_FORMAT_RGB24_PLANNER      = 4,
    PIXEL_FORMAT_I420               = 5, // yuv420 planar
    PIXEL_FORMAT_YV12               = 6, // yvu420 planar
    PIXEL_FORMAT_NV12               = 7, // yuv420 semi-planar
    PIXEL_FORMAT_NV21               = 8, // yvu420 semi-planar
    PIXEL_FORMAT_YUYV               = 9, // yuyv422 packed
    PIXEL_FORMAT_ABGR_8888          = 10,
    PIXEL_FORMAT_ARGB_8888          = 11,
} PixelFormat_t;

typedef enum {
    HI_ACL_DT_UNDEFINED = -1,
    HI_ACL_FLOAT = 0,
    HI_ACL_FLOAT16 = 1,
    HI_ACL_INT8 = 2,
    HI_ACL_INT32 = 3,
    HI_ACL_UINT8 = 4,
    HI_ACL_INT16 = 6,
    HI_ACL_UINT16 = 7,
    HI_ACL_UINT32 = 8,
    HI_ACL_INT64 = 9,
    HI_ACL_UINT64 = 10,
    HI_ACL_DOUBLE = 11,
    HI_ACL_BOOL = 12,
} DataType_t;

typedef enum {
    HI_ACL_FORMAT_UNDEFINED = -1,
    HI_ACL_FORMAT_NCHW = 0,
    HI_ACL_FORMAT_NHWC = 1,
    HI_ACL_FORMAT_ND = 2,
    HI_ACL_FORMAT_NC1HWC0 = 3,
    HI_ACL_FORMAT_FRACTAL_Z = 4,
    HI_ACL_FORMAT_MEMORY = 5,//二进制内存
    HI_ACL_FORMAT_FRACTAL_NZ = 29,
} DataFormat_t;

typedef struct _DetBox{
    int       clsn;//分类
    float     score;
    float     id;//跟踪id
    float     x0;
    float     y0;
    float     x1;
    float     y1;
} DetBox;

typedef struct _HiTensor {
    //========支持传统tensor=============
    uint32_t      n;//支持batch
    uint32_t      c;
    uint32_t      h;
    uint32_t      w;
    DataType_t    data_type;
    DataFormat_t  data_format;
    //=========支持内存图片==============
    PixelFormat_t img_type;

    //=========支持文件(图片/二进制)=====
    string   file_name;

    uint32_t      len;
    uint8_t*      data;
    
    //===========输入/出名称=============
    string        tensor_name;

    //===========extern信息=============
    uint64_t      request_id;//请求id
    uint64_t      time_stamp;//时间戳
    uint64_t      frame_index;//帧序
    uint32_t      img_stride;
    int           video_channel;
} HiTensor;

typedef struct _OutputResult {
    //======支持检测框=========
    vector<DetBox>     boxes;
    //=======支持复合tensor=====
    vector<HiTensor>   out_tensors;
} OutputResult;

class Executor
{
  public:
    Executor() {}
    virtual ~Executor() {}

    virtual int Init(string configName) = 0;

    virtual int Exec(vector<shared_ptr<HiTensor>> &input_vec,
                     vector<shared_ptr<OutputResult>> &output_vec) = 0;

    virtual int Destory() = 0;
};


typedef Executor* (*CreatorExec)();


//#define DEBUG_PRINT 0

#ifdef DEBUG_PRINT
#define debug_print(fmt, ...) \
        do { fprintf(stderr, "%s:%d:%s(): " fmt "\r\n", \
             __FILE__, __LINE__, __func__, ##__VA_ARGS__); } while (0)
#else
#define debug_print(fmt, ...) do {} while (0)
#endif

#endif