#ifndef __OP_EXECUTOR_H__
#define __OP_EXECUTOR_H__

#include <stdint.h>
#include <vector>
#include <string>
#include <memory>
#include <map> 

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
    DT_UNDEFINED = -1,
    FLOAT = 0,
    FLOAT16 = 1,
    INT8 = 2,
    INT32 = 3,
    UINT8 = 4,
    INT16 = 6,
    UINT16 = 7,
    UINT32 = 8,
    INT64 = 9,
    UINT64 = 10,
    DOUBLE = 11,
    BOOL = 12,
} DataType_t;

typedef enum {
    FORMAT_UNDEFINED = -1,
    FORMAT_NCHW = 0,
    FORMAT_NHWC = 1,
    FORMAT_ND = 2,
    FORMAT_NC1HWC0 = 3,
    FORMAT_FRACTAL_Z = 4,
    FORMAT_MEMORY = 5,//二进制内存
    FORMAT_FRACTAL_NZ = 29,
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
    uint32_t      n = 1;//支持batch
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
    
    //===========输入/出 名称=======
    string        tensor_name;
    //======检测框输出=========
    vector<DetBox>     out_boxes;
    //======检测框输入(roi)=========
    vector<DetBox>     in_boxes;
    //===========extern 视频信息=============
    void*         extern_info;
    uint64_t      request_id;//请求id
    uint64_t      time_stamp;//时间戳
    uint64_t      frame_index;//帧序
    uint32_t      img_stride;
    int           video_channel;
} HiTensor;

class Executor
{
  public:
    Executor() {}
    virtual ~Executor() {}

    virtual int Init(string &config_file) = 0;//Init 设备初始化、鉴权、模型加载等;

    virtual int Config(std::map<string, string>&config_data) = 0;//动态配置使用:可变阈值，可变尺寸等;

    virtual int Exec(vector<shared_ptr<HiTensor>> &input_vec,
                     vector<shared_ptr<HiTensor>> &output_vec) = 0;

    virtual int Destory() = 0;
};


typedef Executor* (*CreatorExec)();


#define DEBUG_PRINT 1

#ifdef DEBUG_PRINT
#define debug_print(fmt, ...) \
        do { fprintf(stderr, "%s:%d:%s(): " fmt "\r\n", \
             __FILE__, __LINE__, __func__, ##__VA_ARGS__); } while (0)
#else
#define debug_print(fmt, ...) do {} while (0)
#endif

#endif