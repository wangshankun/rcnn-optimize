#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <thread>
#include <cuda.h>
#include <npp.h>
#include <nppi.h>
#include <nppdefs.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include <nvjpeg.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nppi_geometry_transforms.h>
#include <nppi_support_functions.h>
#include "NvDecoder.h"
#include "NvCodecUtils.h"
#include "FFmpegDemuxer.h"

using namespace std;

#define CUDA_FRAME_ALIGNMENT     256.0
#define NVJPEG_MAX_COMPONENT     4
#define MAX_JPEG_ARRAY_BUF_SIZE  32*1024*1024
#define MAX_HIT_NUM              1024//一次最多解压的张数

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", file, line,
            static_cast<unsigned int>(result), func);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


int dev_malloc(void **p, size_t s)
{
    return (int)cudaMalloc(p, s);
}
int dev_free(void *p)
{
    return (int)cudaFree(p);
}

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)

#define readfile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "rb");\
  if(out != NULL)\
  {\
        fread (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)


class FFmpegBuf:public FFmpegDemuxer::DataProvider 
{
    public:
        uint8_t* _src_buf;
        int      _max_size;
        int      _has_read_size;
        
        int GetData(uint8_t *pBuf, int nBuf)
        {
            if (_has_read_size + nBuf <= _max_size)
            {
                memcpy(pBuf, _src_buf + _has_read_size, nBuf);
                _has_read_size = _has_read_size + nBuf;
                return nBuf;
            }
            else if (_has_read_size < _max_size)
            {
                int res_size =  _max_size - _has_read_size;
                memcpy(pBuf, _src_buf + _has_read_size, res_size);
                _has_read_size = _max_size;
                return res_size;
            }
            else
            {
                return -1;
            }
        }
        
        FFmpegBuf(uint8_t* src_buf, int max_size)
        {
            _src_buf       = src_buf;
            _max_size      = max_size;
            _has_read_size = 0;
        }
        
        ~FFmpegBuf()
        {
            _max_size = 0;
            free(_src_buf);
        }
};

extern "C" void decompress_2_jpeg(unsigned char *video_buf, int buf_size, int* hit_array, int hit_len, unsigned char** out_buf_arry, int** out_size_arry);
void decompress_2_jpeg(unsigned char *video_buf, int buf_size, int* hit_array, int hit_len, unsigned char** out_buf_arry, int** out_size_arry)
{
    //GPU初始化
    int _device_id = 0;
    ck(cuInit(_device_id));
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, _device_id));

    //nvDecoder配置
    FFmpegBuf      mem_buf(video_buf, buf_size);
    FFmpegDemuxer  buf_demuxer(&mem_buf);
    int _height  = buf_demuxer.GetHeight();
    int _width   = buf_demuxer.GetWidth();
    NvDecoder dec(cuContext, _width, _height, true, FFmpeg2NvCodecId(buf_demuxer.GetVideoCodec()));

    //nvjpeg配置
    nvjpegJpegState_t            _nvjpeg_state;
    nvjpegHandle_t               _nvjpeg_handle;
    nvjpegEncoderParams_t        _encode_params;
    nvjpegEncoderState_t         _encoder_state;

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &_nvjpeg_handle));
    checkCudaErrors(nvjpegJpegStateCreate(_nvjpeg_handle, &_nvjpeg_state));
    checkCudaErrors(nvjpegEncoderStateCreate(_nvjpeg_handle, &_encoder_state, NULL));
    checkCudaErrors(nvjpegEncoderParamsCreate(_nvjpeg_handle, &_encode_params, NULL));
    checkCudaErrors(nvjpegEncoderParamsSetQuality(_encode_params, 85, NULL));
    checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(_encode_params, 1, NULL));

    nvjpegImage_t                _imgdesc;
    unsigned char *              _pBuffer;
    unsigned int linesize = (int)(_width/CUDA_FRAME_ALIGNMENT + 1) * CUDA_FRAME_ALIGNMENT;
    cudaError_t eCopy = cudaMalloc(&_pBuffer, linesize * _height * NVJPEG_MAX_COMPONENT);
    if(cudaSuccess != eCopy)
    {
        fprintf(stderr, "cudaMalloc failed\r\n");
    }

    _imgdesc.channel[0] = _pBuffer;
    _imgdesc.channel[1] = _pBuffer + linesize * _height;
    _imgdesc.channel[2] = _pBuffer + linesize * _height * 2;
    _imgdesc.channel[3] = _pBuffer + linesize * _height * 3;
    _imgdesc.pitch[0]   = (unsigned int)(linesize);
    _imgdesc.pitch[1]   = (unsigned int)(linesize/2);
    _imgdesc.pitch[2]   = (unsigned int)(linesize/2);
    _imgdesc.pitch[3]   = (unsigned int)(linesize/2);

    //nppi配置
    NppiSize roi_size       = { _width,  _height };
    Npp8u *pYuv420pDst[3]   = {_imgdesc.channel[0],      _imgdesc.channel[1],      _imgdesc.channel[2]};
    int   pYuv420pPitch[3]  = {(int)(_imgdesc.pitch[0]), (int)(_imgdesc.pitch[1]), (int)(_imgdesc.pitch[2])};


    *out_buf_arry  = (unsigned char*)malloc(MAX_JPEG_ARRAY_BUF_SIZE);//由调用者使用完成后释放
    *out_size_arry = (int*)malloc(MAX_HIT_NUM * sizeof(int));

    av_log_set_level(0);

    int nVideoBytes = 0, nFrameReturned = 0;
    uint8_t *pVideo = NULL, **ppFrame;
    int frame_index = 0;
    int current_hit_index = 0;//hit index列表按照顺序从小到大存放
    int current_out_size  = 0;
    
    do 
    {
        buf_demuxer.Demux(&pVideo, &nVideoBytes);
        dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);//h265->nv12

        //if (!frame_index && nFrameReturned) LOG(INFO) << dec.GetVideoInfo();
        if (!frame_index && nFrameReturned);
        
        NppStatus result; 
        for (int i = 0; i < nFrameReturned; i++) 
        {
            if (frame_index == hit_array[current_hit_index])//将hit的帧保存为图片
            {
                Npp8u *pNV12Src[2] = {(Npp8u *)(ppFrame[i]),
                                      (Npp8u *)(ppFrame[i]) + (_width * _height) };

                result = nppiNV12ToYUV420_8u_P2P3R(pNV12Src,    dec.GetWidth(),      //nv12->yuv420p
                                                   pYuv420pDst, pYuv420pPitch,
                                                   roi_size);
                if (result != NPP_SUCCESS)
                {
                    fprintf(stderr, "Error executing NV12ToYUV420 -- code:%d \r\n", result);
                }

                nvjpegEncoderParamsSetSamplingFactors(_encode_params, NVJPEG_CSS_420, NULL);

                checkCudaErrors(nvjpegEncodeYUV(_nvjpeg_handle,   //yuv420p->jpeg
                    _encoder_state,
                    _encode_params,
                    &_imgdesc,
                    NVJPEG_CSS_420,
                    _width,
                    _height,
                    NULL));

                //std::vector<unsigned char> obuffer;
                size_t length;
                checkCudaErrors(nvjpegEncodeRetrieveBitstream(
                    _nvjpeg_handle,
                    _encoder_state,
                    NULL,
                    &length,
                    NULL));
                //obuffer.resize(length);
                if (length + current_out_size > MAX_JPEG_ARRAY_BUF_SIZE)
                {
                    fprintf(stderr, "JPEG_ARRAY_BUF Buf OverFlow \r\n");
                    break;
                }
                checkCudaErrors(nvjpegEncodeRetrieveBitstream(//get jpeg to host mem
                    _nvjpeg_handle,
                    _encoder_state,
                    //obuffer.data(),
                    ((*out_buf_arry) + current_out_size),
                    &length,
                    NULL));
                    
                //outputFile.write(reinterpret_cast<const char *>(obuffer.data()), static_cast<int>(length));
                //memcpy((*out_buf_arry) + current_out_size, reinterpret_cast<const char *>(obuffer.data()), static_cast<int>(length));
                current_out_size                     =  current_out_size  + static_cast<int>(length);
                (*out_size_arry)[current_hit_index]  =  static_cast<int>(length);
                current_hit_index                    =  current_hit_index + 1;
            }

            frame_index = frame_index + 1;
        }
    } while (nVideoBytes && (current_hit_index != hit_len));

    cudaFree(_pBuffer);
}
