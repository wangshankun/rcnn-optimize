#include "nvJPEG.hpp" 
#include "nvJPEG_helper.hpp" 

int dev_malloc(void **p, size_t s)
{ 
    return (int)cudaMalloc(p, s); 
}
int dev_free(void *p)
{ 
    return (int)cudaFree(p); 
}

NvJpeg::NvJpeg(int channel = 3, int width = 1920, int height = 1080, int device_id = 0, int batch = 1, nvjpegOutputFormat_t fmt = NVJPEG_OUTPUT_RGBI)
{

   _channel    = channel;
   _width      = width;
   _height     = height;
   _device_id  = device_id;
   _batch      = batch;
   _fmt        = fmt;

    _dev_allocator = {&dev_malloc, &dev_free};

    checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &_dev_allocator, &_nvjpeg_handle));
    checkCudaErrors(nvjpegJpegStateCreate(_nvjpeg_handle, &_nvjpeg_state));
    checkCudaErrors(nvjpegDecodeBatchedInitialize(_nvjpeg_handle, _nvjpeg_state, _batch, 1, _fmt));
        
    int devID = gpuDeviceInit(_device_id);
    if (devID < 0)
    {
        printf("exiting...\n");
        exit(EXIT_FAILURE);
    }

    _iout.pitch[0] = _channel * _width;
    checkCudaErrors(cudaMalloc(&_iout.channel[0], _height * _width * _channel));
    _vchanRGB = new std::vector<unsigned char>(_height * _width * _channel);
    checkCudaErrors(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
}

NvJpeg::~NvJpeg()
{
    checkCudaErrors(cudaStreamDestroy(_stream));
    checkCudaErrors(cudaFree(_iout.channel[0]));
    checkCudaErrors(nvjpegJpegStateDestroy(_nvjpeg_state));
    checkCudaErrors(nvjpegDestroy(_nvjpeg_handle));
    delete _vchanRGB;
}

int NvJpeg::decompress_to_hostbuf(std::vector<char> &jpgBuffer, const uint32_t jpgSize, std::vector<unsigned char> **bmp)
{

    checkCudaErrors(cudaStreamSynchronize(_stream));
    checkCudaErrors(nvjpegDecode(_nvjpeg_handle, _nvjpeg_state,
                                     (const unsigned char *)jpgBuffer.data(),
                                     jpgSize, _fmt, &_iout,
                                     _stream));
    checkCudaErrors(cudaStreamSynchronize(_stream));
                         
    unsigned char* chanRGB = (*_vchanRGB).data();
    checkCudaErrors(cudaMemcpy2D(chanRGB, (size_t)_width * _channel, _iout.channel[0], (size_t)_iout.pitch[0],
                               _width * _channel, _height, cudaMemcpyDeviceToHost));
    *bmp = _vchanRGB;
    return EXIT_SUCCESS;
}

int NvJpeg::decompress_to_file(std::vector<char> &jpgBuffer, const uint32_t jpgSize, char* filename)
{

    checkCudaErrors(cudaStreamSynchronize(_stream));
    checkCudaErrors(nvjpegDecode(_nvjpeg_handle, _nvjpeg_state,
                                     (const unsigned char *)jpgBuffer.data(),
                                     jpgSize, _fmt, &_iout,
                                     _stream));
    checkCudaErrors(cudaStreamSynchronize(_stream));
                         
    unsigned char* chanRGB = (*_vchanRGB).data();
    checkCudaErrors(cudaMemcpy2D(chanRGB, (size_t)_width * _channel, _iout.channel[0], (size_t)_iout.pitch[0],
                               _width * _channel, _height, cudaMemcpyDeviceToHost));

    writeBMPi_test(filename, _vchanRGB, (size_t)_iout.pitch[0], _width, _height);

    return EXIT_SUCCESS;
}

int NvJpeg::decompress_to_gpubuf(std::vector<char> &jpgBuffer, const uint32_t jpgSize, unsigned char* gpubuf)
{

    checkCudaErrors(cudaStreamSynchronize(_stream));
    checkCudaErrors(nvjpegDecode(_nvjpeg_handle, _nvjpeg_state,
                                 (const unsigned char *)jpgBuffer.data(),
                                 jpgSize, _fmt, &_iout,
                                 _stream));
    checkCudaErrors(cudaStreamSynchronize(_stream));

    checkCudaErrors(cudaMemcpy2D(gpubuf, (size_t)_width * _channel, _iout.channel[0], (size_t)_iout.pitch[0],
                               _width * _channel, _height, cudaMemcpyDeviceToDevice));

    return EXIT_SUCCESS;
}
