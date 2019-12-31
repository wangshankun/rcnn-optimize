#include <iostream>
#include <string>
#include "nvJPEG.hpp"
#include "nvJPEG_helper.hpp"
#include<time.h>

typedef  std::vector<char>  FileData;

int main()
{
    //将jpeg图片做二进制文件读入内存;
    FileData raw_data;
    std::ifstream input("0001.jpg",std::ios::in | std::ios::binary | std::ios::ate);
    if (!(input.is_open()))
    {
      std::cerr << "Cannot open image: ";
    }
    // Get the size
    std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    // resize if buffer is too small
    if (raw_data.size() < file_size)
    {
      raw_data.resize(file_size);
    }

    if (!input.read(raw_data.data(), file_size)) 
    {
      std::cerr << "Cannot read from file: ";
    }
    

    NvJpeg jpeg_dec(3,1920,1080,0,1,NVJPEG_OUTPUT_RGBI); 

    unsigned char * gpu_rgb_buf;
    checkCudaErrors(cudaMalloc(&gpu_rgb_buf, 1920*1080*3));

    //warm up
    jpeg_dec.decompress_to_gpubuf(raw_data, file_size, gpu_rgb_buf);

    double elapsed;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    //jpeg_dec.decompress_to_file(raw_data, file_size, "./test.bmp");
    jpeg_dec.decompress_to_gpubuf(raw_data, file_size, gpu_rgb_buf);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("elapsed time:%f\r\n",elapsed);

    return EXIT_SUCCESS;
}
