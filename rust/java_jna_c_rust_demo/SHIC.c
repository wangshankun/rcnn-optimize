#include "SHIC.h"
int  init(int enable_gpu, int gpu_id, int enable_cpu_bind, int* cpu_ids, int cpu_ids_len)
{
}


void compressListOfImage(CompressInputImage* input_images, int len, compress_callback callback)
{

    unsigned long ret_num = 0;
    CompressOutputData* cpdatas = compress_images(input_images, len, 16, 0, &ret_num);
    for (unsigned long i = 0; i < ret_num; i++)
    {
        printf("%s  %s   %s  %s  %d \r\n",cpdatas[i].channel_ids, cpdatas[i].image_ids, 
                                           cpdatas[i].ts_arrays, cpdatas[i].offsets, cpdatas[i].compressed_buf_len);
        print_hex_str(cpdatas[i].compressed_buf, 64);

        callback(1, &cpdatas[i]);
        //callback函数执行后(java接收到了结果),释放内存(这部分内存是从rust库中申请的)
        free(cpdatas[i].channel_ids);
        free(cpdatas[i].image_ids);
        free(cpdatas[i].ts_arrays);
        free(cpdatas[i].offsets);
        free(cpdatas[i].compressed_buf);
    }

}

void decompressListOfImage(char* buf, int len, const char* hit_img_ids, decompress_callback callback)
{
    unsigned long ret_num = 0;
    CompressInputImage* dcpdatas = decompress_images(buf, len ,hit_img_ids, &ret_num);

    for (unsigned long i = 0; i < ret_num; i++)
    {
        printf("%s  %s  %u  %p  %u \r\n",dcpdatas[i].image_id, dcpdatas[i].channel_id,
                                           dcpdatas[i].ts_ms, dcpdatas[i].buf, dcpdatas[i].buf_len);

        print_hex_str(dcpdatas[i].buf, 64);

        callback(1, &dcpdatas[i]);

        free(dcpdatas[i].image_id);
        free(dcpdatas[i].channel_id);
        free(dcpdatas[i].buf);
    }
}
