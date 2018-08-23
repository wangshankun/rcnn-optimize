#include "parallel.h"
typedef struct {
    float* bottom_data;
    float* bottom_rois;
    float* top_data;
    int    num;
    int    pooled_height;
    int    pooled_width;
    int    width;
    int    height;
    int    channels;
    float  spatial_scale;
    int    output_dim;
    int    group_size;
} psroi_pool_arg_t;

typedef struct {
    float* bottom_data  ;
    float* top_data     ;
    int    num          ;
    int    channels     ;
    int    height       ;
    int    width        ;
    int    pooled_height;
    int    pooled_width ;
    int    kernel_h     ;
    int    kernel_w     ;
    int    pad_h        ;
    int    pad_w        ;
    int    stride_h     ;
    int    stride_w     ;
} ave_pool_arg_t;

typedef struct {
    float* bottom_data;
    float* top_data;
    int    outer_num;
    int    channels;
    int    inner_num;
} softmax_arg_t;

void psroi_pooling_multithreading(psroi_pool_arg_t* arg);
void ave_pool(ave_pool_arg_t* arg);
void softmax(softmax_arg_t* arg);

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)


void print_hex(const void* buf , size_t size)
{
    unsigned char* str = (unsigned char*)buf;
    char line[512] = {0};
    const size_t lineLength = 16; // 8或者32
    char text[24] = {0};
    char* pc;
    int textLength = lineLength;
    size_t ix = 0 ;
    size_t jx = 0 ;

    for (ix = 0 ; ix < size ; ix += lineLength) {
        sprintf(line, "%.8xh: ", ix);
// 打印16进制
        for (jx = 0 ; jx != lineLength ; jx++) {
            if (ix + jx >= size) {
                sprintf(line + (11 + jx * 3), "   "); // 处理最后一行空白
                if (ix + jx == size)
                    textLength = jx;  // 处理最后一行文本截断
            } else
                sprintf(line + (11 + jx * 3), "%.2X ", * (str + ix + jx));
        }
// 打印字符串
        {
            memcpy(text, str + ix, lineLength);
            pc = text;
            while (pc != text + lineLength) {
                if ((unsigned char)*pc < 0x20) // 空格之前为控制码
                    *pc = '.';                 // 控制码转成'.'显示
                pc++;
            }
            text[textLength] = '\0';
            sprintf(line + (11 + lineLength * 3), "; %s", text);
        }

        printf("%s\n", line);
    }
}