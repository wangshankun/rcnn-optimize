#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
}

void print_hex_str(const void* buf , size_t size)
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

typedef struct CompressInputImage  {
    char*           image_id;
    char*           channel_id;
    long            ts_ms;
    char*           buf;
    unsigned long   buf_len;
} CompressInputImage;

typedef struct CompressOutputData  {
    char*          channel_ids;
    char*          image_ids;
    char*          ts_arrays;
    char*          offsets;
    int            version;
    int            compress_rate;
    int            image_format;
    char*          compressed_buf;
    unsigned long  compressed_buf_len;
} CompressOutputData;

typedef void(*compress_callback)(int , CompressOutputData* );

typedef void(*decompress_callback)(int , CompressInputImage* );

extern CompressOutputData*  compress_images(struct CompressInputImage*, unsigned int, int ,int ,unsigned long*);

extern CompressInputImage*  decompress_images(char*, int , const char*, unsigned long*);
