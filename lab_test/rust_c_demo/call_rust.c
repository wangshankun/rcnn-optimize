//https://dev.to/dandyvica/how-to-call-rust-functions-from-c-on-linux-h37

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

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

extern CompressOutputData*  compress_images(struct CompressInputImage*, unsigned int, int ,int ,unsigned long*);

int main() 
{

    CompressInputImage tts[3];
    tts[0].image_id = "aadf1";
    tts[0].channel_id = "111111233_sdsd";
    tts[0].ts_ms      = 66666;
    tts[0].buf          =  malloc(20);
    memset(tts[0].buf,'a',20);
    tts[0].buf_len      = 20;

    tts[1].image_id = "aadf2";
    tts[1].channel_id = "111111233_sdsd";
    tts[1].ts_ms      = 666666;
    tts[1].buf          =  malloc(21);
    memset(tts[1].buf,'b',21);
    tts[1].buf_len      = 21;
    
    tts[2].image_id = "aadf3";
    tts[2].channel_id = "33331233_sdsd";
    tts[2].ts_ms      = 6666666;
    tts[2].buf          =  malloc(22);
    memset(tts[2].buf,'c',22);
    tts[2].buf_len      = 22;

    unsigned long ret_num = 0;
    CompressOutputData* cpdatas = compress_images(tts, 3, 16, 0, &ret_num);
    printf("ret num: %d \r\n",ret_num);

    for(unsigned long i = 0; i < ret_num; i++)
    {
        printf("%s %s %s %s %d %d %d\r\n",cpdatas[i].channel_ids, cpdatas[i].image_ids,
                                     cpdatas[i].ts_arrays,  cpdatas[i].offsets,
                                      cpdatas[i].compress_rate,cpdatas[i].image_format,
                                     cpdatas[i].compressed_buf_len);
        print_hex_str(cpdatas[i].compressed_buf, cpdatas[i].compressed_buf_len);

        free(cpdatas[i].compressed_buf);
        free(cpdatas[i].channel_ids);
        free(cpdatas[i].image_ids);
        free(cpdatas[i].ts_arrays);
        free(cpdatas[i].offsets);
    }

    return 0;
}
