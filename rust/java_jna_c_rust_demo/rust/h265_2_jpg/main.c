#include<stdio.h>
#include<stdlib.h>

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

extern void decompress_2_jpeg(unsigned char *video_buf, int buf_size, int* hit_array, int hit_len, unsigned char** out_buf_arry, int** out_size_arry);

int main() 
{

    unsigned char* buf = ( unsigned char*)malloc(1952953);
    readfile("test.hevc", buf, 1952953);

    int   hit_array[3]     = {10,20,40};//获取第10/20/40帧

    unsigned char* out_buf_arry  = NULL;
    int*           out_size_arry = NULL;

    decompress_2_jpeg(buf, 1952953, hit_array, 3, &out_buf_arry, &out_size_arry);

    int current_ptr = 0, i = 0;
    for(i = 0; i < 3; i++)
    {
        char name[32] = {};
        sprintf(name, "h265_yuv_%d.jpg", i);
        savefile(name ,(const char*)(out_buf_arry + current_ptr), out_size_arry[i]);
        current_ptr = current_ptr + out_size_arry[i];
    }
    
    //free(buf);这个在ffmpeg avio 流接口使用时候会free掉，不需要再free了
    free(out_buf_arry);
    free(out_size_arry);
    return 0;
}
