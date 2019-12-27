#include<stdio.h>
#include<stdlib.h>
#include<string.h>

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


typedef struct CompressInputImage  {
    char*  image_id;                /// 图像的id
    char*  channel_id;            /// 图像的channel id
    long   ts_ms;                 /// int64 时间戳
    int    compress_rate;           ///  压缩率： 0 不压缩， 100 最大压缩比。
    int    image_format;           ///  图像的输入格式。 1: jpg, 2: png
    void*  buf;                    /// 这个压缩的byte array的buffer, 这个地址通过Memory分配的, 由调用者管理
    long   buf_len;               /// 上面这个buffer的字节数长度
} CompressInputImage ;

typedef struct CompressOutputData  {
    int   version;               ///压缩库的版本号
    char* channel_id;           ///摄像头的id号,代表了空间信息
    char* image_ids;           ///输出的图片id号列表
    long* ts_array;             ///图片的时间戳列表
    int   ts_array_len;         ///图片的时间戳数组的长度
    void* compressed_buf;       ///压缩图片的content buffer的地址
    long  compressed_buf_len;   /// content buffer的长度
} CompressOutputData  ;

typedef struct DecompressInputData  {
    int    version;              ///压缩数据的版本号 
    char*  channel_id;           ///摄像头ID 空间信息 
    long*  ts_array;             ///时间片数组
    long   ts_array_len;         ///数组长度
    char*  image_ids;            ///摄像头id的arra ;分号为分隔符
    char*  compressed_buf;       ///图像压缩的buffer， 压缩结果的 compressed_buf 对应
    long   compressed_buf_len;   ///buffer长度
    char*  target_indexs;        ///需要解压的图片在ts， 以及image id里面的offset;分号为分隔符
    int    image_compress_rate;  ///解压图片使用的压缩格式使用的压缩率，
    int    image_compress_method;///要求返回的图片的压缩格式， 1 为jpeg， 2 为png （jpeg为默认）
} DecompressInputData  ;

typedef struct DecompressoOutput   {
    char* channel_id;  ///解压出的图片channel id
    char* image_id;   ///解压出图片的image id
    long  ts_ms;       ///解压图片的时间戳
    void* image_buf;   ///图片经过image_format里面使用的格式压缩的二进制buffer
    long  image_buf_len;    ///压缩图片buffer的长度
    int   image_format;  ///格式： 1 为jpeg， 2 为png, 默认为jpeg
} DecompressoOutput   ;

typedef void(*compress_callback)(int , CompressOutputData* );

typedef void(*decompress_callback)(int , DecompressoOutput* );

int  init(int enable_gpu, int gpu_id, int enable_cpu_bind, int* cpu_ids, int cpu_ids_len)
{
    printf("%s  %d\r\n",__FUNCTION__,__LINE__);
}

void compressListOfImage( CompressInputImage* input_images, int len, compress_callback callback)
{
    char                guard_id[64]       = {0};
    int                 ids_byte_len       = 0;
    int                 ts_array_len       = 0;
    int                 compressed_buf_len = 0;
    
    CompressOutputData  gurad_cd   =  {0, NULL, NULL, NULL, 0, NULL, 0};
    
    for(int i = 0; i < len; i++)
    {
        if(strcmp(guard_id, input_images[i].channel_id) != 0)//新channel进来
        {
            if(gurad_cd.channel_id != NULL)//gurad_cd不为空，返回callback结果，释放空间，重新开始
            {
                
                callback(1, &gurad_cd);
                free(gurad_cd.channel_id);
                free(gurad_cd.image_ids);
                free(gurad_cd.ts_array);
                free(gurad_cd.compressed_buf);
            }
            //从新channel开始一个压缩包
            strcpy(guard_id, input_images[i].channel_id);
            //版本号
            gurad_cd.version     = 0;
            //channel id
            gurad_cd.channel_id  = malloc(64);
            stpcpy(gurad_cd.channel_id, input_images[i].channel_id);
            //image_id
            gurad_cd.image_ids   = malloc(1);
            gurad_cd.image_ids[0]= '\0';
            ids_byte_len         = 1;//'\0'
            //ts_array
            gurad_cd.ts_array    = malloc(sizeof(long));
            ts_array_len = 0;
            //compressed_buf
            gurad_cd.compressed_buf  = malloc(1);
            compressed_buf_len       = 0;
        }
        //image_id添加 分号为分隔符
        ids_byte_len       = ids_byte_len + strlen(input_images[i].image_id) + 1;//+1是分隔符
        gurad_cd.image_ids = realloc(gurad_cd.image_ids, ids_byte_len);
        gurad_cd.image_ids = strcat(gurad_cd.image_ids, input_images[i].image_id);
        //gurad_cd.image_ids = strcat(gurad_cd.image_ids, ";");
        gurad_cd.image_ids[ids_byte_len - 1] = '\0';
        gurad_cd.image_ids[ids_byte_len - 2] = ';';
        
        //ts添加
        ts_array_len                        = ts_array_len + 1;
        gurad_cd.ts_array                   = realloc(gurad_cd.ts_array, ts_array_len * sizeof(long));
        gurad_cd.ts_array[ts_array_len - 1] = input_images[i].ts_ms;
        gurad_cd.ts_array_len               = ts_array_len;
        
        //compressed_buf添加///这个例子是不压缩的，直接添加
        gurad_cd.compressed_buf = realloc(gurad_cd.compressed_buf, compressed_buf_len + input_images[i].buf_len);
        memcpy(gurad_cd.compressed_buf + compressed_buf_len, input_images[i].buf, input_images[i].buf_len);
        compressed_buf_len          = compressed_buf_len + input_images[i].buf_len;
        gurad_cd.compressed_buf_len = compressed_buf_len;

        if( i == len - 1)//如果是最后一张图，压缩后也callback
        {
            callback(1, &gurad_cd);
            free(gurad_cd.channel_id);
            free(gurad_cd.image_ids);
            free(gurad_cd.ts_array);
            free(gurad_cd.compressed_buf);
        }
    }

}

void decompressListOfImage( DecompressInputData* inputs, int len, decompress_callback callback)
{
    DecompressoOutput gurad_dc = {"hanzhou_1137_12121","107",1235260,NULL,1111,0};
    callback(1, &gurad_dc);
}
