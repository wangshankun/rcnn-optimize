#include "idst_model_crypt.h"
#include "rc4.h"
#include "sha.h"
#include <stdarg.h>

#define TAG "model crypt"

//SHA_DIGEST_LENGTH宏的数值初始化内存，做两次sha1后取最前面16个字符做key
void get_code_key(char* key)
{
    char digest0[2*SHA_DIGEST_LENGTH] = {};
    char digest1[2*SHA_DIGEST_LENGTH] = {};
    char digest2[2*SHA_DIGEST_LENGTH] = {};
    memset(digest0, SHA_DIGEST_LENGTH, SHA_DIGEST_LENGTH);
    sha1((char*)&digest0, SHA_DIGEST_LENGTH, (char*)&digest1);
    sha1((char*)&digest1, SHA_DIGEST_LENGTH, (char*)&digest2);
    digest2[16] = '\0';
    strcpy(key, digest2);
}

int is_idst_encrypt_model(const char* in_file, bool memory_model)
{
    if (memory_model)
    {
        if(strncmp(in_file, header_magic_string, strlen(header_magic_string)) == 0)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
    else
    {
        int header_length = strlen(header_magic_string) + 1;
        int fd_in = open(in_file, O_RDONLY);
        if (fd_in == -1)
        {
            printf("Open file failed！\r\n");
            return -1;
        }
        struct stat st;
        fstat(fd_in, &st);
        int size = st.st_size;

        if (size < header_length)
        {
            return 0;
        }
        
        char parse_header[header_length];
        memset(parse_header, 0, header_length);
        pread(fd_in, parse_header, header_length, 0);
        close(fd_in);

        if(strcmp(parse_header, header_magic_string) == 0)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

}


//RC4内存加解密
static void idst_rc4_de_mem_to_mem(unsigned char* data_in, unsigned char* data_out, int size, const char* pass)
{
    ARC4 rc4;
    rc4.setKey((unsigned char*)(const_cast<char*>(pass)),  strlen(pass));
    rc4.encrypt(data_in, data_out, size);
}

//idst模型加密到文件
  //加文件头header_magic_string
int idst_rc4_en_file(const char* in_file, const char* out_file, const char* pass)
{
    int fd_in = open(in_file, O_RDONLY);
    if (fd_in == -1)
    {
        printf("Open file failed！\r\n");
        return -1;
    }

    struct stat st;
    fstat(fd_in, &st);
    int size = st.st_size;

    unsigned char* data_in;
    data_in = (unsigned char*) calloc(size, sizeof(char));
    pread(fd_in, data_in, size, 0);

    int header_length = strlen(header_magic_string) + 1;
    unsigned char *data_out = (unsigned char*)calloc(size + header_length, sizeof(char));
    memcpy(data_out, header_magic_string, header_length);

    idst_rc4_de_mem_to_mem(data_in, data_out + header_length, size, pass);

    close(fd_in);

    int fd_out = open(out_file, O_RDWR|O_CREAT|O_APPEND, 0777);
    if (fd_out == -1)
    {
        printf("Open out_file failed！\r\n");
        return -1;
    }
    /* 清空文件、重新设置文件偏移量 */
    ftruncate(fd_out, 0);
    lseek(fd_out, 0, SEEK_SET);
    
    pwrite(fd_out, data_out, size + header_length, 0);
    close(fd_out);

    free(data_in);
    free(data_out);
}

//idst模型解密到内存
int idst_rc4_de_file_to_mem(const char* in_file,  unsigned char** data_out, int* size_out, const char* pass)
{
    int header_length = strlen(header_magic_string) + 1;
    int fd_in = open(in_file, O_RDONLY);
    if (fd_in == -1)
    {
        printf("Open file failed！\r\n");
        return -1;
    }
    struct stat st;
    fstat(fd_in, &st);
    int size = st.st_size;

    unsigned char* data_in;
    int data_size = size - header_length;
    data_in = (unsigned char*) calloc(size, sizeof(char));
    pread(fd_in, data_in, data_size, header_length);//offfset:header_length size:data_size
    
    *data_out = (unsigned char*)calloc(data_size, sizeof(char));
    idst_rc4_de_mem_to_mem(data_in, *data_out, data_size, pass);
    *size_out = data_size;

    close(fd_in);
    free(data_in);
    return 0;
}

//idst模型解密到文件
int idst_rc4_de_file(const char* in_file, const char* out_file, const char* pass)
{
     if(is_idst_encrypt_model(in_file) == 1)
     {
        unsigned char* data_out = NULL;
        int size_out = 0;
        idst_rc4_de_file_to_mem(in_file, &data_out, &size_out, pass);
        
        if(size_out > 0  && data_out != NULL)
        {
            int fd_out = open(out_file, O_RDWR|O_CREAT|O_APPEND, 0777);
            if (fd_out == -1)
            {
                printf("Open out_file failed！\r\n");
                return -1;
            }
            /* 清空文件重新设置文件偏移量 */
            ftruncate(fd_out, 0);
            lseek(fd_out, 0, SEEK_SET);
            pwrite(fd_out, data_out, size_out, 0);
            close(fd_out);
            free(data_out);
            return 0;
        }
        else
        {
            printf("de model file failed！\r\n");
            return -1;
        }
     }
     else
     {
         printf("model is not encrypted failed！\r\n");
         return 1;
     }
}

//RC4内存文件解密到内存中
void idst_rc4_de_memfile_to_mem(unsigned char* mem_file, int size, unsigned char** data_out, int* size_out, const char* pass)
{
    int header_length = strlen(header_magic_string) + 1;
    unsigned char* data_in = mem_file + header_length;
    int data_size = size - header_length;
    *data_out = (unsigned char*)calloc(data_size, sizeof(char));
    idst_rc4_de_mem_to_mem(data_in, *data_out, data_size, pass);
    *size_out = data_size;
}