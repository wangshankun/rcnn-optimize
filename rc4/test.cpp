#include <openssl/rc4.h>
#include <openssl/sha.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

//RC4文件加解密
static void rc4_file(const char* in_file, const char* out_file, const char* pass)
{
    RC4_KEY key;
    int length = strlen(pass);
    RC4_set_key(&key, length, (unsigned const char*)pass);//设置密钥

    int fd_in = open(in_file, O_RDONLY);
    struct stat st;
    fstat(fd_in, &st);
    int size = st.st_size;

    unsigned char* data_in;
    data_in = (unsigned char*) calloc(size, sizeof(char));
    pread(fd_in, data_in, size, 0);
    unsigned char *data_out = (unsigned char*)malloc(size);

    RC4(&key, size, data_in, data_out);
    close(fd_in);

    int fd_out = open(out_file, O_RDWR|O_CREAT|O_APPEND, 0777);
    /* 清空文件 */
    ftruncate(fd_out, 0);
    /* 重新设置文件偏移量 */
    lseek(fd_out, 0, SEEK_SET);
    
    pwrite(fd_out, data_out, size, 0);
    close(fd_out);

    free(data_in);
    free(data_out);
}

//RC4内存加解密
static void rc4_mem(unsigned char* data_in, unsigned char* data_out, int size, const char* pass)
{
    RC4_KEY key;
    int length = strlen(pass);
    RC4_set_key(&key, length, (unsigned const char*)pass);//设置密钥

    RC4(&key, size, data_in, data_out);
}

char header_magic_string[] = "idst_encrypt_model";
//idst文件加密
  //加文件头header_magic_string判断模型是否加密
static void idst_rc4_en_file(const char* in_file, const char* out_file, const char* pass)
{
    RC4_KEY key;
    int length = strlen(pass);
    RC4_set_key(&key, length, (unsigned const char*)pass);//设置密钥

    int fd_in = open(in_file, O_RDONLY);
    struct stat st;
    fstat(fd_in, &st);
    int size = st.st_size;

    unsigned char* data_in;
    data_in = (unsigned char*) calloc(size, sizeof(char));
    pread(fd_in, data_in, size, 0);

    int header_length = strlen(header_magic_string) + 1;
    unsigned char *data_out = (unsigned char*)calloc(size + header_length, sizeof(char));
    memcpy(data_out, header_magic_string, header_length);
    
    RC4(&key, size, data_in, data_out + header_length);

    close(fd_in);

    int fd_out = open(out_file, O_RDWR|O_CREAT|O_APPEND, 0777);
    /* 清空文件 */
    ftruncate(fd_out, 0);
    /* 重新设置文件偏移量 */
    lseek(fd_out, 0, SEEK_SET);
    
    pwrite(fd_out, data_out, size + header_length, 0);
    close(fd_out);

    free(data_in);
    free(data_out);
}

//idst模型解密
static int idst_rc4_de_file(const char* in_file, const char* out_file, const char* pass)
{
    RC4_KEY key;
    int length = strlen(pass);
    RC4_set_key(&key, length, (unsigned const char*)pass);//设置密钥

    int header_length = strlen(header_magic_string) + 1;
    int fd_in = open(in_file, O_RDONLY);
    struct stat st;
    fstat(fd_in, &st);
    int size = st.st_size;

    char parse_header[header_length] = {};
    pread(fd_in, parse_header, header_length, 0);

    if(strcmp(parse_header, header_magic_string) == 0)
    {
        unsigned char* data_in;
        data_in = (unsigned char*) calloc(size, sizeof(char));
        pread(fd_in, data_in, size - header_length, header_length);
        
        unsigned char *data_out = (unsigned char*)calloc(size, sizeof(char));
        
        RC4(&key, size, data_in, data_out);

        int fd_out = open(out_file, O_RDWR|O_CREAT|O_APPEND, 0777);
        /* 清空文件 */
        ftruncate(fd_out, 0);
        /* 重新设置文件偏移量 */
        lseek(fd_out, 0, SEEK_SET);
        pwrite(fd_out, data_out, size - header_length, 0);
        close(fd_in);
        close(fd_out);

        free(data_in);
        free(data_out);
    }

    return 0;
}


int main(int argc, char* argv[])
{
    char digest0[SHA_DIGEST_LENGTH] = {};
    char digest1[SHA_DIGEST_LENGTH] = {};
    char digest2[SHA_DIGEST_LENGTH] = {};
    memset(digest0, SHA_DIGEST_LENGTH - 17, SHA_DIGEST_LENGTH);//SHA_DIGEST_LENGTH为初始串内容
    SHA1((unsigned char*)&digest0, SHA_DIGEST_LENGTH, (unsigned char*)&digest1);//做两次sha1
    SHA1((unsigned char*)&digest1, SHA_DIGEST_LENGTH, (unsigned char*)&digest2);
    digest2[17] = '\0';//取前17个字符做rc4的key

    rc4_file("1.jpg", "1.jpg.enc", digest2);
    rc4_file("1.jpg.enc", "1.jpg.denc", digest2);
    
    idst_rc4_en_file("1.jpg", "1.jpg.idst_enc", digest2);
    idst_rc4_de_file("1.jpg.idst_enc", "1.jpg.idst_denc", digest2);
    
}
