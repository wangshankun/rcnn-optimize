#ifndef IDST_MODEL_CRYPT
#define IDST_MODEL_CRYPT

#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

//文件头header_magic_string用来判断模型是否加密
static char header_magic_string[] = "idst_encrypt_model";

void get_code_key(char* key);

int is_idst_encrypt_model(const char* in_file, bool memory_model = 0);

int idst_rc4_de_file_to_mem(const char* in_file,  unsigned char** data_out, int* size_out, const char* pass);

int idst_rc4_en_file(const char* in_file, const char* out_file, const char* pass);
int idst_rc4_de_file(const char* in_file, const char* out_file, const char* pass);

void idst_rc4_de_memfile_to_mem(unsigned char* mem_file, int size, unsigned char** data_out, int* size_out, const char* pass);

#endif /* IDST_MODEL_CRYPT */
