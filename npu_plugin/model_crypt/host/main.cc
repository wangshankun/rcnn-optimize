#include "idst_model_crypt.h"
int main(int argc, char* argv[])
{
    const char *choice = argv[1];
    const char *sourceFile = argv[2];
    const char *resultFile = argv[3];
    if(argc < 4)
    {
        printf("Missing arg, it has to be like this: \n");
        printf("\t crypt_tool en plain_file  cipher_file\n");
        printf("\t crypt_tool de cipher_file plain_file\n");
        return 0;
    }

    char key[64] = {};//至少16个有效字符
    get_code_key(key);
    
    if(strcmp(choice,"en") == 0)
    {
        idst_rc4_en_file(sourceFile, resultFile, key);
    }
    else if(strcmp(choice,"de") == 0)
    {
        idst_rc4_de_file(sourceFile, resultFile, key);
    }
    else
    {
        printf("Missing arg, it has to be like this: \n");
        printf("\t crypt_tool en plain_file  cipher_file\n");
        printf("\t crypt_tool de cipher_file plain_file\n");
        return 0;
    }

    return 1;
}
