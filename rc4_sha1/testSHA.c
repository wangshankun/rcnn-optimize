#include"sha1.h"
#define SHA_DIGEST_LENGTH 20
int main(int argc, char const *argv[]){
    char digest0[2*SHA_DIGEST_LENGTH] = {};
    char digest1[2*SHA_DIGEST_LENGTH] = {};
    char digest2[2*SHA_DIGEST_LENGTH] = {};
    memset(digest0, SHA_DIGEST_LENGTH, SHA_DIGEST_LENGTH);
    sha1((unsigned char*)&digest0, SHA_DIGEST_LENGTH, (unsigned char*)&digest1);
    sha1((unsigned char*)&digest1, SHA_DIGEST_LENGTH, (unsigned char*)&digest2);
    digest2[16] = '\0';
    printf("%s \r\n", digest2);
    return 0;
}

