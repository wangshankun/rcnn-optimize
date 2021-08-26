#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "jfif.h"

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)

int main(int argc, char *argv[])
{
    void *jfif = NULL;

    if (argc < 3) {
        printf(
            "jfif test program\n"
            "usage: ffjpeg -db filename  (decode jpg file to decode.bin) \r\n"
        );
        return 0;
    }

   if (strcmp(argv[1], "-db") == 0) {
        uint8_t* data;
        int w = 0;
        int h = 0;
        if( 0 == jpg_decode_rgb(argv[2], &data, &w, &h))
        {
            savefile("decode.bin", data, w*h*3);
            free(data);
        }
        else
        {
            printf("jpg_decode_rgb error!\r\n");
            return -1;
        }

    }

    return 0;
}

