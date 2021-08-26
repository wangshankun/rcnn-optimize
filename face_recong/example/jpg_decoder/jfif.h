#ifndef __FFJPEG_JFIF_H__
#define __FFJPEG_JFIF_H__

#include<stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int jpg_decode_rgb(const char *file, uint8_t **data, int *w, int *h);

#ifdef __cplusplus
}
#endif

#endif

