#ifndef __FFJPEG_COLOR_H__
#define __FFJPEG_COLOR_H__

// ����ͷ�ļ�
#include "stdefine.h"

#ifdef __cplusplus
extern "C" {
#endif

void yuv_to_rgb(int y, int u, int v, BYTE *r, BYTE *g, BYTE *b);
void rgb_to_yuv(BYTE r, BYTE g, BYTE b, int *y, int *u, int *v);

#ifdef __cplusplus
}
#endif

#endif











