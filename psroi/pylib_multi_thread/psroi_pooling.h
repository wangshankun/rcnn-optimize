#include "parallel.h"
typedef struct {
    float* bottom_data;
    float* bottom_rois;
    float* top_data;
    int    num_rois;
    int    pooled_height;
    int    pooled_width;
    int    width;
    int    height;
    int    channels;
    float  spatial_scale;
    int    output_dim;
    int    group_size;
} psroi_pool_arg_t;

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)

