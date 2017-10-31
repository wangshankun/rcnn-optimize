#include "parallel.h"
typedef struct {
    float* bottom_data;
    float* bottom_rois;
    float* top_data;
    int    num_rois;
    int    pooled_height_;
    int    pooled_width_;
    int    width_;
    int    height_;
    int    channels_;
    float  spatial_scale_;
} roi_pool_arg_t;

#define max(a, b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
     
#define min(a, b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _b : _a; })

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)

