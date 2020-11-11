
#include <memory>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <iostream>
#include <algorithm>

#include "omp.h"


#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)

#define readfile(name, buffer, elem_size) do\
{\
  FILE *out = fopen(name, "rb");\
  if(out != NULL)\
  {\
        fread (buffer , sizeof(buffer[0]), elem_size, out);\
        fclose (out);\
  }\
} while(0)


void softmax(float *input, float* out, size_t input_len) {

  float m = -INFINITY;
  for (size_t i = 0; i < input_len; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }

  float sum = 0.0;
  for (size_t i = 0; i < input_len; i++) {
    sum += expf(input[i] - m);
  }

  float offset = m + logf(sum);
  for (size_t i = 0; i < input_len; i++) {
    out[i] = expf(input[i] - offset);
  }
}

int main()
{
  omp_set_num_threads(1);

  float* data = (float*)malloc(1024*1024*3*sizeof(float));
  readfile("softmax.bin", data, 1024*1024*3);

  float* out = (float*)malloc(1024*1024*3*sizeof(float));
   
    double elapsed;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

        #pragma omp parallel for
	for (int i = 0; i < 1024*1024; i++)
        {
            softmax(data + i*3, out + i*3, 3);
        }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("elapsed time:%f out[x]:%f\r\n",elapsed, out[888888]);


    return 0;
}
