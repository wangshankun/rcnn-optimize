
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


static bool PairCompare(const std::pair<float, float>& lhs,
                        const std::pair<float, float>& rhs) {
  return lhs.second > rhs.second;
}

void Argmax(const float* data, int v_len, int N, float* out) {
  std::vector<float> v;
  v.assign(data, data + v_len);

  std::vector<std::pair<float, float> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
  {
    pairs.push_back(std::make_pair(float(i), v[i]));
  }

  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  for (size_t k = 0; k < N; k++)
  {
    out[k] = pairs[k].first; 
  }

  for (size_t k = 0; k < N; k++)
  {
    out[N + k] = pairs[k].second; 
  }
}

int main()
{
  omp_set_num_threads(4);

  float* data = (float*)malloc(75*8000*sizeof(float));
  readfile("prob.bin", data, 75*8000);

  float* out = (float*)malloc(75*3*2*sizeof(float));
   
    double elapsed;
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

        #pragma omp parallel for
	for (int i = 0; i < 75; i++)
        {
           Argmax(data + i * 8000, 8000, 3, out + i * 2 * 3);
        }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("elapsed time:%f\r\n",elapsed);

    for(int i = 0; i < 75; i++) {
      for(int j = 0; j < 6; j++)
        {
           printf("%-9.4f ", out[i*6 + j]);
        }
         printf("\r\n");
       }
    return 0;
}
