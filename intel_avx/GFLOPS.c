#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>
#include <smmintrin.h>  /* SSE 4.1 */
#include <immintrin.h>  /* SSE 4.1 */
#include <math.h>
        
 int  mask8i[9][8]={
  { 0, 0, 0, 0, 0, 0, 0, 0},
  {-1, 0, 0, 0, 0, 0, 0, 0},
  {-1,-1, 0, 0, 0, 0, 0, 0},
  {-1,-1,-1, 0, 0, 0, 0, 0},
  {-1,-1,-1,-1, 0, 0, 0, 0},
  {-1,-1,-1,-1,-1, 0, 0, 0},
  {-1,-1,-1,-1,-1,-1, 0, 0},
  {-1,-1,-1,-1,-1,-1,-1, 0},
  {-1,-1,-1,-1,-1,-1,-1,-1}
  };
  
 float dot_avx_27(float * __restrict x, float * __restrict y) 
 {
    x = (float*)__builtin_assume_aligned (x, 32);
    y = (float*)__builtin_assume_aligned (y, 32);
    register __m256i load_3_mask = _mm256_castps_si256(_mm256_load_ps((const float*)mask8i[3]));
    __m256  b1 = _mm256_loadu_ps(&x[0]);
    __m256  b2 = _mm256_loadu_ps(&x[8]);
    __m256  b3 = _mm256_loadu_ps(&x[16]);
    __m256  b4 = _mm256_loadu_ps(&x[24]);

    __m256 a1 = _mm256_loadu_ps(&y[0]);
    __m256 a2 = _mm256_loadu_ps(&y[8]);
    __m256 a3 = _mm256_loadu_ps(&y[16]);
    __m256 a4 = _mm256_maskload_ps(&y[24], load_3_mask);

    __m256 m1 = _mm256_mul_ps(a1, b1);
    __m256 m2 = _mm256_mul_ps(a2, b2);
    __m256 m3 = _mm256_mul_ps(a3, b3);
    __m256 m4 = _mm256_mul_ps(a4, b4);
    m1 = _mm256_add_ps(m1, m2);
    m3 = _mm256_add_ps(m3, m4);
    m1 = _mm256_add_ps(m1, m3);
    __m256 t1 = _mm256_hadd_ps(m1, m1);
    __m256 t2 = _mm256_hadd_ps(t1, t1);
    __m128 t3 = _mm256_extractf128_ps(t2, 1);
    __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2), t3);
    return  _mm_cvtss_f32(t4);
}

void main()
{
    float *x = (float*)malloc(32000000*4);
    float *y = (float*)malloc(32000000*4);
    volatile float sum = 0;
    int i, repeat = 100;
    for(i=0; i<32; i++) { x[i] = 1.0*rand()/RAND_MAX - 0.5; y[i] = 1.0*rand()/RAND_MAX - 0.5;}
    for( i=0; i<repeat; i++) sum += dot_avx_27(x,y);
    printf("%s %d sum:%f\r\n",__FUNCTION__,__LINE__,sum);
    free(x);
    free(y);
}
