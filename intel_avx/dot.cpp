#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>
#include <omp.h>

extern "C" inline float horizontal_add(__m256 a) {
    __m256 t1 = _mm256_hadd_ps(a,a);
    __m256 t2 = _mm256_hadd_ps(t1,t1);
    __m128 t3 = _mm256_extractf128_ps(t2,1);
    __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
    return _mm_cvtss_f32(t4);
}

extern "C" float dot_avx(float * __restrict x, float * __restrict y, const int n) {
    x = (float*)__builtin_assume_aligned (x, 32);
    y = (float*)__builtin_assume_aligned (y, 32);
    float sum = 0;
    #pragma omp parallel reduction(+:sum)
    {
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        __m256 sum4 = _mm256_setzero_ps();
        __m256 x8, y8;
        #pragma omp for
        for(int i=0; i<n; i+=32) {
            x8 = _mm256_loadu_ps(&x[i]);
            y8 = _mm256_loadu_ps(&y[i]);
            sum1 = _mm256_add_ps(_mm256_mul_ps(x8,y8),sum1);
            x8 = _mm256_loadu_ps(&x[i+8]);
            y8 = _mm256_loadu_ps(&y[i+8]);
            sum2 = _mm256_add_ps(_mm256_mul_ps(x8,y8),sum2);
            x8 = _mm256_loadu_ps(&x[i+16]);
            y8 = _mm256_loadu_ps(&y[i+16]);
            sum3 = _mm256_add_ps(_mm256_mul_ps(x8,y8),sum3);
            x8 = _mm256_loadu_ps(&x[i+24]);
            y8 = _mm256_loadu_ps(&y[i+24]);
            sum4 = _mm256_add_ps(_mm256_mul_ps(x8,y8),sum4);
        }
        sum += horizontal_add(_mm256_add_ps(_mm256_add_ps(sum1,sum2),_mm256_add_ps(sum3,sum4)));
    }
    return sum; 
}

extern "C" float dot(float * __restrict x, float * __restrict y, const int n) {
    x = (float*)__builtin_assume_aligned (x, 32);
    y = (float*)__builtin_assume_aligned (y, 32);
    float sum = 0;
    for(int i=0; i<n; i++) {
        sum += x[i]*y[i];
    }
    return sum;
}

int main(){
    uint64_t LEN = 1 << 27;
    float *x = (float*)_mm_malloc(sizeof(float)*LEN,64);
    float *y = (float*)_mm_malloc(sizeof(float)*LEN,64);
    for(uint64_t i=0; i<LEN; i++) { x[i] = 1.0*rand()/RAND_MAX - 0.5; y[i] = 1.0*rand()/RAND_MAX - 0.5;}

    uint64_t size = 2*sizeof(float)*LEN;

    volatile float sum = 0;
    double dtime, rate, flops;  
    int repeat = 100;

    dtime = omp_get_wtime();
    for(int i=0; i<repeat; i++) sum += dot(x,y,LEN);
    dtime = omp_get_wtime() - dtime;
    rate = 1.0*repeat*size/dtime*1E-9;
    flops = 2.0*repeat*LEN/dtime*1E-9;
    printf("%f GB, sum %f, time %f s, %.2f GB/s, %.2f GFLOPS\n", 1.0*size/1024/1024/1024, sum, dtime, rate,flops);

    sum = 0;
    dtime = omp_get_wtime();
    for(int i=0; i<repeat; i++) sum += dot_avx(x,y,LEN);
    dtime = omp_get_wtime() - dtime;
    rate = 1.0*repeat*size/dtime*1E-9;
    flops = 2.0*repeat*LEN/dtime*1E-9;

    printf("%f GB, sum %f, time %f s, %.2f GB/s, %.2f GFLOPS\n", 1.0*size/1024/1024/1024, sum, dtime, rate,flops);
}