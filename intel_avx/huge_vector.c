//gcc -w -msse4 -mavx -fopenmp -O3 -pg -g Huge_Vector.c
/********************************************************
64*27 与 27*480000的矩阵乘法优化
A B两个文件按照一维数组形式存储两个矩阵
*********************************************************/
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>
#include <smmintrin.h>  /* SSE 4.1 */
#include <immintrin.h>  /* SSE 4.1 */
#include <math.h>
#include <omp.h>

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

inline void vector_27xN27_looplittle(float * __restrict x, float * __restrict y, float * __restrict OUT, int N)
{
    __m256i mask = _mm256_castps_si256(_mm256_load_ps((const float*)mask8i[3]));
    int i=0;
    int index=0;
    x = (float*)__builtin_assume_aligned (x, 32);
    y = (float*)__builtin_assume_aligned (y, 32);
    register __m256i load_3_mask = _mm256_castps_si256(_mm256_load_ps((const float*)mask8i[3]));
   // omp_set_num_threads(8);
    //#pragma omp parallel shared(x,y,OUT) private(i,index)
    {
    //#pragma omp for schedule(static, 60000)
        for(i=0;i<480000;i++)
        {
            __m256  b1 = _mm256_loadu_ps(&x[0 + 27*i]);
            __m256  b2 = _mm256_loadu_ps(&x[8 + 27*i]);
            __m256  b3 = _mm256_loadu_ps(&x[16 + 27*i]);
            __m256  b4 = _mm256_maskload_ps(&x[24 + 27*i], load_3_mask);
           for(index=0;index<64;index++)
           {
                __m256 a1 = _mm256_loadu_ps(&y[0 + 27*index]);
                __m256 a2 = _mm256_loadu_ps(&y[8 + 27*index]);
                __m256 a3 = _mm256_loadu_ps(&y[16 + 27*index]);
                __m256 a4 = _mm256_loadu_ps(&y[24 + 27*index]);

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
                *(OUT + 480000*index + i) =  _mm_cvtss_f32(t4);
           }
        }
    }
}
inline void vector_27xN27_loopbig(float * __restrict x, float * __restrict y, float * __restrict OUT, int N)
{
    __m256i mask = _mm256_castps_si256(_mm256_load_ps((const float*)mask8i[3]));
    int i=0;
    int index=0;
    x = (float*)__builtin_assume_aligned (x, 32);
    y = (float*)__builtin_assume_aligned (y, 32);
    register __m256i load_3_mask = _mm256_castps_si256(_mm256_load_ps((const float*)mask8i[3]));
//   omp_set_num_threads(8);
//#pragma omp parallel
    {
    //#pragma omp for schedule(static, 8)
        for(i=0;i<64;i++)
        {
            __m256  a1 = _mm256_loadu_ps(&x[0 + 27*i]);
            __m256  a2 = _mm256_loadu_ps(&x[8 + 27*i]);
            __m256  a3 = _mm256_loadu_ps(&x[16 + 27*i]);
            __m256  a4 = _mm256_maskload_ps(&x[24 + 27*i], load_3_mask);
           for(index=0;index<480000;index++)
           {
                __m256 b1 = _mm256_loadu_ps(&y[0 + 27*index]);
                __m256 b2 = _mm256_loadu_ps(&y[8 + 27*index]);
                __m256 b3 = _mm256_loadu_ps(&y[16 + 27*index]);
                __m256 b4 = _mm256_loadu_ps(&y[24 + 27*index]);

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
                *(OUT + i*480000 + index) =  _mm_cvtss_f32(t4);
           }
        }
    }
}

#define LOAD_DOT(i,OUT) do{\
        __m256  a1 = _mm256_loadu_ps(&x[0 + 27*i]);\
        __m256  a2 = _mm256_loadu_ps(&x[8 + 27*i]);\
        __m256  a3 = _mm256_loadu_ps(&x[16 + 27*i]);\
        __m256  a4 = _mm256_maskload_ps(&x[24 + 27*i], load_3_mask);\
        __m256 m1 = _mm256_mul_ps(a1, b1);\
        __m256 m2 = _mm256_mul_ps(a2, b2);\
        __m256 m3 = _mm256_mul_ps(a3, b3);\
        __m256 m4 = _mm256_mul_ps(a4, b4);\
        m1 = _mm256_add_ps(m1, m2);\
        m3 = _mm256_add_ps(m3, m4);\
        m1 = _mm256_add_ps(m1, m3);\
        __m256 t1 = _mm256_hadd_ps(m1, m1);\
        __m256 t2 = _mm256_hadd_ps(t1, t1);\
        __m128 t3 = _mm256_extractf128_ps(t2, 1);\
        __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2), t3);\
        *(OUT + 480000*i + index) =  _mm_cvtss_f32(t4);\
        }while (0)

inline void vector_27xN27_apart_loopbig(float * __restrict x, float * __restrict y, float * __restrict OUT)
{
    __m256i mask = _mm256_castps_si256(_mm256_load_ps((const float*)mask8i[3]));
    //int i=0;
    int index=0;
    x = (float*)__builtin_assume_aligned (x, 32);
    y = (float*)__builtin_assume_aligned (y, 32);
    register __m256i load_3_mask = _mm256_castps_si256(_mm256_load_ps((const float*)mask8i[3]));

   for(index=0;index<480000;index++)
   {
        __m256 b1 = _mm256_loadu_ps(&y[0 + 27*index]);
        __m256 b2 = _mm256_loadu_ps(&y[8 + 27*index]);
        __m256 b3 = _mm256_loadu_ps(&y[16 + 27*index]);
        __m256 b4 = _mm256_loadu_ps(&y[24 + 27*index]);

/*      __m256  a1 = _mm256_loadu_ps(&x[0 + 27*i]);
        __m256  a2 = _mm256_loadu_ps(&x[8 + 27*i]);
        __m256  a3 = _mm256_loadu_ps(&x[16 + 27*i]);
        __m256  a4 = _mm256_maskload_ps(&x[24 + 27*i], load_3_mask);
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
        *(OUT + 480000*i + index) =  _mm_cvtss_f32(t4); */
        LOAD_DOT(0 ,OUT);
        LOAD_DOT(1 ,OUT);
        LOAD_DOT(2 ,OUT);
        LOAD_DOT(3 ,OUT);
        LOAD_DOT(4 ,OUT);
        LOAD_DOT(5 ,OUT);
        LOAD_DOT(6 ,OUT);
        LOAD_DOT(7 ,OUT);  
        LOAD_DOT(8 ,OUT);
        LOAD_DOT(9 ,OUT);
        LOAD_DOT(10,OUT);
        LOAD_DOT(11,OUT);
        LOAD_DOT(12,OUT);
        LOAD_DOT(13,OUT);
        LOAD_DOT(14,OUT);
        LOAD_DOT(15,OUT); 
        LOAD_DOT(16,OUT);
        LOAD_DOT(17,OUT);
        LOAD_DOT(18,OUT);
        LOAD_DOT(19,OUT);
        LOAD_DOT(20,OUT);
        LOAD_DOT(21,OUT);
        LOAD_DOT(22,OUT);
        LOAD_DOT(23,OUT);  
        LOAD_DOT(24,OUT);
        LOAD_DOT(25,OUT);
        LOAD_DOT(26,OUT);
        LOAD_DOT(27,OUT);
        LOAD_DOT(28,OUT);
        LOAD_DOT(29,OUT);
        LOAD_DOT(30,OUT);
        LOAD_DOT(31,OUT);
        LOAD_DOT(32,OUT);
        LOAD_DOT(33,OUT);
        LOAD_DOT(34,OUT);
        LOAD_DOT(35,OUT);
        LOAD_DOT(36,OUT);
        LOAD_DOT(37,OUT);
        LOAD_DOT(38,OUT);
        LOAD_DOT(39,OUT);  
        LOAD_DOT(40,OUT);
        LOAD_DOT(41,OUT);
        LOAD_DOT(42,OUT);
        LOAD_DOT(43,OUT);
        LOAD_DOT(44,OUT);
        LOAD_DOT(45,OUT);
        LOAD_DOT(46,OUT);
        LOAD_DOT(47,OUT); 
        LOAD_DOT(48,OUT);
        LOAD_DOT(49,OUT);
        LOAD_DOT(50,OUT);
        LOAD_DOT(51,OUT);
        LOAD_DOT(52,OUT);
        LOAD_DOT(53,OUT);
        LOAD_DOT(54,OUT);
        LOAD_DOT(55,OUT);  
        LOAD_DOT(56,OUT);
        LOAD_DOT(57,OUT);
        LOAD_DOT(58,OUT);
        LOAD_DOT(59,OUT);
        LOAD_DOT(60,OUT);
        LOAD_DOT(61,OUT);
        LOAD_DOT(62,OUT);
        LOAD_DOT(63,OUT);        
   }
}

#define LOAD(i,x)\
        __m256  a##i##1 = _mm256_loadu_ps(&x[0 + 27*i]);\
        __m256  a##i##2 = _mm256_loadu_ps(&x[8 + 27*i]);\
        __m256  a##i##3 = _mm256_loadu_ps(&x[16 + 27*i]);\
        __m256  a##i##4 = _mm256_maskload_ps(&x[24 + 27*i], load_3_mask);
        
#define DOT(i,OUT) do{\
        __m256 m1 = _mm256_mul_ps(a##i##1, b1);\
        __m256 m2 = _mm256_mul_ps(a##i##2, b2);\
        __m256 m3 = _mm256_mul_ps(a##i##3, b3);\
        __m256 m4 = _mm256_mul_ps(a##i##4, b4);\
        m1 = _mm256_add_ps(m1, m2);\
        m3 = _mm256_add_ps(m3, m4);\
        m1 = _mm256_add_ps(m1, m3);\
        __m256 t1 = _mm256_hadd_ps(m1, m1);\
        __m256 t2 = _mm256_hadd_ps(t1, t1);\
        __m128 t3 = _mm256_extractf128_ps(t2, 1);\
        __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2), t3);\
        *(OUT + 480000*i + index) =  _mm_cvtss_f32(t4);\
        }while (0)

inline void vector_27xN27_onceload_apart_loopbig(float * __restrict x, float * __restrict y, float * __restrict OUT)
{
    __m256i mask = _mm256_castps_si256(_mm256_load_ps((const float*)mask8i[3]));
    int index=0;
    x = (float*)__builtin_assume_aligned (x, 32);
    y = (float*)__builtin_assume_aligned (y, 32);
    register __m256i load_3_mask = _mm256_castps_si256(_mm256_load_ps((const float*)mask8i[3]));
    LOAD(0 ,x);
    LOAD(1 ,x);
    LOAD(2 ,x);
    LOAD(3 ,x);
    LOAD(4 ,x);
    LOAD(5 ,x);
    LOAD(6 ,x);
    LOAD(7 ,x);  
    LOAD(8 ,x);
    LOAD(9 ,x);
    LOAD(10,x);
    LOAD(11,x);
    LOAD(12,x);
    LOAD(13,x);
    LOAD(14,x);
    LOAD(15,x); 
    LOAD(16,x);
    LOAD(17,x);
    LOAD(18,x);
    LOAD(19,x);
    LOAD(20,x);
    LOAD(21,x);
    LOAD(22,x);
    LOAD(23,x);  
    LOAD(24,x);
    LOAD(25,x);
    LOAD(26,x);
    LOAD(27,x);
    LOAD(28,x);
    LOAD(29,x);
    LOAD(30,x);
    LOAD(31,x);
    LOAD(32,x);
    LOAD(33,x);
    LOAD(34,x);
    LOAD(35,x);
    LOAD(36,x);
    LOAD(37,x);
    LOAD(38,x);
    LOAD(39,x);  
    LOAD(40,x);
    LOAD(41,x);
    LOAD(42,x);
    LOAD(43,x);
    LOAD(44,x);
    LOAD(45,x);
    LOAD(46,x);
    LOAD(47,x); 
    LOAD(48,x);
    LOAD(49,x);
    LOAD(50,x);
    LOAD(51,x);
    LOAD(52,x);
    LOAD(53,x);
    LOAD(54,x);
    LOAD(55,x);  
    LOAD(56,x);
    LOAD(57,x);
    LOAD(58,x);
    LOAD(59,x);
    LOAD(60,x);
    LOAD(61,x);
    LOAD(62,x);
    LOAD(63,x);
   for(index=0;index<480000;index++)
   {
        __m256 b1 = _mm256_loadu_ps(&y[0 + 27*index]);
        __m256 b2 = _mm256_loadu_ps(&y[8 + 27*index]);
        __m256 b3 = _mm256_loadu_ps(&y[16 + 27*index]);
        __m256 b4 = _mm256_loadu_ps(&y[24 + 27*index]);

        DOT(0 ,OUT);
        DOT(1 ,OUT);
        DOT(2 ,OUT);
        DOT(3 ,OUT);
        DOT(4 ,OUT);
        DOT(5 ,OUT);
        DOT(6 ,OUT);
        DOT(7 ,OUT);  
        DOT(8 ,OUT);
        DOT(9 ,OUT);
        DOT(10,OUT);
        DOT(11,OUT);
        DOT(12,OUT);
        DOT(13,OUT);
        DOT(14,OUT);
        DOT(15,OUT); 
        DOT(16,OUT);
        DOT(17,OUT);
        DOT(18,OUT);
        DOT(19,OUT);
        DOT(20,OUT);
        DOT(21,OUT);
        DOT(22,OUT);
        DOT(23,OUT);  
        DOT(24,OUT);
        DOT(25,OUT);
        DOT(26,OUT);
        DOT(27,OUT);
        DOT(28,OUT);
        DOT(29,OUT);
        DOT(30,OUT);
        DOT(31,OUT);
        DOT(32,OUT);
        DOT(33,OUT);
        DOT(34,OUT);
        DOT(35,OUT);
        DOT(36,OUT);
        DOT(37,OUT);
        DOT(38,OUT);
        DOT(39,OUT);  
        DOT(40,OUT);
        DOT(41,OUT);
        DOT(42,OUT);
        DOT(43,OUT);
        DOT(44,OUT);
        DOT(45,OUT);
        DOT(46,OUT);
        DOT(47,OUT); 
        DOT(48,OUT);
        DOT(49,OUT);
        DOT(50,OUT);
        DOT(51,OUT);
        DOT(52,OUT);
        DOT(53,OUT);
        DOT(54,OUT);
        DOT(55,OUT);  
        DOT(56,OUT);
        DOT(57,OUT);
        DOT(58,OUT);
        DOT(59,OUT);
        DOT(60,OUT);
        DOT(61,OUT);
        DOT(62,OUT);
        DOT(63,OUT);        
   }
}

int main()
{
    float* abuff = malloc(1728 * 4);
    float* bbuff = malloc(12960000 * 4);
    float* cbuff = malloc(30720000 * 4);
    //float* abuff = aligned_alloc(16, 1728*4);
    //float* bbuff = aligned_alloc(16, 12960000*4);
    //float* cbuff = aligned_alloc(16, 30720000*4);
    
    int fd_a, fd_b;
    clock_t start,finish;
    double int_time = 0;

    if((fd_a = fopen("./A","rb")) ==-1)
    {
        printf("A creat file wrong!");
    }
    if((fd_b = fopen("./B","rb")) ==-1)
    {
        printf("B creat file wrong!");
    }
    printf("A read size:%d \r\n",  fread(abuff, 4, 1728, fd_a));
    printf("B read size:%d  \r\n", fread(bbuff, 4, 12960000, fd_b));
    close(fd_a);
    close(fd_b);
    start=clock();
    vector_27xN27_looplittle(bbuff,abuff,cbuff,480000);
    finish=clock();
    int_time = (double)(finish-start)/CLOCKS_PER_SEC;
    printf("little vector_27xN27 time %f  cbuff[1480602]:%f\r\n", int_time, cbuff[1480602]);
    start=clock();
    vector_27xN27_loopbig(abuff,bbuff,cbuff,480000);
    finish=clock();
    int_time = (double)(finish-start)/CLOCKS_PER_SEC;
    printf("big vector_27xN27 time %f  cbuff[1480602]:%f\r\n", int_time, cbuff[1480602]);
    start=clock();
    vector_27xN27_apart_loopbig(abuff,bbuff,cbuff);
    finish=clock();
    int_time = (double)(finish-start)/CLOCKS_PER_SEC;
    printf("vector_27xN27_apart_loopbig  time %f  cbuff[1480602]:%f\r\n", int_time, cbuff[1480602]);
    start=clock();
    vector_27xN27_onceload_apart_loopbig(abuff,bbuff,cbuff);
    finish=clock();
    int_time = (double)(finish-start)/CLOCKS_PER_SEC;
    printf("vector_27xN27_onceload_apart_loopbig  time %f  cbuff[1480602]:%f\r\n", int_time, cbuff[1480602]);    
}
