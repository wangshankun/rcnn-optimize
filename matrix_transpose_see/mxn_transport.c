#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <x86intrin.h>
#include <immintrin.h>  /* SSE 4.1 */
#include <math.h>
#include<error.h>
#include<fcntl.h>
#include<poll.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<sys/mman.h>

inline void transpose4x4_SSE(int x, int k_delta, int lda, int y, int y_delta, int ldb, float* A, float* B, int i, int j)
{
    __m128 row1 = _mm_loadu_ps(&A[(y + 0 + i) * lda + x + j]);
    __m128 row2 = _mm_loadu_ps(&A[(y + 1 + i) * lda + x + j]);
    __m128 row3 = _mm_loadu_ps(&A[(y + 2 + i) * lda + x + j]);
    __m128 row4 = _mm_loadu_ps(&A[(y + 3 + i) * lda + x + j]);
     _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
     _mm_storeu_ps(&B[(j + 0) * y_delta + i], row1);
     _mm_storeu_ps(&B[(j + 1) * y_delta + i], row2);
     _mm_storeu_ps(&B[(j + 2) * y_delta + i], row3);
     _mm_storeu_ps(&B[(j + 3) * y_delta + i], row4);
}

void transpose_remainder(int x, int y, int j_from, int x_end, int i_form, int y_end, int lda, int y_delta, float* A, float* B)
{
    int ii,jj;
    for(ii = i_form; ii < y_end; ii++)
    {
        for(jj = j_from; jj < x_end; jj++)
        {
            B[jj * y_delta + ii] = A[(y + ii) * lda + x + jj];
        }
    }
}  
    
inline void R_T_C_COPY_OPERATION(int x, int k_delta, int lda, int y, int y_delta, int ldb, float* src, float* dts)
{
    int i, j;

    int x_remainderint = k_delta%4;
    int y_remainderint = y_delta%4;
    
    int kk_delta = (k_delta/4)*4;
    int yy_delta = (y_delta/4)*4;

    for(i = 0; i < yy_delta; i+=4)
    {
        for(j = 0; j < kk_delta; j+=4)
        { 
            transpose4x4_SSE(x, k_delta, lda, y, y_delta, ldb, src, dts, i ,j);
        }
        transpose_remainder(x, y, j, k_delta, i , i + 4, lda, y_delta, src, dts);
    }
    transpose_remainder(x, y, 0, k_delta, i , y_delta, lda, y_delta, src, dts);
}

//从8*256矩阵中取在位第0行,4列的位置出一块7*15数据然后转置
//
void main()
{
//    struct timespec start, finish;
  //  double elapsed;
    int i=0;
    float* A = malloc(8*256*4);
    float* B = malloc(7*15*4);
    for(i=0;i<8*256;i++)
    {
        if(i%256==0)printf("\r\n");
        *(A+i)=i;
        printf("%04.0f ",*(A+i));
    }
    printf("\r\n");
    R_T_C_COPY_OPERATION(4,15,256,0,7,8,A,B);
    for(i=0;i<7*15;i++)
    {
        if(i%7==0)printf("\r\n");
	printf("%04.0f ",*(B+i));
    }
    printf("\r\n");
    free(A);
    free(B);
}
