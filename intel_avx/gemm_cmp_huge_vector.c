//gcc -w -msse4 -mavx -O3 gemm_cmp_huge_vector.c
/********************************************************
64*27 与 27*480000的矩阵乘法优化
A B两个文件按照一维数组形式存储两个矩阵
*********************************************************/
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <x86intrin.h>
#include <smmintrin.h>  /* SSE 4.1 */
#include <immintrin.h>  /* SSE 4.1 */
#include <math.h>
#include<time.h>
#include <omp.h>

void random_matrix(float*buf, int len)
{
    int i = 0;
    srand48(time(0));
    for ( i = 0; i < len; i++ )
    {
       buf[i]= 2.0 * (float)drand48() - 1.0;
    }
}

inline bool is_a_ge_zero_and_a_lt_b(int a, int b)
{
  return (unsigned)(a) < (unsigned)(b);
}

void caffe_im2col_cpu (const float* data_im,
                 const int channels,
                 const int height,
                 const int width,
                 const int kernel_h,
                 const int kernel_w,
                 const int pad_h,
                 const int pad_w,
                 const int stride_h,
                 const int stride_w,
                 const int dilation_h,
                 const int dilation_w,
                 float* data_col)
{
    clock_t start,finish;
    double int_time;

    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    start=clock();
    const int channel_size = height * width;
    for (int channel = 0; channel < channels; channel++,data_im += channel_size)
    {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
      {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
      {
        int input_row = -pad_h + kernel_row * dilation_h;

        int index_cc = channel*kernel_h*kernel_w + kernel_row*kernel_w + kernel_col;
        for (int output_rows = 0; output_rows<output_h; output_rows++)
        {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height))
          {
            for (int output_cols = 0; output_cols < output_w; output_cols++)
            {
           // int tmp = index_cc + (output_rows*output_w + output_cols)*(kernel_h*kernel_w*channels);
             // *(data_col+tmp) = 0;
              *(data_col++) = 0;
            }
          }
          else
          {
            int input_col = -pad_w + kernel_col * dilation_w;

            for (int output_cols = 0; output_cols<output_w; output_cols++)
            {
            int tmp = index_cc + (output_rows*output_w + output_cols)*(kernel_h*kernel_w*channels);
              if (is_a_ge_zero_and_a_lt_b(input_col, width))
              {

                //*(data_col+tmp) = data_im[input_row * width + input_col];
                *(data_col++) = data_im[input_row * width + input_col];
              }
              else
              {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
finish=clock();
int_time = (double)(finish-start)/CLOCKS_PER_SEC;
printf("##########caffe img2 clo used times:%f\r\n", int_time);
}

static inline float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col_cpu(const float* data_im,
     const int channels,  const int height,  const int width,
     const int ksize,  const int stride, const int pad,  float* data_col) 
{
    int c,h,w;
    const int height_col = (height + 2*pad - ksize) / stride + 1;
    const int width_col = (width + 2*pad - ksize) / stride + 1;
    clock_t start,finish;
    double int_time = 0;

    const int channels_col = channels * ksize * ksize;
start=clock();
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
finish=clock();
int_time = int_time + (double)(finish-start)/CLOCKS_PER_SEC;
printf("##########img2 cloi used times:%f\r\n", int_time);
}

void im2row_cpu(const float* data_im,
     const int channels,  const int height,  const int width,
     const int ksize,  const int stride, const int pad,  float* data_col)
{
    const int height_col = (height + 2*pad - ksize) / stride + 1;
    const int width_col = (width + 2*pad - ksize) / stride + 1;
    clock_t start,finish;
    double int_time = 0;

    const int channels_col = channels * ksize * ksize;
    int im_row, im_col, row_index, w_offset, h_offset, c_im;
    int c, h, w;
start=clock();
    for (c = 0; c < channels_col; ++c) {
        w_offset = c % ksize;
        h_offset = (c / ksize) % ksize;
        c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                im_row = h_offset + h * stride - pad;
                im_col = w_offset + w * stride - pad;
                row_index = channels_col*(h * width_col + w) + c;
                 __builtin_prefetch(&data_col[row_index + channels_col], 1, 1);
                if (((unsigned)(im_row) < (unsigned)(height)) && ((unsigned)(im_col) < (unsigned)(width)))
                {
                    data_col[row_index] = data_im[im_col + width*(im_row + height*c_im)];
                }
                else
                {
                    data_col[row_index] = 0; 
                }
            }
        }
    }
finish=clock();
int_time = int_time + (double)(finish-start)/CLOCKS_PER_SEC;
printf("##########img2 row used times:%f\r\n", int_time);
}

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

#define savefile(name, buffer, size) do\
{\
  FILE *out = fopen(name, "wb");\
  if(out != NULL)\
  {\
        fwrite (buffer , sizeof(char), size, out);\
        fclose (out);\
  }\
} while(0)

#define show32(adr) printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\r\n",\
(adr)[0],(adr)[1],(adr)[2],(adr)[3],(adr)[4],(adr)[5],(adr)[6],(adr)[7],(adr)[8],(adr)[9],\
(adr)[10],(adr)[11],(adr)[12],(adr)[13],(adr)[14],(adr)[15],(adr)[16],(adr)[17],(adr)[18],(adr)[19],\
(adr)[20],(adr)[21],(adr)[22],(adr)[23],(adr)[24],(adr)[25],(adr)[26],(adr)[27],(adr)[28],(adr)[29],\
(adr)[30],(adr)[31])

#define show16(adr) printf("%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\r\n",\
(adr)[0],(adr)[1],(adr)[2],(adr)[3],(adr)[4],(adr)[5],(adr)[6],(adr)[7],(adr)[8],(adr)[9],\
(adr)[10],(adr)[11],(adr)[12],(adr)[13],(adr)[14],(adr)[15])

#define show_mem_float(addr, size) do{\
    int i;\
    for(i=0;i<size;i=i+16)\
    {\
        show16((addr+i));\
    }\
}while(0)

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
                __builtin_prefetch(&OUT[480000*(index + 1) + i], 1, 1);
                *(OUT + 480000*index + i) =  _mm_cvtss_f32(t4);
           }
        }
    }
}

inline void vector_27xN27_looplittle_sum(float * __restrict x, float * __restrict y, float * __restrict OUT, int N)
{
    __m256i mask = _mm256_castps_si256(_mm256_load_ps((const float*)mask8i[3]));
    int i=0;
    int index=0;
    x = (float*)__builtin_assume_aligned (x, 32);
    y = (float*)__builtin_assume_aligned (y, 32);
    register __m256i load_3_mask = _mm256_castps_si256(_mm256_load_ps((const float*)mask8i[3]));
    const float* sum;
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
                
                __builtin_prefetch(&OUT[480000*(index + 1) + i], 1, 1);
                sum = (const float*)&m1;
                *(OUT + 480000*index + i) = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
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

#define DOT(i,OUT) do{\
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

void gemm_nn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
//omp_set_num_threads(8);
{
    int i,j,k;
//#pragma omp for schedule(dynamic)
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}
}

void gemm_nt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_tn(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}


void gemm_tt(int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

int main()
{
    float* abuff = malloc(1728 * 4);
    float* bbuff = malloc(12960000 * 4);
    float* cbuff = malloc(30720000 * 4);
    float* img_buff = malloc(480000 * 4 * 3);
    
    random_matrix(img_buff, 3*600*800);
    random_matrix(abuff, 1728);
    random_matrix(bbuff, 12960000);
    
    im2col_cpu(img_buff,3,600,800,3,1,1,bbuff);
    caffe_im2col_cpu(img_buff,3,600,800,3,3,1,1,1,1,1,1,bbuff);

    clock_t start,finish;
    double int_time = 0;
    start=clock();
    gemm (0, 0, 64, 480000, 27, 1, abuff, 27, bbuff, 480000, 0, cbuff,480000 );
    finish=clock();
    int_time = (double)(finish-start)/CLOCKS_PER_SEC;
    printf("native c gemm                time:%f  cbuff[1480602]:%f\r\n", int_time, cbuff[1480602]);

    im2row_cpu(img_buff,3,600,800,3,1,1,bbuff);

    start=clock();
    vector_27xN27_looplittle(bbuff,abuff,cbuff,480000);
    finish=clock();
    int_time = (double)(finish-start)/CLOCKS_PER_SEC;
    printf("little vector_27xN27         time:%f  cbuff[1480602]:%f\r\n", int_time, cbuff[1480602]);
    start=clock();
    vector_27xN27_looplittle_sum(bbuff,abuff,cbuff,480000);
    finish=clock();
    int_time = (double)(finish-start)/CLOCKS_PER_SEC;
    printf("little vector_27xN27 sum     time:%f  cbuff[1480602]:%f\r\n", int_time, cbuff[1480602]);
    start=clock();
    vector_27xN27_loopbig(abuff,bbuff,cbuff,480000);
    finish=clock();
    int_time = (double)(finish-start)/CLOCKS_PER_SEC;
    printf("big vector_27xN27            time:%f  cbuff[1480602]:%f\r\n", int_time, cbuff[1480602]);
    start=clock();
    vector_27xN27_apart_loopbig(abuff,bbuff,cbuff);
    finish=clock();
    int_time = (double)(finish-start)/CLOCKS_PER_SEC;
    printf("vector_27xN27_apart_loopbig  time:%f  cbuff[1480602]:%f\r\n", int_time, cbuff[1480602]);
}
