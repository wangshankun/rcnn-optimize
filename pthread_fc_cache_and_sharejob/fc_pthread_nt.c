#define _GNU_SOURCE
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
#include"cblas.h"

#define BLASLONG long
#define FLOAT    float
#define YIELDING sched_yield()
#define GEMM_P  512
#define GEMM_Q  64
#define SGEMM_P GEMM_P
#define SGEMM_Q GEMM_Q
#define GEMM_ALIGN 16383
#define CACHE_LINE_SIZE 8
#define GEMM_UNROLL_N 4
#define GEMM_UNROLL_M 16
#define THREAD_STATUS_SLEEP 2
#define THREAD_STATUS_WAKEUP 4
#define MAX_CPU_NUMBER 8
#define MAX_SUB_PTHREAD_INDEX 7
#define COMPSIZE 1

#define BUFFER_SIZE   (8 << 20)


#define DP printf("%s  %d\r\n",__FUNCTION__,__LINE__);

typedef struct {
  volatile BLASLONG working[MAX_CPU_NUMBER][CACHE_LINE_SIZE];
} ww_job_t;


typedef struct {
  FLOAT *a, *b, *c;
  BLASLONG m, n, k;
  BLASLONG nthreads;
  void *common;
} ww_arg_t;

typedef struct blas_queue {
  void *routine;
  volatile int assigned;
  void *sa, *sb;
  ww_arg_t *args;
} ww_queue_t;

typedef struct {
    ww_queue_t * volatile queue  __attribute__((aligned(32)));
    volatile long status;
    pthread_mutex_t lock;
    pthread_cond_t wakeup;
} ww_thread_status_t;

#define MMAP_ACCESS (PROT_READ | PROT_WRITE)
#define MMAP_POLICY (MAP_PRIVATE | MAP_ANONYMOUS)

#define MB
#define WMB


static ww_thread_status_t thread_status[MAX_CPU_NUMBER] __attribute__((aligned(128)));
static pthread_t     ww_threads_id [MAX_CPU_NUMBER];
static ww_queue_t    queue[MAX_CPU_NUMBER];
static BLASLONG      range_M[MAX_CPU_NUMBER + 1];
static BLASLONG      range_N[MAX_CPU_NUMBER + 1];
//static ww_job_t         job[MAX_CPU_NUMBER];
static ww_job_t *  job = NULL;
static ww_arg_t    execute_arg;

void   sub_pthread_exec(void);

static void pthread_bind(int cpu)
{
    cpu_set_t mask; 
    CPU_ZERO(&mask);      
    CPU_SET(cpu, &mask);  
    if(sched_setaffinity(0, sizeof(mask), &mask) == -1)  
    {  
        printf("set affinity failed..");  
    }
}

inline void COPY_OPERATION(int k_delta, int x, int y, int y_delta, float* a, float* sa, int lda)
{
    int i;
    for(i = 0; i < y_delta; i++)
    {
        memcpy(sa + i*k_delta, a + (y + i)*lda + x, k_delta*sizeof(float));
    }
}

inline void transpose4x4_SSE(float *A, float *B, int x, int lda, int y, int ldb)
{
    __m128 row1 = _mm_load_ps(&A[y * lda + x]);
    __m128 row2 = _mm_load_ps(&A[(y + 1) * lda + x]);
    __m128 row3 = _mm_load_ps(&A[(y + 2) * lda + x]);
    __m128 row4 = _mm_load_ps(&A[(y + 3) * lda + x]);
     _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
     _mm_store_ps(&B[x * ldb + y], row1);
     _mm_store_ps(&B[(x + 1) * ldb + y], row2);
     _mm_store_ps(&B[(x + 2) * ldb + y], row3);
     _mm_store_ps(&B[(x + 3) * ldb + y], row4);
}
inline void R_T_C_COPY_OPERATION(int k_delta, int x, int y, int y_delta, float* b, float* sb, int lda)
{
    int i,j;
    for(i = 0; i < y_delta; i+=4)
    {
        for(j = 0; j < k_delta; j+=4)
        {
            transpose4x4_SSE(b + y * lda + x, sb, j, k_delta, i, y_delta);
        }
    }
}

inline void KERNEL_OPERATION(int k_delta, int ym, int m_delta, int yn, int n_delta, float* sa, float* sb, float* c, int ldc)
{
    int i,j,k;
    __m256 a,b;
    __m256 a1,a2,a3,a4,a5,a6,a7,a8,b1,b2,b3,b4,b5,b6,b7,b8;
    for(i = 0; i < m_delta; i++)
    {
        for(j = 0; j < n_delta; j+=8)
        {
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();
            __m256 sum4 = _mm256_setzero_ps();
            __m256 sum5 = _mm256_setzero_ps();
            __m256 sum6 = _mm256_setzero_ps();
            __m256 sum7 = _mm256_setzero_ps();
            __m256 sum8 = _mm256_setzero_ps();

            for(k = 0; k < k_delta; k+=8)
            {
                a1 = _mm256_broadcast_ss(&sa[i*k_delta + k + 0]);
                b1 = _mm256_loadu_ps(&sb[k*8]);
                sum1 = _mm256_fmadd_ps(a1, b1, sum1);

                a2 = _mm256_broadcast_ss(&sa[i*k_delta + k + 1]);
                b2 = _mm256_loadu_ps(&sb[(k + 1)*8]);
                sum2 = _mm256_fmadd_ps(a2, b2, sum2);

                a3 = _mm256_broadcast_ss(&sa[i*k_delta + k + 2]);
                b3 = _mm256_loadu_ps(&sb[(k + 2)*8]);
                sum3 = _mm256_fmadd_ps(a3, b3, sum3);

                a4 = _mm256_broadcast_ss(&sa[i*k_delta + k + 3]);
                b4 = _mm256_loadu_ps(&sb[(k + 3)*8]);
                sum4 = _mm256_fmadd_ps(a4, b4, sum4);
                
                a5 = _mm256_broadcast_ss(&sa[i*k_delta + k + 4]);
                b5 = _mm256_loadu_ps(&sb[(k + 4)*8]);
                sum5 = _mm256_fmadd_ps(a5, b5, sum5);

                a6 = _mm256_broadcast_ss(&sa[i*k_delta + k + 5]);
                b6 = _mm256_loadu_ps(&sb[(k + 5)*8]);
                sum6 = _mm256_fmadd_ps(a6, b6, sum6);

                a7 = _mm256_broadcast_ss(&sa[i*k_delta + k + 6]);
                b7 = _mm256_loadu_ps(&sb[(k + 6)*8]);
                sum7 = _mm256_fmadd_ps(a7, b7, sum7);

                a8 = _mm256_broadcast_ss(&sa[i*k_delta + k + 7]);
                b8 = _mm256_loadu_ps(&sb[(k + 7)*8]);
                sum8 = _mm256_fmadd_ps(a8, b8, sum8);
            }
            _mm256_storeu_ps(&c[ldc*(ym + i) + yn + j],(sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8));
        }
    }
}


#define PRINT_TIME
typedef int (*ROUTINE)(BLASLONG);
static inline int inner_thread(BLASLONG mypos)
{
    struct timespec start, finish;
    double elapsed1, elapsed2, elapsed3, elapsed4;

    BLASLONG m_from, m_to, n_from, n_to, N_from, N_to;
    BLASLONG lda, ldb, ldc;
    BLASLONG i, current;

    FLOAT *a, *b, *c, *alpha, *beta;
    BLASLONG m, n, k;
    
    FLOAT ALP = 1;
    FLOAT BET = 0;

    FLOAT *sa     = queue[mypos].sa;
    FLOAT *buffer = queue[mypos].sb;
    
    a = execute_arg.a;
    b = execute_arg.b;
    c = execute_arg.c;
    m = execute_arg.m;
    n = execute_arg.n;
    k = execute_arg.k;
    
    m_from = range_M[mypos + 0];
    m_to   = range_M[mypos + 1];

    n_from = range_N[mypos + 0];
    n_to   = range_N[mypos + 1];

    N_from = range_N[0];
    N_to   = range_N[MAX_CPU_NUMBER];
  
    int x,ym,yn,m_delta;

    int n_delta  = 32;
    int k_delta  = GEMM_Q;

    for(x = 0; x < k; x += k_delta)
    {
        m_delta = m_to - m_from;
        ym = m_from;
#ifdef PRINT_TIME
        clock_gettime(CLOCK_MONOTONIC, &start);
#endif
        COPY_OPERATION(k_delta, x, ym, m_delta, a, sa, k);
#ifdef PRINT_TIME
        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed1 += (finish.tv_sec - start.tv_sec);
        elapsed1 += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif
        for (i = 0; i < MAX_CPU_NUMBER; i++) while(job[mypos].working[i][CACHE_LINE_SIZE]) {YIELDING;}; 
        //n_delta = 32;
        for(yn = n_from; yn < n_to; yn += n_delta)
        {
#ifdef PRINT_TIME
            clock_gettime(CLOCK_MONOTONIC, &start);
#endif
            R_T_C_COPY_OPERATION(k_delta, x, yn, n_delta, b, buffer + (yn - n_from)*k_delta, k);
#ifdef PRINT_TIME
            clock_gettime(CLOCK_MONOTONIC, &finish);
            elapsed2 += (finish.tv_sec - start.tv_sec);
            elapsed2 += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
            clock_gettime(CLOCK_MONOTONIC, &start);
#endif
            KERNEL_OPERATION(k_delta, ym, m_delta, yn, n_delta, sa, buffer + (yn - n_from)*k_delta, c, n);
#ifdef PRINT_TIME
            clock_gettime(CLOCK_MONOTONIC, &finish);
            elapsed3 += (finish.tv_sec - start.tv_sec);
            elapsed3 += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif
        }

        for (i = 0; i < MAX_CPU_NUMBER; i++) job[mypos].working[i][CACHE_LINE_SIZE] = (BLASLONG)buffer;

        current = mypos;
        do
        {
            current++; if (current >= MAX_CPU_NUMBER) current = 0;
            if (current != mypos)
            {
                while(job[current].working[mypos][CACHE_LINE_SIZE] == 0) {YIELDING;};
                //n_delta = 32;
                for(yn = range_N[current]; yn < range_N[current + 1]; yn += n_delta)
                {
#ifdef PRINT_TIME
                    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
                    KERNEL_OPERATION(k_delta, ym, m_delta, yn, n_delta, sa, (FLOAT *)(job[current].working[mypos][CACHE_LINE_SIZE]) + (yn - range_N[current])*k_delta, c, n);
#ifdef PRINT_TIME
                    clock_gettime(CLOCK_MONOTONIC, &finish);
                    elapsed4  += (finish.tv_sec - start.tv_sec);
                    elapsed4  += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif
                }
            }
            job[current].working[mypos][CACHE_LINE_SIZE] &= 0;
        } while (current != mypos);

    }
    for (i = 0; i < MAX_CPU_NUMBER; i++)  while (job[mypos].working[i][CACHE_LINE_SIZE] ) {YIELDING;};
#ifdef PRINT_TIME
    printf("mypos:%d elapsed1:%f elapsed2:%f elapsed3:%f elapsed4:%f\r\n",mypos,elapsed1,elapsed2,elapsed3,elapsed4);
#endif
}

void divide(BLASLONG M, BLASLONG* range_M)
{
    int dx = M%MAX_CPU_NUMBER;
    int dy = M/MAX_CPU_NUMBER;
    int index = 0;
    int i;
    for(i = 0;i < MAX_CPU_NUMBER + 1; i++)
    {
        range_M[i] = index;
        if(i < dx)
        {
            index = index + dy + 1;
        }
        else
        {
            index = index + dy;
        }
    }
}

void ww_sgemm_thread_nt(float* A, float* B, float* C, BLASLONG M, BLASLONG N, BLASLONG K)
{
    int i;
    execute_arg.a        = A;
    execute_arg.b        = B;
    execute_arg.c        = C;
    execute_arg.m        = M;
    execute_arg.n        = N;
    execute_arg.k        = K;

    divide(execute_arg.m, range_M);
    divide(execute_arg.n, range_N);

    sub_pthread_exec();
    
    ((ROUTINE)(queue[0].routine))(0);
    queue[0].assigned = 0;

    for (i = 0; i < MAX_CPU_NUMBER; i++) while (queue[i].assigned) {YIELDING;};
}

void sub_pthread_exec(void)
{
    int pthread_pos;    
    for(pthread_pos = 1; pthread_pos < MAX_CPU_NUMBER; pthread_pos++)
    {
        if (thread_status[pthread_pos].status == THREAD_STATUS_SLEEP) 
        {
            pthread_mutex_lock  (&thread_status[pthread_pos].lock);
            thread_status[pthread_pos].status = THREAD_STATUS_WAKEUP;
            pthread_cond_signal(&thread_status[pthread_pos].wakeup);
            pthread_mutex_unlock(&thread_status[pthread_pos].lock);
        }
    }
}

static void* sub_pthread_body(void *arg)
{
    int  pthread_pos = (int)arg;
    pthread_mutex_lock  (&thread_status[pthread_pos].lock);

    while (thread_status[pthread_pos].status == THREAD_STATUS_SLEEP)
    {
        pthread_cond_wait(&thread_status[pthread_pos].wakeup, &thread_status[pthread_pos].lock);
    }
    pthread_mutex_unlock(&thread_status[pthread_pos].lock);
    ((ROUTINE)(queue[pthread_pos].routine))(pthread_pos);
    queue[pthread_pos].assigned = 0;
    thread_status[pthread_pos].status = THREAD_STATUS_SLEEP;
}

void sub_pthread_init(void)
{
    int i, j, pthread_pos;
    job = (ww_job_t*)malloc(MAX_CPU_NUMBER * sizeof(ww_job_t));
    if(job == NULL)
    {
        fprintf(stderr, "OpenBLAS: malloc failed in %s\n", __func__);
        exit(1);
    } 

    for (i = 0; i < MAX_CPU_NUMBER; i++)
    {
        queue[i].sa       = mmap(NULL, BUFFER_SIZE, MMAP_ACCESS, MMAP_POLICY, -1, 0);
        queue[i].sb       = (void *)(((BLASLONG)(queue[i].sa) + ((SGEMM_P * SGEMM_Q * sizeof(float) + GEMM_ALIGN) & ~GEMM_ALIGN)));
        queue[i].assigned = i + 1;
        queue[i].routine  = inner_thread;
        
        for (j = 0; j < MAX_CPU_NUMBER; j++)
        {
            job[i].working[j][CACHE_LINE_SIZE] = 0;
        }
    }

    for(pthread_pos = 1; pthread_pos < MAX_CPU_NUMBER; pthread_pos++)
    {
        pthread_mutex_init(&thread_status[pthread_pos].lock, NULL);
        pthread_cond_init (&thread_status[pthread_pos].wakeup, NULL);
        thread_status[pthread_pos].status = THREAD_STATUS_SLEEP;
        pthread_create(&ww_threads_id[pthread_pos], NULL, &sub_pthread_body, (void *)pthread_pos);
    }
}

void sub_pthread_exit(void)
{
    int i;
    for (i = 0; i < MAX_CPU_NUMBER; i++)
    {
        munmap(queue[i].sa, BUFFER_SIZE);
    }
}

int main()
{
    float* abuff = malloc(300 * 9216 * 4);
    float* bbuff = malloc(4096 * 9216 * 4);
    float* cbuff = malloc(300 * 4096 * 4);
    int fd_a, fd_b;

    struct timespec start, finish;
    double elapsed;  
    if((fd_a = fopen("./bottom_data","rb")) ==-1)
    {
        printf("A creat file wrong!");
    }
    if((fd_b = fopen("./weight","rb")) ==-1)
    {
        printf("B creat file wrong!");
    }
    printf("A read size:%d \r\n",  fread(abuff, 4, 300 * 9216, fd_a));
    printf("B read size:%d  \r\n", fread(bbuff, 4, 4096 * 9216, fd_b));
    close(fd_a);
    close(fd_b);

    sub_pthread_init();

    clock_gettime(CLOCK_MONOTONIC, &start);
    {
        ww_sgemm_thread_nt(abuff,bbuff,cbuff,300,4096,9216);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("ww_sgemm_thread_nt elapsed time:%f cbuff[3580602]:%f\r\n",elapsed,cbuff[3580602]);

    sub_pthread_exit();

    free(abuff);
    free(bbuff);
    free(cbuff);
    
    return 0;
}
