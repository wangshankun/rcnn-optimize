#define SIZE    4
#define BASE_SHIFT 2
#define ZBASE_SHIFT 3

#define HALT hlt

#ifndef ALIGN_2
#define ALIGN_2 .align 4
#endif

#ifndef ALIGN_3
#define ALIGN_3 .align 8
#endif

#ifndef ALIGN_4
#define ALIGN_4 .align 16
#endif

#ifndef ALIGN_5
#define ALIGN_5 .align 32
#endif

#ifndef ALIGN_6
#define ALIGN_6 .align 64
#endif

#ifndef ffreep
#define ffreep .byte 0xdf, 0xc0 #
#endif

#define FLD	flds
#define FST	fstps
#define FSTU	fsts
#define FMUL	fmuls
#define FADD	fadds
#define MOVSD	movss
#define MULSD	mulss
#define MULPD	mulps
#define CMPEQPD	cmpeqps
#define COMISD	comiss
#define PSRLQ	psrld
#define ANDPD	andps
#define ADDPD	addps
#define ADDSD	addss
#define SUBPD	subps
#define SUBSD	subss
#define MOVQ	movd
#define MOVUPD	movups
#define XORPD	xorps

#ifndef ASSEMBLER
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

#define BLASLONG long
#define FLOAT    float
#define YIELDING sched_yield()
//#define GEMM_P 768
//#define GEMM_Q 384
#define GEMM_P 512
#define GEMM_Q 256
#define SGEMM_P GEMM_P
#define SGEMM_Q GEMM_Q
#define GEMM_ALIGN 16383
#define CACHE_LINE_SIZE 8
#define GEMM_UNROLL_N 4
#define GEMM_UNROLL_M 16
#define THREAD_STATUS_SLEEP 2
#define THREAD_STATUS_WAKEUP 4
#define MAX_CPU_NUMBER_ 4
#define MAX_SUB_PTHREAD_INDEX 3
#define COMPSIZE 1

int sgemm_kernel(BLASLONG, BLASLONG, BLASLONG, float,  float  *, float  *, float  *, BLASLONG);
int sgemm_oncopy(BLASLONG m, BLASLONG n, float *a, BLASLONG lda, float *b);
int sgemm_itcopy(BLASLONG m, BLASLONG n, float *a, BLASLONG lda, float *b);
int sgemm_beta(BLASLONG, BLASLONG, BLASLONG, float, float  *, BLASLONG, float   *, BLASLONG, float  *, BLASLONG);

FLOAT * SHOW;
#define BETA_OPERATION(M_FROM, M_TO, N_FROM, N_TO, BETA, C, LDC) \
    sgemm_beta((M_TO) - (M_FROM), (N_TO - N_FROM), 0, \
          BETA[0], NULL, 0, NULL, 0, \
          (FLOAT *)(C) + ((M_FROM) + (N_FROM) * (LDC)) * COMPSIZE, LDC)

#define ICOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER)\
    sgemm_itcopy(M, N, (FLOAT *)(A) + ((Y) + (X) * (LDA)) * COMPSIZE, LDA, BUFFER);

#define OCOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER)\
    sgemm_oncopy(M, N, (FLOAT *)(A) + ((X) + (Y) * (LDA)) * COMPSIZE, LDA, BUFFER);

#define KERNEL_OPERATION(M, N, K, ALPHA, SA, SB, C, LDC, X, Y) \
        sgemm_kernel(M, N, K, ALPHA[0], SA, SB, (FLOAT *)(C) + ((X) + (Y) * LDC) * COMPSIZE, LDC)


typedef struct {
  volatile BLASLONG working[MAX_CPU_NUMBER_][CACHE_LINE_SIZE];
} job_t;

typedef struct {
  FLOAT *a, *b, *c;
  BLASLONG m, n, k;
  BLASLONG nthreads;
  void *common;
} blas_arg_t;

typedef struct blas_queue {
  void *routine;
  volatile int assigned;
  void *sa, *sb;
  blas_arg_t *args;
} blas_queue_t;

typedef struct {
    blas_queue_t * volatile queue  __attribute__((aligned(32)));
    volatile long status;
    pthread_mutex_t lock;
    pthread_cond_t wakeup;
} thread_status_t;

#define MMAP_ACCESS (PROT_READ | PROT_WRITE)
#define MMAP_POLICY (MAP_PRIVATE | MAP_ANONYMOUS)

#define MB
#define WMB

#ifdef ENABLE_SSE_EXCEPTION

#define IDEBUG_START \
{ \
  unsigned int fp_sse_mode, new_fp_mode; \
  __asm__ __volatile__ ("stmxcsr %0" : "=m" (fp_sse_mode) : ); \
  new_fp_mode = fp_sse_mode & ~0xd00; \
  __asm__ __volatile__ ("ldmxcsr %0" : : "m" (new_fp_mode) );

#define IDEBUG_END \
  __asm__ __volatile__ ("ldmxcsr %0" : : "m" (fp_sse_mode) ); \
}

#endif

#ifdef XDOUBLE
#define GET_IMAGE(res)  __asm__ __volatile__("fstpt %0" : "=m"(res) : : "memory")
#elif defined(DOUBLE)
#define GET_IMAGE(res)  __asm__ __volatile__("movsd %%xmm1, %0" : "=m"(res) : : "memory")
#else
#define GET_IMAGE(res)  __asm__ __volatile__("movss %%xmm1, %0" : "=m"(res) : : "memory")
#endif

#ifndef PAGESIZE
#define PAGESIZE      ( 4 << 10)
#endif
#define HUGE_PAGESIZE ( 2 << 20)

#define BUFFER_SIZE   ( 2 << 20)

#endif

#ifdef ASSEMBLER

#if defined(PILEDRIVER) || defined(BULLDOZER) || defined(STEAMROLLER) || defined(EXCAVATOR)
//Enable some optimazation for barcelona.
#define BARCELONA_OPTIMIZATION
#endif

#if defined(HAVE_3DNOW)
#define EMMS	femms
#elif defined(HAVE_MMX)
#define EMMS	emms
#endif

#ifndef EMMS
#define EMMS
#endif

#define BRANCH		.byte 0x3e
#define NOBRANCH	.byte 0x2e
#define PADDING		.byte 0x66

#ifdef OS_WINDOWS
#define ARG1	%rcx
#define ARG2	%rdx
#define ARG3	%r8
#define ARG4	%r9
#else
#define ARG1	%rdi
#define ARG2	%rsi
#define ARG3	%rdx
#define ARG4	%rcx
#define ARG5	%r8
#define ARG6	%r9
#endif

#ifndef COMPLEX
#ifdef XDOUBLE
#define LOCAL_BUFFER_SIZE  QLOCAL_BUFFER_SIZE
#elif defined DOUBLE
#define LOCAL_BUFFER_SIZE  DLOCAL_BUFFER_SIZE
#else
#define LOCAL_BUFFER_SIZE  SLOCAL_BUFFER_SIZE
#endif
#else
#ifdef XDOUBLE
#define LOCAL_BUFFER_SIZE  XLOCAL_BUFFER_SIZE
#elif defined DOUBLE
#define LOCAL_BUFFER_SIZE  ZLOCAL_BUFFER_SIZE
#else
#define LOCAL_BUFFER_SIZE  CLOCAL_BUFFER_SIZE
#endif
#endif

#if defined(OS_WINDOWS)
#if   LOCAL_BUFFER_SIZE > 16384
#define STACK_TOUCHING \
	movl	$0,  4096 * 4(%rsp);\
	movl	$0,  4096 * 3(%rsp);\
	movl	$0,  4096 * 2(%rsp);\
	movl	$0,  4096 * 1(%rsp);
#elif LOCAL_BUFFER_SIZE > 12288
#define STACK_TOUCHING \
	movl	$0,  4096 * 3(%rsp);\
	movl	$0,  4096 * 2(%rsp);\
	movl	$0,  4096 * 1(%rsp);
#elif LOCAL_BUFFER_SIZE > 8192
#define STACK_TOUCHING \
	movl	$0,  4096 * 2(%rsp);\
	movl	$0,  4096 * 1(%rsp);
#elif LOCAL_BUFFER_SIZE > 4096
#define STACK_TOUCHING \
	movl	$0,  4096 * 1(%rsp);
#else
#define STACK_TOUCHING
#endif
#else
#define STACK_TOUCHING
#endif

#if defined(CORE2)
#define movapd	movaps
#define andpd	andps
#define movlpd	movlps
#define movhpd	movhps
#endif

#ifndef F_INTERFACE
#define REALNAME ASMNAME
#else
#define REALNAME ASMFNAME
#endif

#ifdef OS_DARWIN
#define PROLOGUE .text;.align 5; .globl REALNAME; REALNAME:
#define EPILOGUE	.subsections_via_symbols
#define PROFCODE
#endif

#ifdef OS_WINDOWS
#define SAVEREGISTERS \
	subq	$256, %rsp;\
	movups	%xmm6,    0(%rsp);\
	movups	%xmm7,   16(%rsp);\
	movups	%xmm8,   32(%rsp);\
	movups	%xmm9,   48(%rsp);\
	movups	%xmm10,  64(%rsp);\
	movups	%xmm11,  80(%rsp);\
	movups	%xmm12,  96(%rsp);\
	movups	%xmm13, 112(%rsp);\
	movups	%xmm14, 128(%rsp);\
	movups	%xmm15, 144(%rsp)

#define RESTOREREGISTERS \
	movups	   0(%rsp), %xmm6;\
	movups	  16(%rsp), %xmm7;\
	movups	  32(%rsp), %xmm8;\
	movups	  48(%rsp), %xmm9;\
	movups	  64(%rsp), %xmm10;\
	movups	  80(%rsp), %xmm11;\
	movups	  96(%rsp), %xmm12;\
	movups	 112(%rsp), %xmm13;\
	movups	 128(%rsp), %xmm14;\
	movups	 144(%rsp), %xmm15;\
	addq	$256, %rsp
#else
#define SAVEREGISTERS
#define RESTOREREGISTERS
#endif

#if defined(OS_WINDOWS) && !defined(C_PGI)
#define PROLOGUE \
	.text; \
	.align 16; \
	.globl REALNAME ;\
	.def REALNAME;.scl	2;.type	32;.endef; \
REALNAME:

#define PROFCODE

#define EPILOGUE .end
#endif

#if defined(OS_LINUX) || defined(OS_FREEBSD) || defined(OS_NETBSD) || defined(__ELF__) || defined(C_PGI)
#define PROLOGUE \
	.text; \
	.align 512; \
	.globl REALNAME ;\
       .type REALNAME, @function; \
REALNAME:

#ifdef PROFILE
#define PROFCODE call *mcount@GOTPCREL(%rip)
#else
#define PROFCODE
#endif

#define EPILOGUE \
        .size	 REALNAME, .-REALNAME; \
        .section .note.GNU-stack,"",@progbits


#endif

#endif

