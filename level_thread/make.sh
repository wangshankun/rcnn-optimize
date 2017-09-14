#!/bin/sh
rm -rf *.o *.a test
gcc -g -DMAX_STACK_ALLOC=2048 -Wall -m64 -DF_INTERFACE_GFORT -fPIC -DSMP_SERVER -DNO_WARMUP -DMAX_CPU_NUMBER=8 -DASMNAME=sgemm_beta -DASMFNAME=sgemm_beta_ -DNAME=sgemm_beta_ -DCNAME=sgemm_beta -DCHAR_NAME=\"sgemm_beta_\" -DCHAR_CNAME=\"sgemm_beta\" -DNO_AFFINITY -I.. -UDOUBLE  -UCOMPLEX -c -UDOUBLE -UCOMPLEX ./gemm_beta.S -o sgemm_beta.o
gcc -g -DMAX_STACK_ALLOC=2048 -Wall -m64 -DF_INTERFACE_GFORT -fPIC -DSMP_SERVER -DNO_WARMUP -DMAX_CPU_NUMBER=8 -DASMNAME=sgemm_itcopy -DASMFNAME=sgemm_itcopy_ -DNAME=sgemm_itcopy_ -DCNAME=sgemm_itcopy -DCHAR_NAME=\"sgemm_itcopy_\" -DCHAR_CNAME=\"sgemm_itcopy\" -DNO_AFFINITY -I.. -UDOUBLE  -UCOMPLEX -c -UDOUBLE -UCOMPLEX ./gemm_tcopy_16.c -o sgemm_itcopy.o
gcc -g -DMAX_STACK_ALLOC=2048 -Wall -m64 -DF_INTERFACE_GFORT -fPIC -DSMP_SERVER -DNO_WARMUP -DMAX_CPU_NUMBER=8 -DASMNAME=sgemm_oncopy -DASMFNAME=sgemm_oncopy_ -DNAME=sgemm_oncopy_ -DCNAME=sgemm_oncopy -DCHAR_NAME=\"sgemm_oncopy_\" -DCHAR_CNAME=\"sgemm_oncopy\" -DNO_AFFINITY -I.. -UDOUBLE  -UCOMPLEX -c -UDOUBLE -UCOMPLEX ./gemm_ncopy_4.c -o sgemm_oncopy.o
gcc -g -DMAX_STACK_ALLOC=2048 -Wall -m64 -DF_INTERFACE_GFORT -fPIC -DSMP_SERVER -DNO_WARMUP -DMAX_CPU_NUMBER=8 -DASMNAME=sgemm_kernel -DASMFNAME=sgemm_kernel_ -DNAME=sgemm_kernel_ -DCNAME=sgemm_kernel -DCHAR_NAME=\"sgemm_kernel_\" -DCHAR_CNAME=\"sgemm_kernel\" -DNO_AFFINITY -I.. -UDOUBLE  -UCOMPLEX -c -UDOUBLE -UCOMPLEX ./sgemm_kernel_16x4_haswell.S -o sgemm_kernel.o

ar  -ru libspopenblas.a  sgemm_beta.o sgemm_itcopy.o sgemm_oncopy.o sgemm_kernel.o

ranlib  libspopenblas.a

gcc -O3 -w  -o  test thread_level3.c -lpthread -static -L./ -lspopenblas

