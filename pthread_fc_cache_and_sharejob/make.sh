#!/bin/sh
#rm -rf mlockwait
#gcc mlockwait.c -g -O3 -mavx2 -mfma  -lpthread -o mlockwait
#gcc native_mlockwait.c -g -O3 -mavx2 -mfma  -lpthread -o native_mlockwait

#perf stat -e LLC-load-misses,LLC-store-misses,cache-misses,cache-references,L1-dcache-load-misses ./mlockwait
#perf stat -e LLC-load-misses,LLC-store-misses,cache-misses,cache-references,L1-dcache-load-misses ./native_mlockwait

rm -rf fc_pthread_nt
#gcc -w fc_pthread_nt.c -g -O3 -lpthread -o fc_pthread_nt
#gcc -w native_mlockwait.c -g -O3 -lpthread -o native_mlockwait

gcc -w fc_pthread_nt.c -g -O3 -mavx2 -mfma -lpthread  -o fc_pthread_nt

#gcc -w s.c -g -O3 -mavx2 -mfma  -I /opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas -lpthread -lgfortran  -o s

#gcc my.c  -O3 -g -mavx2 -mfma -o my -w -I /opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas  -lgfortran
#gcc org.c  -O3  -g -mavx2 -mfma -o org -w -I /opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas  -lgfortran
#gcc mxn_fc_block_8x8.c  -O3 -g -mavx2 -mfma -o mxn_fc_block_8x8 -w -I /opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas  -lgfortran
