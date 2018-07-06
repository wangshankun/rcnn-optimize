#!/bin/sh
test_len=1048576
rmmod  /lib/modules/`uname -r`/extra/wr_fpga_ddr.ko
insmod /lib/modules/`uname -r`/extra/wr_fpga_ddr.ko dma_len=$test_len

sed -i "s/\(data_size = \)(\(.*\))/\1($test_len)/g" pyhardrock.py

rm   hardrock.so
gcc -O3 -Wall -ffast-math -ffunction-sections -fno-stack-protector  -fno-stack-check  -fPIC  -c -o hardrock.o user_so_wr_fpga_ddr.c

cc -Wl,--gc-sections -ffunction-sections  -fmerge-all-constants -fno-stack-protector  -o hardrock.so -shared hardrock.o
rm  hardrock.o

python ./pyhardrock.py
